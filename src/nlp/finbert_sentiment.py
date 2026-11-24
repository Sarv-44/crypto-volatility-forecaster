"""FinBERT financial sentiment analysis module with batching and caching.

Uses the pretrained 'ProsusAI/finbert' model from Hugging Face Transformers
to compute positive, negative, and neutral sentiment probabilities.
Calculates sentiment_score = positive_prob - negative_prob.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch

from src.utils.logger import get_logger

logger = get_logger("FinBERTSentiment")

DEFAULT_MODEL_NAME = "ProsusAI/finbert"
DEFAULT_CACHE_PATH = Path("data/raw/social/sentiment_cache.json")


class FinBERTSentimentAnalyzer:
    """Financial sentiment analyzer using Hugging Face's ProsusAI/finbert."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: Optional[str] = None,
        cache_file: Optional[Path] = DEFAULT_CACHE_PATH,
        enable_cache: bool = True,
    ):
        """Initializes the tokenizer, model, and sentiment cache.

        Args:
            model_name: Hugging Face model identifier (default: 'ProsusAI/finbert').
            device: 'cuda', 'cpu', or None (auto-detects).
            cache_file: Path to persistent JSON cache file.
            enable_cache: Whether to cache inference results.
        """
        self.model_name = model_name
        self.enable_cache = enable_cache
        self.cache_file = Path(cache_file) if cache_file else None
        self.memory_cache: Dict[str, Dict[str, float]] = {}

        if self.enable_cache and self.cache_file:
            self._load_cache()

        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = None
        self.model = None
        self.id2label = {}
        self.is_fallback = False

        self._load_model()

    def _load_cache(self) -> None:
        """Loads cached sentiment scores from disk."""
        if self.cache_file and self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    self.memory_cache = json.load(f)
                logger.info("Loaded %d cached sentiment entries from %s", len(self.memory_cache), self.cache_file)
            except Exception as e:
                logger.warning("Failed to load sentiment cache: %s", str(e))
                self.memory_cache = {}

    def _save_cache(self) -> None:
        """Saves cached sentiment scores to disk."""
        if self.enable_cache and self.cache_file:
            try:
                self.cache_file.parent.mkdir(parents=True, exist_ok=True)
                with open(self.cache_file, "w", encoding="utf-8") as f:
                    json.dump(self.memory_cache, f, indent=2)
            except Exception as e:
                logger.warning("Failed to save sentiment cache: %s", str(e))

    def _load_model(self) -> None:
        """Attempts to load FinBERT tokenizer and weights from Hugging Face."""
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            logger.info("Loading %s onto %s...", self.model_name, self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()

            # Store mapping: e.g. {0: 'positive', 1: 'negative', 2: 'neutral'}
            raw_labels = self.model.config.id2label
            self.id2label = {idx: str(name).lower() for idx, name in raw_labels.items()}
            logger.info("FinBERT model loaded successfully. Label map: %s", self.id2label)
        except Exception as e:
            logger.warning(
                "Unable to load FinBERT model (%s): %s. "
                "Enabling rule-based financial lexicon fallback mode for offline/restricted environments.",
                self.model_name,
                str(e),
            )
            self.is_fallback = True

    def _get_cache_key(self, text: str) -> str:
        """Generates MD5 hash for cached text lookups."""
        return hashlib.md5(text.strip().lower().encode("utf-8")).hexdigest()

    def _fallback_sentiment(self, text: str) -> Dict[str, float]:
        """Simple rule-based financial sentiment heuristic when FinBERT is offline.
        
        Labels output as rule-based fallback.
        """
        lower = text.lower()
        pos_words = ["surge", "gain", "bull", "rally", "high", "breakout", "profit", "pump", "record", "jump"]
        neg_words = ["crash", "drop", "bear", "plunge", "low", "dump", "fall", "liquidation", "ban", "loss"]

        pos_count = sum(w in lower for w in pos_words)
        neg_count = sum(w in lower for w in neg_words)

        if pos_count > neg_count:
            pos_prob = min(0.4 + 0.2 * pos_count, 0.95)
            neg_prob = 0.05
            neu_prob = round(1.0 - pos_prob - neg_prob, 4)
        elif neg_count > pos_count:
            neg_prob = min(0.4 + 0.2 * neg_count, 0.95)
            pos_prob = 0.05
            neu_prob = round(1.0 - pos_prob - neg_prob, 4)
        else:
            pos_prob = 0.10
            neg_prob = 0.10
            neu_prob = 0.80

        score = round(pos_prob - neg_prob, 4)
        return {
            "positive": round(pos_prob, 4),
            "negative": round(neg_prob, 4),
            "neutral": round(neu_prob, 4),
            "sentiment_score": score,
        }

    def predict_single(self, text: Optional[str]) -> Dict[str, float]:
        """Scores a single text input.

        Args:
            text: Headline or post string.

        Returns:
            Dict containing positive, negative, neutral probabilities and sentiment_score.
        """
        # Handle missing, null, or empty text
        if text is None or not isinstance(text, str) or not text.strip() or pd.isna(text):
            return {
                "positive": 0.0,
                "negative": 0.0,
                "neutral": 1.0,
                "sentiment_score": 0.0,
            }

        cleaned = text.strip()
        key = self._get_cache_key(cleaned)

        if self.enable_cache and key in self.memory_cache:
            return self.memory_cache[key]

        # Use batch inference for single item
        result = self.predict_batch([cleaned])[0]
        return result

    def predict_batch(self, texts: List[Union[str, Any]], batch_size: int = 32) -> List[Dict[str, float]]:
        """Scores a list of texts in batches with caching.

        Args:
            texts: List of text strings.
            batch_size: Batch size for model inference.

        Returns:
            List of dicts with 'positive', 'negative', 'neutral', and 'sentiment_score'.
        """
        results: List[Optional[Dict[str, float]]] = [None] * len(texts)
        unprocessed_indices: List[int] = []
        unprocessed_texts: List[str] = []

        # Check cache and handle invalid/missing text first
        for i, raw_text in enumerate(texts):
            if raw_text is None or not isinstance(raw_text, str) or not raw_text.strip() or pd.isna(raw_text):
                results[i] = {
                    "positive": 0.0,
                    "negative": 0.0,
                    "neutral": 1.0,
                    "sentiment_score": 0.0,
                }
                continue

            cleaned = raw_text.strip()
            key = self._get_cache_key(cleaned)

            if self.enable_cache and key in self.memory_cache:
                results[i] = self.memory_cache[key]
            else:
                unprocessed_indices.append(i)
                unprocessed_texts.append(cleaned)

        if not unprocessed_texts:
            return [r for r in results if r is not None]

        # If model is in fallback mode
        if self.is_fallback or self.model is None or self.tokenizer is None:
            for idx, text_item in zip(unprocessed_indices, unprocessed_texts):
                score_dict = self._fallback_sentiment(text_item)
                results[idx] = score_dict
                if self.enable_cache:
                    self.memory_cache[self._get_cache_key(text_item)] = score_dict
            self._save_cache()
            return [r for r in results if r is not None]

        # FinBERT model inference in batches
        for start_idx in range(0, len(unprocessed_texts), batch_size):
            chunk = unprocessed_texts[start_idx : start_idx + batch_size]
            chunk_indices = unprocessed_indices[start_idx : start_idx + batch_size]

            try:
                inputs = self.tokenizer(
                    chunk,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt",
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

                for j, prob_vec in enumerate(probs):
                    # Map labels using id2label
                    pred_dict = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
                    for label_idx, prob_val in enumerate(prob_vec):
                        label_name = self.id2label.get(label_idx, f"class_{label_idx}")
                        if label_name in pred_dict:
                            pred_dict[label_name] = round(float(prob_val), 4)

                    # sentiment_score = positive - negative
                    score = round(pred_dict["positive"] - pred_dict["negative"], 4)
                    pred_dict["sentiment_score"] = score

                    orig_idx = chunk_indices[j]
                    text_str = chunk[j]
                    results[orig_idx] = pred_dict

                    if self.enable_cache:
                        self.memory_cache[self._get_cache_key(text_str)] = pred_dict

            except Exception as batch_err:
                logger.warning("Batch inference error: %s. Using fallback for current chunk.", str(batch_err))
                for j, text_str in enumerate(chunk):
                    fallback_res = self._fallback_sentiment(text_str)
                    results[chunk_indices[j]] = fallback_res
                    if self.enable_cache:
                        self.memory_cache[self._get_cache_key(text_str)] = fallback_res

        self._save_cache()
        return [r for r in results if r is not None]

    def analyze_dataframe(
        self,
        df: pd.DataFrame,
        text_col: str = "text",
        batch_size: int = 32,
    ) -> pd.DataFrame:
        """Scores an entire DataFrame containing a text column.

        Adds columns:
            - sentiment_positive
            - sentiment_negative
            - sentiment_neutral
            - sentiment_score
        """
        if text_col not in df.columns:
            raise ValueError(f"Column '{text_col}' not found in DataFrame.")

        df = df.copy()
        texts = df[text_col].tolist()
        scores = self.predict_batch(texts, batch_size=batch_size)

        df["sentiment_positive"] = [s["positive"] for s in scores]
        df["sentiment_negative"] = [s["negative"] for s in scores]
        df["sentiment_neutral"] = [s["neutral"] for s in scores]
        df["sentiment_score"] = [s["sentiment_score"] for s in scores]

        return df
