# Design System & Direction: Quantitative Crypto Volatility Forecaster

**Author:** Senior Product Designer & Front-End Engineer  
**Objective:** Deliver an authentic, human-designed institutional quantitative research terminal. Eliminate AI-generated tropes (neon glows, generic bento grids, emojis, interchangeable cards, robotic buzzwords) in favor of restrained, editorial, and functional financial software design.

---

## 1. Design Principles

1. **Information Density over Decorative Fluff:** Every square pixel must serve analytical utility. Negative space is functional separation, not empty decorative padding.
2. **Hairline Precision over Shadows:** No drop shadows, ambient blurs, or glow effects. Visual hierarchy is established via 1px crisp borders, subtle tonal shifts, and typography scale.
3. **Authentic Editorial Finance Voice:** Clear, factual, precise financial terminology. No hyperbole ("seamless", "unlock", "supercharge").
4. **Typographic Rigor:** Tight negative tracking on display headings paired with strict tabular monospaced alignment (`font-variant-numeric: tabular-nums`) for all temporal and numerical data.

---

## 2. Restrained Palette (Tokens)

| Token Name | Value | Purpose |
|---|---|---|
| `--color-bg` | `#090b10` | Base background: deep obsidian slate with zero chromatic glare |
| `--color-surface` | `#10131a` | Primary container surface for analytical cards |
| `--color-surface-subtle` | `#161a24` | Inset tables, form fields, and secondary blocks |
| `--color-surface-hover` | `#1b212e` | Interactive hover states |
| `--color-border` | `#1f2533` | 1px hairline separation dividers |
| `--color-border-active` | `#2f384d` | Focused borders, active tabs, and highlighted rows |
| `--color-text` | `#f1f3f7` | High-contrast body and numerical data (WCAG AAA) |
| `--color-text-muted` | `#86909f` | Secondary labels, descriptions, and axis text |
| `--color-text-subtle` | `#525c6c` | Metadata timestamps and unit indicators |
| `--color-accent` | `#2563eb` | Institutional cobalt blue: focal action buttons, selection indicators |
| `--color-positive` | `#10b981` | Restrained emerald for positive returns and bullish shifts |
| `--color-negative` | `#f43f5e` | Restrained crimson for negative returns and drawdowns |
| `--color-warning` | `#f59e0b` | Muted amber for volatility warnings |

*Rule:* Zero radial gradients, zero glowing text-shadows, zero multi-color badges.

---

## 3. Typography Hierarchy

- **Primary UI / Display:** `Inter`, `-apple-system`, `BlinkMacSystemFont`, `"Segoe UI"`, sans-serif.
  - Headings: `letter-spacing: -0.025em`, font-weight `600`, sentence/uppercase case with purpose.
- **Data / Quantitative:** `JetBrains Mono`, `IBM Plex Mono`, `monospace`.
  - Applied to all timestamps, prices, percentages, metric values, regression coefficients, and matrices.
  - Required CSS: `font-variant-numeric: tabular-nums;`
- **Scale:**
  - `Hero / Primary Forecast`: `26px` (`1.625rem`), font-weight `600`, tabular-nums.
  - `Section Title`: `12px` (`0.75rem`), uppercase, font-weight `600`, `letter-spacing: 0.05em`.
  - `Body / Table Cells`: `12px` (`0.75rem`), font-weight `400`.
  - `Caption / Metadata`: `11px` (`0.6875rem`), font-weight `400`.

---

## 4. Spacing Scale & Border Radius

- **Grid:** 4px / 8px baseline. Standard paddings: `8px`, `12px`, `16px`, `20px`.
- **Border Radius:** Architectural and sharp:
  - Cards, tables, and canvases: `4px` (`border-radius: 4px`)
  - Input fields, buttons, and select boxes: `3px` (`border-radius: 3px`)
  - Status pills / badges: `2px` (`border-radius: 2px`)
  - *Strictly prohibited:* Generic `rounded-2xl` or pill-shaped cards.

---

## 5. Layout Architecture (Asymmetric & Functional)

- **Header:** Dense ticker ribbon with live exchange status, pair indicator, and clean utility buttons.
- **Top Metrics Strip:** Asymmetric layout grouping:
  - Market Price & 24H Volatility (Historical anchor).
  - Primary Forecast Hero (Focal point: 4H Forward Volatility + Model selector + Regime).
  - Sentiment Composite (FinBERT NLP stream).
- **Analytical Split (65% / 35%):**
  - Left Stage: Interactive price canvas with Bollinger volatility envelopes + Aligned multi-factor matrix.
  - Right Stage: Live FinBERT headline inference sandbox with real-time class probability breakdown.
- **Empirical Validation Tabs / Panels:**
  - Full-width rigorous benchmark evaluation table with right-aligned numbers.
  - Two-column split for Multimodal Factor Ablation & Volatility-Targeted Risk Simulation.
  - Academic Capstone Report Drawer and high-resolution figure viewer with clean hairline borders.
