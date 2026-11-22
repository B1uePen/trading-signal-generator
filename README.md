Algorithmic Trading Signal Generation for BTC/USDT Perpetual Futures using Policy Gradient Optimization, Temporal-Difference Learning, and Genetic Algorithms under strict Out-Of-Sample backtesting.

A Python quantitative research framework for direct-learning trading signal generation on the **BTC/USDT perpetual futures** market. Instead of predicting next-candle prices via MSE regression, the system trains three distinct direct-learning agents—**Direct Policy Gradient**, **Q-Learning**, and a **Genetic Algorithm**—to output discrete trading decisions (`Long`, `Short`, or `Cash`) directly against risk-adjusted net returns inside a realistic **Triple-Barrier (`1:3`)** execution simulator.

---

## Key Highlights

- **Zero-Lookahead Data Pipeline:** Processes nearly 5 years of 1-minute historical data (`2021-12-01` to `2026-09-30`, plus a 6-day warm-up buffer from `2021-11-25`) aggregated into **42,360 hourly candles** across bear, accumulation, bull, and deep-drawdown (`-53.77%`) market regimes.
- **Realistic Publication Lags:** Applies strict time shifts before resampling (`+5 min` for Open Interest, Top-Trader Long/Short Ratio, and taker flows; `+60 min` for Deribit DVOL; `+1440 min` for VIX and DXY), cleans zero-OI artifacts via forward fill, and fits Z-score standardization (`[-5σ, +5σ]` clipping) exclusively on the training split.
- **34 Multi-Modal Features (10% Calendar Cap):** Combines order-flow microstructure (`12`), futures metrics & OI-Price quadrants (`11`), macro & volatility indicators (`5`), and calendar cycles (`6`) with a hard `10%` L1 weight budget cap on time features so at least `90%` of every decision relies on market signals.
- **Triple-Barrier Execution & Portfolio Backtester:** Enters on the next candle open (`t + 1`) with asymmetric `1:3` risk-to-reward barriers (`Stop-Loss = -1.5%`, `Take-Profit = +4.5%`, vertical horizon `H = 36h`), 1-minute tie-breaking for ambiguous hourly candles, `0.10%` round-trip fee + slippage, real 8-hour funding payments (`00:00`, `08:00`, `16:00 UTC`), zero overlapping positions, conviction-based `1x / 2x` position sizing, and liquidation protection at `Equity <= 0`.
- **Self-Contained Mathematical Core:** Built from scratch in pure Python without external deep learning or reinforcement learning frameworks, validated by **16 automated unit tests**.

---

## Chronological Data Split

| Split | Period (UTC) | Duration | Hourly Candles | Role |
| :--- | :--- | :---: | :---: | :--- |
| **Train** | `2021-12-01` – `2024-03-31` | 28 months | `20,448` | Model optimization & scaler statistics |
| **Validation** | `2024-04-01` – `2024-09-30` | 6 months | `4,392` | Early stopping, elite selection & `2x` conviction thresholds |
| **Out-of-Sample Test** | `2024-10-01` – `2026-09-30` | 24 months | `17,520` | Blind evaluation with frozen parameters |

**Data Sources:** Binance Spot & USD-M Futures API (OHLCV, taker buy volume, Open Interest, Funding Rate, Premium Index, Top-20% Trader Long/Short Ratio), Deribit Public API (`DVOL`), and Yahoo Finance / FRED / Alternative.me (`VIX`, `DXY`, `Fear & Greed Index`), stored in binary Apache Parquet (`.parquet`) format.

---

## Models & Benchmarks

### Trained Direct-Learning Agents
1. **Direct Policy Gradient (Adam):** Two-layer neural network (`16` hidden `tanh` units, `Softmax` output over `{Long, Short, Cash}`) optimizing expected Sharpe ratio with entropy and L2 regularization via analytical backpropagation and Adam (`lr = 0.008`, early stopping on Validation).
2. **Q-Learning (Step-by-Step Value Estimation):** Linear action-value estimator predicting expected net trade payoff for `Long` and `Short` separately, trained with mini-batch (`256`) updates, decaying learning rate, and L2 weight decay.
3. **Genetic Algorithm:** Population-based evolutionary search (`150` chromosomes, `40` generations, correlation-seeded initialization, `15%` elitism, 5-way tournament selection, crossover, and Gaussian mutation) optimizing risk-adjusted fitness with a drawdown penalty (`β = 2.5`).

### Baseline Benchmarks
- **B0 — Buy & Hold BTC:** Passive holding across the 24-month test window (`0.10%` round-trip fee).
- **B1 — Naive Always-Long (1:3):** Continuously re-enters `Long` (`1x`) after every barrier exit without market filtering.
- **B2 — Random Monte-Carlo (1:3):** Average of `1,000` random trading simulations (`10%` Long / `10%` Short / `80%` Cash; `75%` `1x` / `25%` `2x`), used to compute empirical `p-values`.

---

## Out-of-Sample Test Results (`2024-10-01` – `2026-09-30`)

| Model / Benchmark | Net PnL | CAGR | Sharpe | Sortino | MaxDD (from peak) | Lowest PnL (from start) | Win Rate | Profit Factor | Trades | p-value |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Direct Policy Gradient** | **+109.27%** | **+44.66%** | **+4.15** | **+6.44** | 25.60% | -8.41% | **42.1%** | **1.30** | 278 | **0.0020** |
| **Genetic Algorithm** | +97.75% | +40.62% | +2.21 | +3.20 | 25.15% | -6.53% | 40.1% | 1.12 | 686 | 0.0160 |
| **Q-Learning** | +65.56% | +28.67% | +2.07 | +3.28 | **23.32%** | **-2.36%** | 39.9% | 1.14 | 474 | 0.0190 |
| **B0: Buy & Hold BTC** | +31.46% | +14.66% | +0.53 | +0.75 | 53.77% | -8.34% | 100.0% | 1.50 | 1 | 0.0839 |
| **B1: Always-Long (1:3)** | -100.00% | -100.00% | -3.42 | -4.22 | 100.00% | -100.00% | 34.5% | 0.86 | 744 | 0.7013 |
| **B2: Random MC (1:3)** | -63.44% | -59.54% | -2.26 | -2.88 | 84.12% | — | 35.6% | 0.88 | 564 | 0.4376 |

### Key Takeaways
- All three direct-learning models achieved statistically significant outperformance (`p < 0.02`), delivering **2.1×–3.5× higher net return** than Buy & Hold while cutting peak-to-trough drawdown by more than half (`23.32%–25.60%` vs `53.77%`).
- None of the trained agents ever dropped below `-8.41%` relative to initial capital (`-2.36%` for Q-Learning), meaning their reported MaxDD occurred strictly from accumulated profit peaks.
- **Direct Policy Gradient** achieved the strongest directional balance (`+48.69%` across `217` Long trades and `+60.59%` across `61` selective Short trades) and highest expectancy on high-conviction `2x` entries (`+0.604%` per trade).
- Feature weight analysis converged across all three models on the same top market drivers: Top-Trader Long/Short Ratio (`lsr_top_traders`), 24h normalized CVD (`cvd_change_24h_norm`), Deribit `dvol_close`, `funding_rate`, `vix`, and `premium_index`.

---

## Architecture Overview

- **Data Pipeline:** Loads 1-minute Parquet market data, applies real publication lags, cleans zero-OI artifacts, resamples to 1-hour candles, builds the 34-feature matrix, and performs leak-free standardization.
- **Triple-Barrier Simulator & Backtester:** Vectorized SL/TP/horizon barrier engine with 1-minute ambiguity resolution, funding & fee accounting, sequential single-position execution, dynamic `1x / 2x` sizing, and portfolio metrics.
- **Direct-Learning Agents:** Self-contained implementations of Direct Policy Gradient (with Adam optimizer), Q-Learning, and Genetic Algorithm with the `10%` calendar feature weight cap.
- **Experiment Runner (`run_experiment.py`):** Runs the full out-of-sample evaluation, `1,000`-iteration Monte-Carlo benchmark, Long/Short & `1x/2x` trade decomposition, SL:TP risk-reward sensitivity study (`1:1`, `1:2`, `1:3`, `1.5%:4.5%`), and exports tables + 300 DPI plots to `results/`.
- **Verification Suite (`test_coursework.py`):** `16` unit tests covering analytical vs. numerical gradient checks (`< 1e-5`), strict zero-lookahead invariance, non-overlapping trade execution, extreme outlier clipping (`1000σ`), NaN/zero-OI recovery, and liquidation handling.

