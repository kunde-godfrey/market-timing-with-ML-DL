# Does Depth Pay? Machine Learning and Deep Learning for Multi-Asset Market Timing

**MScFE 690 Capstone Project – WorldQuant University**

This repository contains the code for a study that asks a simple question: **do machine learning (ML) and deep learning (DL) models give useful market-timing signals for a multi-asset ETF portfolio, once the comparison is fair and trading costs are paid?**

We predict whether each of six ETFs will rise over the next week. The predicted probabilities are turned into long, short or flat positions, and the results are measured after costs. Simple benchmarks, logistic regression, tree models and deep networks all get the same data, test periods, tuning budget and costs.

**Short answer: no.** None of the models predicts direction better than chance. Every learning model earns a lower net Sharpe ratio than buy-and-hold and simple trend rules. This holds even with zero trading costs, at every horizon tested and with every design variant. See [Key findings](#key-findings).

---

## Contents

- [Research design](#research-design)
- [Key findings](#key-findings)
- [Repository structure](#repository-structure)
- [Getting started](#getting-started)
- [Configuration](#configuration)
- [Outputs](#outputs)
- [Reproducibility and good practice](#reproducibility-and-good-practice)
- [Limitations](#limitations)
- [Authors](#authors)
- [References](#references)

---

## Research design

### Asset universe

| Ticker | Asset class |
|---|---|
| SPY | US large-company stocks (S&P 500) |
| QQQ | US large technology stocks (Nasdaq-100) |
| IWM | US small-company stocks (Russell 2000) |
| EEM | Emerging-market stocks |
| TLT | Long-term US Treasury bonds (20+ years) |
| GLD | Gold |

Daily adjusted prices (open, high, low, close, volume) come from **Yahoo Finance**, with Stooq as a fallback. Seven **daily FRED** series are used, each lagged by one day so that it was known when the decision was made: DGS10, DTB3, T10Y3M, BAA10Y, T10YIE, VIXCLS and DTWEXBGS. The sample starts in January 2005, and the first date with a full set of features is February 2006.

### Primary specification (fixed before any result was seen)

| Choice | Setting |
|---|---|
| Target | Sign of the open-to-open return over the next **5 trading days** |
| Timing | Features at the close of day *t*; trade at the open of day *t*+1 |
| Pooling | **One model trained on all six ETFs together** (per-ETF scaling + ETF indicator variables) |
| Features | **34 per ETF**, in five blocks (see below) |
| Validation | **Rolling** walk-forward: 5-year training window, 2-year validation window, 2-year test blocks from 2013; purging and embargo of *h* + 1 + 5 days at each boundary |
| Trading rule | Long if $p_t > p_L$, short if $p_t < p_S$, flat otherwise; positions reviewed every 5 days |
| Costs | 5 bp one-way per trade; 0.5% a year borrowing fee on short positions; cash earns the 3-month T-bill rate |

All other choices are run only as **robustness checks**, each changing one element at a time. They cover horizons of 1, 10 and 21 days, separate per-ETF models, equity-only pooling, a "band" label and an 8-year training window.

### Feature blocks

| Block | # | Features |
|---|---|---|
| **R** – Returns | 7 | 1-, 5-, 21-, 63-day returns; 12-1 month momentum; volatility-adjusted 21-day return; overnight return |
| **T** – Technical | 6 | RSI(14); MACD histogram; price vs 50- and 200-day averages; Bollinger %B(20); volume z-score(20) |
| **V** – Volatility | 6 | Realised volatility (5, 21 days); EWMA volatility; downside volatility; volatility ratio 5/63; Parkinson high–low volatility |
| **X** – Cross-asset | 10 | 5-day returns of all six ETFs; VIX level and 5-day change; US dollar 21-day return; SPY–TLT 63-day correlation |
| **M** – Macro | 5 | Term spread (10y–3m); Baa–10y credit spread and its 21-day change; 21-day change in the 10-year yield; 10-year breakeven inflation |

The blocks are added one at a time (B1 = R, …, B5 = R+T+V+X+M), always on exactly the same dates. This measures what each type of information adds.

### Models

| Family | Models |
|---|---|
| Benchmarks | Buy-and-hold; 200-day moving-average trend rule; 12-month time-series momentum; volatility-scaled buy-and-hold (10% target) |
| Linear | Logistic regression (L2 penalty) |
| Trees | Random forest; XGBoost |
| Deep learning | **MLP** on a flattened 20-day window (the control: depth without time order); **LSTM** on the same 20-day window |

Every learning model gets **four hyperparameter configurations**, chosen by validation log-loss. Deep networks use early stopping and are averaged over three random seeds.

### Choosing the thresholds $p_L$ and $p_S$

The thresholds are chosen **on the validation period only**, separately for each ETF and fold, in four steps:

1. **Cost-aware limits:** $p_L \ge 0.5 + c/m_h$ and $p_S \le 0.5 - c/m_h - b\,h/(504\,m_h)$, where $m_h$ is the typical *h*-day move.
2. **Grid search:** every pair on a 0.01 grid, plus "never long" and "never short", is scored by its **net Sharpe ratio** on validation data.
3. **Stability rule:** each pair is scored by the average of its neighbourhood, to avoid lucky, fragile peaks.
4. **No-edge rule:** if no pair makes money on validation, the strategy stays in cash for the next test block.

### Evaluation

- **Prediction:** accuracy against the base rate, AUC, Brier score, Pesaran–Timmermann test, and Diebold–Mariano tests against logistic regression.
- **Economics:** net and gross Sharpe ratio, Sortino ratio, maximum drawdown, turnover, break-even costs and an equal-weight six-ETF portfolio.
- **Significance and selection bias:** block-bootstrap Sharpe-difference tests and the **deflated Sharpe ratio**, which counts every model version and threshold pair tried.
- **Robustness:**
  - a cost grid from 0 to 30 bp and borrowing fees from 0% to 2%;
  - sub-periods and VIX regimes;
  - threshold and rebalancing variants;
  - XGBoost block attribution;
  - the one-change-at-a-time experiments.

---

## Key findings

The test period runs from January 2013 to October 2026, about 3,460 days per ETF. All figures are for the primary specification and net of costs.

| Strategy | Mean AUC | Accuracy − base rate | Mean net Sharpe (6 ETFs) | Equal-weight portfolio Sharpe |
|---|---|---|---|---|
| Buy-and-hold | – | 0.0 pp | **0.46** | **0.70** |
| Volatility-scaled | – | 0.0 pp | 0.41 | 0.64 |
| Trend (SMA200) | 0.498 | −2.8 pp | 0.38 | 0.67 |
| TSMOM (12 months) | 0.494 | −2.9 pp | 0.38 | 0.67 |
| Logistic regression | 0.516 | −1.9 pp | 0.27 | 0.56 |
| LSTM | 0.501 | −1.8 pp | 0.28 | 0.46 |
| MLP | 0.506 | −3.0 pp | 0.22 | 0.40 |
| XGBoost | 0.496 | −1.6 pp | 0.21 | 0.34 |
| Random forest | 0.497 | −1.6 pp | 0.07 | 0.16 |

1. **No predictive skill.** AUC values lie between 0.50 and 0.52. Every model is less accurate than "always up", and no Pesaran–Timmermann test is significant.
2. **Depth does not help.** Tree and deep models do no better than logistic regression. The LSTM is no better than the MLP that sees the same data without time order.
3. **Information blocks add very little.** Going from B1 to B5 raises the AUC by 0.002 to 0.018, mostly through the cross-asset block. The gain is not larger for the non-linear models.
4. **Benchmarks win.** No model beats buy-and-hold on any ETF with statistical significance, and no strategy passes the deflated Sharpe ratio test at 0.95.
5. **Costs are not the cause.** The gap remains even at zero trading costs. The models sit in cash 53% to 69% of the time and miss much of a rising market.
6. **Where models help:** they have smaller drawdowns than buy-and-hold and lost much less in 2022. However, the simple trend rules give the same protection more cheaply.
7. **Robust null result.** The conclusion holds across all horizons, both pooling choices, both labels and both window lengths. Doubling the tuning budget did not change it.

---

## Repository structure

```
market-timing-with-ML-DL/
├── capstone_project_code.ipynb   # Full pipeline: data → features → models → backtest → evaluation
├── requirements.txt              # Python dependencies
├── README.md
├── data/                         # Cached input data (created on first online run)
│   ├── prices.csv                #   daily OHLCV for the six ETFs
│   └── fred_daily.csv            #   daily FRED series
└── results/                      # Created when the notebook runs
    ├── preds/<run_id>/           #   saved predictions per (experiment, model, ETF, block set)
    ├── chosen_thresholds.csv
    ├── specification_log.csv
    ├── xgb_block_attribution.csv
    └── all_results_<run_id>.xlsx #   all result tables in one workbook
```

The notebook follows the five layers of the methodology:

| Notebook section | Content |
|---|---|
| 0. Configuration | All research choices in one place |
| 1. Data layer | Download, cleaning, alignment, exploratory analysis |
| 2. Target | Trade timing and labels |
| 3. Features | Feature library, the 34 selected features, pooled panel |
| 4. Walk-forward | Rolling folds with purging and embargo |
| 5. Models | Benchmarks, logistic regression, RF, XGBoost, MLP, LSTM |
| 6. Backtest | Trading rule, costs, threshold selection |
| 7. Run | Runs all experiments (resumes where it stopped) |
| 8. Evaluation | Predictive and economic results, tests, costs, deflated Sharpe ratio |
| 9. Robustness | Sub-periods, VIX regimes, threshold variants, one-change experiments |
| 10. Summary | Answers by objective; Excel export |

---

## Getting started

### 1. Install

Python 3.10 or later is recommended.

```bash
git clone https://github.com/kunde-godfrey/market-timing-with-ML-DL.git
cd market-timing-with-ML-DL
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Get the data

On the first run, set `DATA_SOURCE = "online"` in the configuration cell. The notebook downloads prices from Yahoo Finance (or Stooq if Yahoo fails) and the FRED series, then saves them to `data/`. Later runs can use the cached files by setting `DATA_SOURCE = "cache"`; any value other than `"online"` reads from `data/`. This also guarantees that every run uses exactly the same data.

### 3. Run

Open `capstone_project_code.ipynb` in Jupyter, JupyterLab or Google Colab, and run all cells.

- **`RUN_SCOPE = "primary"`** runs the primary specification only: the block ablation, benchmarks, costs and tests. This takes a few hours on a laptop CPU, mostly for the deep models.
- **`RUN_SCOPE = "full"`** also runs the seven robustness experiments. Expect several more hours; the separate per-ETF deep models take the longest.

Every model run is saved as soon as it finishes. If the session stops, re-running section 7 skips the work already done.

---

## Configuration

All choices are made in the `CONFIG` dictionary in section 0, so the study can be changed without editing the code.

| Key | Meaning | Default |
|---|---|---|
| `PRIMARY` | Horizon, label, pooling, training years and block sets of the primary specification | `HORIZON=5`, `LABEL="sign"`, `POOLING="all"`, `TRAIN_YEARS=5` |
| `ROBUSTNESS` | One-change-at-a-time experiments | `h1`, `h10`, `h21`, `separate`, `equity_pool`, `label_band`, `train8` |
| `TICKERS`, `START`, `END` | Universe and sample | six ETFs, from 2005 to the latest date |
| `FEATURES`, `BLOCK_SETS` | Selected features and nested block sets | 34 features, B1–B5 |
| `WALK_FORWARD`, `VAL_YEARS`, `TEST_START_YEAR` | Validation design | `"rolling"`, 2, 2013 |
| `ML_GRIDS`, `DL_GRID`, `N_SEEDS` | Tuning budget | 4 configurations per model, 3 seeds |
| `POSITION_MODE`, `REBALANCE` | Trading rule | `"long_short"`, review every *h* days |
| `PL_GRID`, `PS_GRID`, `THRESH_STEP`, `MIN_EXPOSURE` | Threshold search | 0.50–0.70, 0.30–0.50, 0.01, 5% |
| `COST_BPS`, `BORROW_FEE` | Base-case frictions | 5 bp, 0.5% a year |
| `COST_GRID_BPS`, `BORROW_GRID` | Cost sensitivity | 0–30 bp; 0%, 0.5%, 2% |
| `BOOT_REPS`, `DSR_THRESHOLD` | Tests | 2,000 resamples; 0.95 |

---

## Outputs

| File | Content |
|---|---|
| `results/all_results_<run_id>.xlsx` | Master results, thresholds, block sets, costs, regimes, variants, robustness experiments |
| `results/chosen_thresholds.csv` | $p_L$, $p_S$, limits and validation Sharpe for every model, ETF and fold |
| `results/specification_log.csv` | Every configuration fitted, used for the deflated Sharpe ratio |
| `results/xgb_block_attribution.csv` | Drop in test AUC when each block is shuffled, by fold |

The `run_id` is a hash of the configuration and the data source. Changing any setting therefore starts a new, separate set of results instead of overwriting the old one.

---

## Reproducibility and good practice

The design follows the main lessons of the literature on backtesting:

- **No look-ahead:** features use data up to the close of *t*, trades happen at the open of *t*+1, and FRED data are lagged by one day.
- **No leakage:** splits are made by date, never at random. Purging and embargo remove overlapping labels at every boundary, and scaling uses training-window statistics only.
- **Fair comparison:** every model gets the same data, folds, tuning budget and costs.
- **No test-set snooping:** hyperparameters and trading thresholds are chosen on validation data only.
- **Full disclosure:** every version tried is logged, and the results are corrected with the deflated Sharpe ratio.
- **Fixed seeds and cached data**, so a run can be repeated exactly.

---

## Limitations

- The test period (2013–2026) is a single history, dominated by a long bull market.
- The six ETFs are among the most liquid and most closely watched assets in the world, so there is little room for mispricing.
- Costs are modelled as a proportional fee, which is realistic for small investors but not for very large orders.
- Only price-based and daily macro information is used; news, options or fundamental data might give different results.
- The tuning budget, although doubled, is still limited. The near-zero AUC suggests, however, that more tuning would not change the conclusion.

---

## Authors

- Wenskidly Charles
- [@kunde-godfrey](https://github.com/kunde-godfrey)

MScFE 690 Capstone, WorldQuant University, 2026.

This project is for academic research only and is **not investment advice**.

---

## References

- Bailey, D. H., & López de Prado, M. (2014). The deflated Sharpe ratio: Correcting for selection bias, backtest overfitting, and non-normality. *The Journal of Portfolio Management, 40*(5), 94–107.
- Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics, 13*(3), 253–263.
- Faber, M. T. (2007). A quantitative approach to tactical asset allocation. *The Journal of Wealth Management, 9*(4), 69–79.
- Fischer, T., & Krauss, C. (2018). Deep learning with long short-term memory networks for financial market predictions. *European Journal of Operational Research, 270*(2), 654–669.
- Gu, S., Kelly, B., & Xiu, D. (2020). Empirical asset pricing via machine learning. *The Review of Financial Studies, 33*(5), 2223–2273.
- Ledoit, O., & Wolf, M. (2008). Robust performance hypothesis testing with the Sharpe ratio. *Journal of Empirical Finance, 15*(5), 850–859.
- López de Prado, M. (2018). *Advances in financial machine learning*. Wiley.
- Moreira, A., & Muir, T. (2017). Volatility-managed portfolios. *The Journal of Finance, 72*(4), 1611–1644.
- Moskowitz, T. J., Ooi, Y. H., & Pedersen, L. H. (2012). Time series momentum. *Journal of Financial Economics, 104*(2), 228–250.
- Pesaran, M. H., & Timmermann, A. (1992). A simple nonparametric test of predictive performance. *Journal of Business & Economic Statistics, 10*(4), 461–465.
- Welch, I., & Goyal, A. (2008). A comprehensive look at the empirical performance of equity premium prediction. *The Review of Financial Studies, 21*(4), 1455–1508.
