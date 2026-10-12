# Import Libraries
import os
import json
import numpy as np
# "online" (Yahoo Finance + FRED, then cached) | "cache" (./data only) | "synthetic" (code test only)
DATA_SOURCE = "online"
FAST_MODE = False      # True = quick end-to-end check with reduced settings

CONFIG = dict(
    # ---------------- 1. DATA ----------------
    TICKERS=["SPY", "QQQ", "IWM", "EEM", "TLT", "GLD"],
    START="2005-01-01",        # GLD starts Nov-2004; broad dollar index starts Jan-2006
    END=None,                # None = latest available date
    FRED_SERIES=["DGS10", "DTB3", "T10Y3M", "BAA10Y",
                 "T10YIE", "VIXCLS", "DTWEXBGS"],  # daily series only
    # daily FRED values are used one calendar day after their date
    FRED_LAG_DAYS=1,
    DATA_DIR="data",
    RESULTS_DIR="results",

    # ---------------- 2. FEATURES ----------------
    # Feature names per block (see §3 for definitions). Remove or add names here to change the inputs.
    # "OTHER_RET5" expands into the 5-day returns of the five other ETFs.
    FEATURES={
        "R": ["R_ret1", "R_ret5", "R_ret21", "R_ret63", "R_mom_12_1", "R_ret21_voladj", "R_overnight"],
        "T": ["T_rsi14", "T_macd_hist", "T_px_sma50", "T_px_sma200", "T_boll_pctb20", "T_volume_z20"],
        "V": ["V_rv5", "V_rv21", "V_ewma", "V_downside21", "V_ratio_5_63", "V_parkinson21"],
        "X": ["OTHER_RET5", "X_vix", "X_vix_chg5", "X_usd_ret21", "X_corr_spy_tlt63"],
        "M": ["M_term_10y3m", "M_credit_baa10y", "M_credit_chg21", "M_y10_chg21", "M_breakeven10"],
    },
    BLOCK_SETS={"B1": ["R"], "B2": ["R", "T"], "B3": ["R", "T", "V"],
                "B4": ["R", "T", "V", "X"], "B5": ["R", "T", "V", "X", "M"]},
    # cap standardised features at +/- 8 (linear and deep models)
    CLIP_Z=8.0,

    # ---------------- 3. TARGET ----------------
    HORIZONS=[1, 5, 21],          # prediction horizons h (trading days)

    # ---------------- 4. WALK-FORWARD ----------------
    # "rolling" = non-anchored (fixed-length training window) | "expanding" = anchored
    WALK_FORWARD="rolling",
    TRAIN_YEARS=5,                   # length of the rolling training window
    VAL_YEARS=2,                   # validation window just before each test block
    TEST_START_YEAR=2013,            # first out-of-sample year
    # retrain logistic and tree models every N years (= test block length)
    ML_RETRAIN_YEARS=1,
    DL_RETRAIN_YEARS=2,               # retrain deep models every N years
    # extra gap on top of the label length (purge = h + 1 + embargo)
    EMBARGO_DAYS=5,
    # logistic/trees: refit on train+validation before predicting the test block
    REFIT_ON_TRAIN_VAL=True,

    # ---------------- 5. MODELS ----------------
    BENCHMARKS=["BUY_HOLD", "TREND_SMA200", "TSMOM_12M", "VOL_SCALED"],
    ML_MODELS=["LOGIT", "RF", "XGB"],
    DL_MODELS=["MLP", "LSTM"],
    ML_BLOCK_SETS=["B1", "B2", "B3", "B4", "B5"],
    DL_BLOCK_SETS=["B1", "B2", "B3", "B4", "B5"],
    ML_GRIDS={
        "LOGIT": [{"C": 0.05}, {"C": 1.0}],
        "RF":    [{"max_depth": 4, "min_samples_leaf": 50}, {"max_depth": 8, "min_samples_leaf": 50}],
        "XGB":   [{"max_depth": 2, "learning_rate": 0.05}, {"max_depth": 4, "learning_rate": 0.05}],
    },
    RF_TREES=200,
    XGB_TREES=300,
    DL_GRID=[{"units": 32, "dropout": 0.2}, {"units": 64, "dropout": 0.3}],
    # look-back window for deep models (the MLP gets the same window, flattened)
    SEQ_LEN=20,
    N_SEEDS=3,
    DL_EPOCHS=60,
    DL_PATIENCE=8,
    DL_BATCH=64,
    DL_LR=1e-3,

    # ---------------- 6. TRADING RULE AND THRESHOLDS ----------------
    # "long_short" (+1/0/-1) | "long_flat" (+1/0, i.e. pS = 0)
    POSITION_MODE="long_short",
    REBALANCE="horizon",          # "horizon" = review the position every h days | "daily"
    PL_GRID=(0.50, 0.70),       # search range for the long threshold pL
    PS_GRID=(0.30, 0.50),       # search range for the short threshold pS
    THRESH_STEP=0.01,
    # Step 1: pL >= 0.5 + c/m_h ; pS <= 0.5 - c/m_h - b*h/(504*m_h)
    USE_COST_BOUNDS=True,
    # Step 2: discard pairs in the market on < 5% of validation days
    MIN_EXPOSURE=0.05,
    # Step 3: neighbourhood of +/- 1 grid step (0.01) for the plateau rule
    STABILITY_RADIUS=1,
    # Step 4: stay flat if no pair has a positive validation net Sharpe
    NO_EDGE_FLAT=True,
    THRESH_OBJECTIVE="sharpe",        # "sharpe" (net of costs) | "accuracy"

    # ---------------- 7. COSTS ----------------
    # one-way cost per unit of position change (long->short = 2 units)
    COST_BPS=5.0,
    BORROW_FEE=0.005,              # annual fee for short positions
    COST_GRID_BPS=[0, 1, 2, 5, 10, 20, 30],
    BORROW_GRID=[0.0, 0.005, 0.02],
    # re-choose thresholds on validation at each cost level
    REOPTIMISE_THRESHOLDS_IN_COST_GRID=True,
    # e.g. 0.10: same volatility-targeting overlay for ALL strategies
    VOL_TARGET=None,
    VOL_BENCH_TARGET=0.10,            # target volatility of the VOL_SCALED benchmark
    ANN=252,

    # ---------------- 8. EVALUATION ----------------
    # robustness: fixed (pL, pS) pairs
    FIXED_THRESHOLDS=[(0.50, 0.50), (0.55, 0.45), (0.60, 0.40)],
    SUBPERIODS={"2013-2019": ("2013", "2019"), "2020-2021": ("2020", "2021"),
                "2022": ("2022", "2022"), "2023-": ("2023", "2100")},
    BOOT_REPS=2000,
    BOOT_BLOCK=20,
    DSR_THRESHOLD=0.95,
    RANDOM_STATE=42,
)

if FAST_MODE:   # reduced settings for a quick check of the whole pipeline
    CONFIG.update(HORIZONS=[1, 5], TEST_START_YEAR=2016, ML_RETRAIN_YEARS=2, DL_RETRAIN_YEARS=3,
                  ML_BLOCK_SETS=["B1", "B3", "B5"], DL_BLOCK_SETS=["B5"], N_SEEDS=1,
                  DL_EPOCHS=12, DL_PATIENCE=3, BOOT_REPS=300, COST_GRID_BPS=[0, 5, 20],
                  ML_GRIDS={k: v[:1] for k, v in CONFIG["ML_GRIDS"].items()}, DL_GRID=CONFIG["DL_GRID"][:1])

os.makedirs(CONFIG["DATA_DIR"], exist_ok=True)
np.random.seed(CONFIG["RANDOM_STATE"])
ANN = CONFIG["ANN"]
print(json.dumps(CONFIG, indent=1, default=str)[:3000])
