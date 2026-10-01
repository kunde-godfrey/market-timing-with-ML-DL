import os
import json
import numpy as np

# Operational Mode
# "online" (Yahoo Finance + FRED, cached), "cache" (use ./data only), "synthetic" (code test only)
DATA_SOURCE = "online"
FAST_MODE = False         # True = quick end-to-end check with reduced settings

CONFIG = dict(
    TICKERS=["SPY", "QQQ", "IWM", "EEM", "TLT", "GLD"],
    START="2005-01-01",      # GLD starts Nov-2004; the broad dollar index starts Jan-2006
    END=None,              # None = today
    DATA_DIR="data",
    RESULTS_DIR="results",

    # Targets and horizons
    HORIZONS=,        # prediction horizons in trading days
    # extra gap (on top of the label length) between train/validation/test
    EMBARGO_DAYS=5,

    # Walk-forward design
    TEST_START_YEAR=2013,              # first out-of-sample year
    VAL_YEARS=2,                 # validation window just before each test block
    ML_RETRAIN_YEARS=1,                 # retrain logistic/trees every N years
    # retrain deep models every N years (computational budget)
    DL_RETRAIN_YEARS=2,

    # Information blocks (cumulative sets, added one at a time)
    BLOCK_SETS={"B1": ["R"], "B2": ["R", "T"], "B3": ["R", "T", "V"],
                "B4": ["R", "T", "V", "X"], "B5": ["R", "T", "V", "X", "M"]},
    ML_BLOCK_SETS=["B1", "B2", "B3", "B4", "B5"],
    # DL is expensive: price-only vs all information
    DL_BLOCK_SETS=["B1", "B5"],

    # Models
    BENCHMARKS=["BUY_HOLD", "TREND_SMA200", "TSMOM_12M", "VOL_SCALED"],
    ML_MODELS=["LOGIT", "RF", "XGB"],
    DL_MODELS=["MLP", "LSTM", "CNN"],
    # look-back window (days) for the deep models (MLP gets the same window, flattened)
    SEQ_LEN=20,
    N_SEEDS=3,                 # deep models: average of N random initialisations
    DL_EPOCHS=60,
    DL_PATIENCE=8,
    DL_BATCH=64,

    # Backtest
    POSITION_MODE="long_flat",       # "long_flat" or "long_short"
    COST_BPS=5.0,               # one-way cost per unit of position change, in basis points
    COST_GRID_BPS=,
    THRESHOLD_GRID=[round(x, 2) for x in np.arange(0.40, 0.605, 0.01)],
    # threshold chosen on validation by "sharpe" (net of costs) or "accuracy"
    THRESHOLD_OBJ="sharpe",
    # hysteresis around the threshold (e.g. 0.02) to reduce trading
    NO_TRADE_BAND=0.00,
    # e.g. 0.10 applies the same volatility-targeting overlay to ALL strategies
    VOL_TARGET=None,
    # target volatility for the VOL_SCALED benchmark (no return forecast)
    VOL_BENCH_TARGET=0.10,
    ANN=252,

    # Evaluation
    BOOT_REPS=2000,
    BOOT_BLOCK=20,
    RANDOM_STATE=42,
)

if FAST_MODE:
    CONFIG.update(HORIZONS=[1, 5], TEST_START_YEAR=2016, ML_RETRAIN_YEARS=2, DL_RETRAIN_YEARS=3,
                  ML_BLOCK_SETS=["B1", "B3", "B5"], DL_BLOCK_SETS=["B5"], N_SEEDS=1,
                  DL_EPOCHS=12, DL_PATIENCE=3, BOOT_REPS=300)

# Build the required environment folders automatically upon execution/import
os.makedirs(CONFIG["DATA_DIR"], exist_ok=True)
os.makedirs(os.path.join(CONFIG["RESULTS_DIR"], "preds"), exist_ok=True)

# Set global reproducibility seeds
np.random.seed(CONFIG["RANDOM_STATE"])
