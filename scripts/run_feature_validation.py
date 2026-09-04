"""
Feature Engineering Validation Script for Step 3.

Loads the processed NIFTY 50 market dataset from Step 2 storage, executes the
add_features pipeline, validates indicator ranges, and verifies zero look-ahead bias.
"""

import datetime
import logging
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features import add_features
from src.storage import StorageManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("FeatureValidation")


def run_feature_validation(
    symbol: str = "^NSEI",
    start_date: str = "2015-01-01",
    end_date: str = None,
    interval: str = "1d",
):
    """Execute feature engineering validation on real processed NIFTY 50 data.

    Parameters
    ----------
    symbol : str, default "^NSEI"
        Ticker symbol.
    start_date : str, default "2015-01-01"
        Start date.
    end_date : str, default None
        End date (defaults to today's date if None).
    interval : str, default "1d"
        Sampling interval.
    """
    if end_date is None:
        end_date = datetime.date.today().strftime("%Y-%m-%d")

    print("\n" + "=" * 80)
    print(" QUANTMARKET STEP 3: FEATURE ENGINEERING VALIDATION ")
    print("=" * 80)
    print(f"Target Instrument : NIFTY 50 ({symbol})")
    print(f"Dataset Range     : {start_date} to {end_date}")
    print("=" * 80 + "\n")

    storage_mgr = StorageManager()

    # Step 1: Load processed dataset from Step 2
    logger.info(f"Loading processed dataset for {symbol} from Parquet storage...")
    if not storage_mgr.dataset_exists(symbol, start_date, end_date, interval, tier="processed"):
        print(f"[ERROR]: Processed dataset not found for {symbol}. Please run Step 2 validation first.")
        sys.exit(1)

    df_raw = storage_mgr.load_dataset(symbol, start_date, end_date, interval, tier="processed")
    rows_count = len(df_raw)
    cols_before = len(df_raw.columns)
    print(f"[DATASET LOADED]: {rows_count} rows, {cols_before} columns before features.")

    # Step 2: Compute features
    logger.info("Computing point-in-time market features...")
    df_feat = add_features(
        df_raw,
        sma_windows=(20, 50, 200),
        ema_windows=(20, 50),
        volatility_windows=(20, 50),
        rsi_period=14,
        atr_period=14,
        breakout_window=20,
    )
    cols_after = len(df_feat.columns)
    feature_cols = [c for c in df_feat.columns if c not in df_raw.columns]

    print(f"[FEATURES COMPUTED]: {len(feature_cols)} feature columns generated ({cols_before} -> {cols_after} columns).")
    print(f"Feature Columns: {feature_cols}\n")

    # Step 3: Display First 5 & Last 5 rows of selected features
    selected_cols = [
        "close",
        "return",
        "sma_20",
        "ema_20",
        "volatility_20",
        "rsi_14",
        "atr_14",
        "previous_high_20",
    ]
    disp_cols = [c for c in selected_cols if c in df_feat.columns]

    print("First 5 rows of selected features:")
    print(df_feat[disp_cols].head(5))
    print("\nLast 5 rows of selected features:")
    print(df_feat[disp_cols].tail(5))

    # Step 4: Missing-value counts for feature columns
    print("\nMissing-Value Counts in Feature Columns:")
    missing_summary = df_feat[feature_cols].isna().sum()
    for col, count in missing_summary.items():
        print(f"  - {col:20s}: {count:4d} NaNs (initial rolling window)")

    # Step 5: Sanity Checks on Indicator Ranges
    print("\n" + "=" * 80)
    print(" SANITY CHECKS & LOOK-AHEAD BIAS VERIFICATION ")
    print("=" * 80)

    # 1. RSI Range Check
    rsi_valid = df_feat["rsi_14"].dropna()
    rsi_min, rsi_max = rsi_valid.min(), rsi_valid.max()
    rsi_passed = (rsi_min >= 0.0) and (rsi_max <= 100.0)
    print(f"RSI Range Check      : [{rsi_min:.2f}, {rsi_max:.2f}] -> {'PASSED' if rsi_passed else 'FAILED'}")

    # 2. Volatility Check
    vol_valid = df_feat["volatility_20"].dropna()
    vol_min = vol_valid.min()
    vol_passed = vol_min >= 0.0
    print(f"Volatility Check     : Min = {vol_min:.4f} (Annualized) -> {'PASSED' if vol_passed else 'FAILED'}")

    # 3. ATR Check
    atr_valid = df_feat["atr_14"].dropna()
    atr_min = atr_valid.min()
    atr_passed = atr_min >= 0.0
    print(f"ATR Range Check      : Min = {atr_min:.2f} -> {'PASSED' if atr_passed else 'FAILED'}")

    # 4. Look-Ahead Shift Check
    # Verify previous_high_20 at index t equals rolling_high_20 at index t-1
    t_idx = 100
    today_prev_high = df_feat["previous_high_20"].iloc[t_idx]
    yesterday_rolling_high = df_feat["rolling_high_20"].iloc[t_idx - 1]
    lookahead_passed = abs(today_prev_high - yesterday_rolling_high) < 1e-6
    print(
        f"Look-Ahead Shift Check: previous_high_20(t) [{today_prev_high:.2f}] == rolling_high_20(t-1) [{yesterday_rolling_high:.2f}] -> {'PASSED' if lookahead_passed else 'FAILED'}"
    )

    all_passed = rsi_passed and vol_passed and atr_passed and lookahead_passed
    print(f"\nFinal Feature Pipeline Status : {'PASSED' if all_passed else 'FAILED'}")
    print("=" * 80 + "\n")

    return df_feat


if __name__ == "__main__":
    run_feature_validation()
