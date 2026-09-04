"""
Pipeline Validation Script for Step 2.

Downloads real historical market data for NIFTY 50 (^NSEI) from yfinance, saves raw data,
cleans the dataset using DataCleaner, saves processed data to Parquet storage, reloads it,
validates schema integrity, and prints summary stats.
"""

import argparse
import datetime
import logging
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_cleaner import DataCleaner
from src.data_loader import DataLoader
from src.storage import StorageManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("PipelineValidation")


def run_pipeline_validation(
    symbol: str = "^NSEI",
    start_date: str = "2015-01-01",
    end_date: str = None,
    interval: str = "1d",
    is_index: bool = True,
):
    """Execute end-to-end data pipeline validation on real market data.

    Parameters
    ----------
    symbol : str, default "^NSEI"
        Market ticker symbol.
    start_date : str, default "2015-01-01"
        Configurable historical start date.
    end_date : str, default None
        Configurable end date (None defaults to today's date).
    interval : str, default "1d"
        Sampling interval.
    is_index : bool, default True
        If True, volume is treated as optional.
    """
    if end_date is None:
        end_date = datetime.date.today().strftime("%Y-%m-%d")

    print("\n" + "=" * 80)
    print(" QUANTMARKET STEP 2: DATA PIPELINE VALIDATION ")
    print("=" * 80)
    print(f"Target Instrument : NIFTY 50 ({symbol})")
    print(f"Configured Start  : {start_date}")
    print(f"Configured End    : {end_date}")
    print(f"Interval          : {interval}")
    print(f"Is Index          : {is_index}")
    print("=" * 80 + "\n")

    storage_mgr = StorageManager()
    loader = DataLoader(storage_manager=storage_mgr)
    cleaner = DataCleaner()

    # Step 1: Download / Fetch Raw Data
    logger.info(f"1. Fetching historical data for {symbol}...")
    raw_df = loader.fetch_data(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
        use_cache=False,  # Force fresh fetch for validation
    )
    print(f"[RAW DATA SUMMARY]: Downloaded {len(raw_df)} rows from yfinance.")

    # Save raw data explicitly
    raw_path = storage_mgr.save_dataset(
        df=raw_df,
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
        tier="raw",
    )
    print(f"[RAW STORAGE]: Saved raw dataset to {raw_path}")

    # Step 2: Clean and Validate Data
    logger.info("2. Cleaning and validating market data...")
    cleaned_df, validation_res = cleaner.clean(raw_df, is_index=is_index)
    print(f"[CLEANING RESULT]: Retained {len(cleaned_df)} rows after cleaning.")

    # Step 3: Save Processed Data to Parquet
    logger.info("3. Saving processed dataset to Parquet storage...")
    processed_path = storage_mgr.save_dataset(
        df=cleaned_df,
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
        tier="processed",
    )
    print(f"[PROCESSED STORAGE]: Saved processed dataset to {processed_path}")

    # Step 4: Reload Processed Data from Parquet
    logger.info("4. Reloading processed dataset from Parquet storage...")
    reloaded_df = storage_mgr.load_dataset(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
        tier="processed",
    )

    # Step 5: Final Validation Check on Loaded Dataset
    logger.info("5. Running validation checks on reloaded dataset...")
    reloaded_val = cleaner.validate(reloaded_df, is_index=is_index)

    # Print requested validation metrics
    print("\n" + "=" * 80)
    print(" LOADED DATASET VALIDATION REPORT ")
    print("=" * 80)
    print(f"Number of rows   : {len(reloaded_df)}")
    print(f"Columns          : {list(reloaded_df.columns)}")
    print(f"Date Range       : {reloaded_df.index.min()} to {reloaded_df.index.max()}")
    print("\nMissing-Value Counts:")
    for col, count in reloaded_val.missing_counts.items():
        print(f"  - {col}: {count}")

    print("\nFirst 3 rows:")
    print(reloaded_df.head(3))

    print("\nLast 3 rows:")
    print(reloaded_df.tail(3))

    print("\nValidation Pass Status : ", "PASSED" if reloaded_val.is_valid else "FAILED")
    if reloaded_val.issues:
        print("Detected Issues / Notes:")
        for issue in reloaded_val.issues:
            print(f"  - {issue}")
    else:
        print("Detected Issues / Notes: None (Clean dataset)")
    print("=" * 80 + "\n")

    return reloaded_df, reloaded_val


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run QuantMarket Data Pipeline Validation")
    parser.add_argument("--symbol", type=str, default="^NSEI", help="Ticker symbol")
    parser.add_argument("--start", type=str, default="2015-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--interval", type=str, default="1d", help="Interval (1d, 1wk)")
    args = parser.parse_args()

    run_pipeline_validation(
        symbol=args.symbol,
        start_date=args.start,
        end_date=args.end,
        interval=args.interval,
    )
