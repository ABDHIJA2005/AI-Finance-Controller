"""
Signal Engine Validation Script for Step 4.

Loads real processed NIFTY 50 market data and STEP 3 technical features,
runs all three initial trading strategies (MA Crossover, Momentum, N-Day Breakout),
and reports signal event statistics, latest diagnostic states, and sample event records.
"""

import datetime
import logging
import sys
from pathlib import Path

import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features import add_features
from src.storage import StorageManager
from src.strategies.breakout import generate_breakout_signals
from src.strategies.momentum import generate_momentum_signals
from src.strategies.moving_average import generate_ma_crossover_signals

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("SignalValidation")


def run_signal_validation(
    symbol: str = "^NSEI",
    start_date: str = "2015-01-01",
    end_date: str = None,
    interval: str = "1d",
):
    """Execute signal engine validation on real processed NIFTY 50 data.

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
    print(" QUANTMARKET STEP 4: SIGNAL ENGINE VALIDATION ")
    print("=" * 80)
    print(f"Target Instrument : NIFTY 50 ({symbol})")
    print(f"Dataset Range     : {start_date} to {end_date}")
    print("=" * 80 + "\n")

    storage_mgr = StorageManager()

    # Step 1: Load processed dataset & compute STEP 3 features
    logger.info(f"Loading processed dataset for {symbol}...")
    if not storage_mgr.dataset_exists(symbol, start_date, end_date, interval, tier="processed"):
        print(f"[ERROR]: Processed dataset not found for {symbol}. Please run Step 2/3 validation first.")
        sys.exit(1)

    df_raw = storage_mgr.load_dataset(symbol, start_date, end_date, interval, tier="processed")
    df_feat = add_features(df_raw)
    rows_count = len(df_feat)
    print(f"[DATASET LOADED]: {rows_count} rows with computed technical features.\n")

    # --- Strategy 1: Moving Average Crossover (20 / 50) ---
    logger.info("Executing Strategy 1: Moving-Average Crossover (SMA-20 / SMA-50)...")
    ma_signals = generate_ma_crossover_signals(df_feat, fast_window=20, slow_window=50)
    ma_entries = int((ma_signals["signal_event"] == 1.0).sum())
    ma_exits = int((ma_signals["signal_event"] == -1.0).sum())
    ma_latest_state = ma_signals["position_state"].iloc[-1]
    ma_latest_event = ma_signals["signal_event"].iloc[-1]

    # --- Strategy 2: Momentum Strategy (N=20, Threshold=0.0) ---
    logger.info("Executing Strategy 2: Momentum Strategy (Lookback=20, Threshold=0.0)...")
    mom_signals = generate_momentum_signals(df_feat, lookback=20, threshold=0.0)
    mom_entries = int((mom_signals["signal_event"] == 1.0).sum())
    mom_exits = int((mom_signals["signal_event"] == -1.0).sum())
    mom_latest_state = mom_signals["position_state"].iloc[-1]
    mom_latest_event = mom_signals["signal_event"].iloc[-1]

    # --- Strategy 3: N-Day Breakout Strategy (Lookback=20) ---
    logger.info("Executing Strategy 3: N-Day Breakout Strategy (Lookback=20)...")
    bo_signals = generate_breakout_signals(df_feat, lookback=20)
    bo_entries = int((bo_signals["signal_event"] == 1.0).sum())
    bo_exits = int((bo_signals["signal_event"] == -1.0).sum())
    bo_latest_state = bo_signals["position_state"].iloc[-1]
    bo_latest_event = bo_signals["signal_event"].iloc[-1]

    # Print Summary Table
    print("\n" + "=" * 80)
    print(" STRATEGY SIGNAL EVENT SUMMARY ")
    print("=" * 80)
    print(f"{'Strategy Name':<30} | {'Entries (+1)':<12} | {'Exits (-1)':<12} | {'Latest State':<12} | {'Latest Event':<12}")
    print("-" * 88)
    print(f"{'MA Crossover (20/50)':<30} | {ma_entries:<12d} | {ma_exits:<12d} | {ma_latest_state:<12.0f} | {ma_latest_event:<12.0f}")
    print(f"{'Momentum (N=20, Threshold=0)':<30} | {mom_entries:<12d} | {mom_exits:<12d} | {mom_latest_state:<12.0f} | {mom_latest_event:<12.0f}")
    print(f"{'N-Day Breakout (N=20)':<30} | {bo_entries:<12d} | {bo_exits:<12d} | {bo_latest_state:<12.0f} | {bo_latest_event:<12.0f}")
    print("=" * 88 + "\n")

    # Print Sample Events for Each Strategy
    print("=" * 80)
    print(" SAMPLE STRATEGY SIGNAL EVENTS ")
    print("=" * 80)

    print("\n[Moving Average Crossover] First 3 Entry Events:")
    ma_entry_rows = ma_signals[ma_signals["signal_event"] == 1.0].head(3)
    print(ma_entry_rows[["fast_ma", "slow_ma", "position_state", "signal_event"]])

    print("\n[Momentum Strategy] First 3 Entry Events:")
    mom_entry_rows = mom_signals[mom_signals["signal_event"] == 1.0].head(3)
    print(mom_entry_rows[["momentum", "threshold", "position_state", "signal_event"]])

    print("\n[N-Day Breakout Strategy] First 3 Entry Events:")
    bo_entry_rows = bo_signals[bo_signals["signal_event"] == 1.0].head(3)
    print(bo_entry_rows[["close", "previous_high", "previous_low", "position_state", "signal_event"]])

    # Print Latest Strategy Diagnostics
    print("\n" + "=" * 80)
    print(" LATEST STRATEGY DIAGNOSTIC VALUES ")
    print("=" * 80)
    latest_dt = df_feat.index[-1]
    print(f"Timestamp : {latest_dt}")
    print(f"MA Crossover Diagnostics : Fast MA (20) = {ma_signals['fast_ma'].iloc[-1]:.2f}, Slow MA (50) = {ma_signals['slow_ma'].iloc[-1]:.2f}")
    print(f"Momentum Diagnostics     : 20d Momentum = {mom_signals['momentum'].iloc[-1]:.4f} ({mom_signals['momentum'].iloc[-1]*100:.2f}%)")
    print(f"Breakout Diagnostics     : Close = {bo_signals['close'].iloc[-1]:.2f}, Prev High (20) = {bo_signals['previous_high'].iloc[-1]:.2f}, Prev Low (20) = {bo_signals['previous_low'].iloc[-1]:.2f}")
    print("=" * 80 + "\n")

    print("[SUCCESS]: Signal Engine generation complete. Zero trade execution or profitability modeling performed.")
    return ma_signals, mom_signals, bo_signals


if __name__ == "__main__":
    run_signal_validation()
