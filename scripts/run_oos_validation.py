"""
Out-of-Sample Testing & Robustness Analysis Validation Script for Step 7.

Loads real processed NIFTY 50 market data, calculates point-in-time features and signals,
partitions the time-series chronologically into Development (2015-2021), Validation (2022-2023),
and Out-of-Sample (2024-2026) periods, executes period-isolated backtests with fresh $100,000
capital, and reports performance degradation and signal stability statistics.
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

from src.backtest import Backtester
from src.features import add_features
from src.oos import compare_robustness, run_multi_period_backtest, split_chronologically
from src.storage import StorageManager
from src.strategies.breakout import generate_breakout_signals
from src.strategies.momentum import generate_momentum_signals
from src.strategies.moving_average import generate_ma_crossover_signals

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("OOSValidation")


def run_oos_validation(
    symbol: str = "^NSEI",
    start_date: str = "2015-01-01",
    end_date: str = None,
    interval: str = "1d",
    initial_capital: float = 100000.0,
    cost_rate: float = 0.0005,
    slippage_rate: float = 0.0005,
):
    """Execute Out-of-Sample testing and robustness analysis on real NIFTY 50 data.

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
    initial_capital : float, default 100000.0
        Starting capital per period partition.
    cost_rate : float, default 0.0005
        Transaction cost rate.
    slippage_rate : float, default 0.0005
        Slippage rate.
    """
    if end_date is None:
        end_date = datetime.date.today().strftime("%Y-%m-%d")

    print("\n" + "=" * 95)
    print(" QUANTMARKET STEP 7: OUT-OF-SAMPLE TESTING & ROBUSTNESS ANALYSIS ")
    print("=" * 95)
    print(f"Target Instrument     : NIFTY 50 ({symbol})")
    print(f"Full Dataset Range    : {start_date} to {end_date}")
    print(f"Fresh Period Capital  : ${initial_capital:,.2f}")
    print(f"Execution Frictions   : Costs = {cost_rate*10000:.1f} bps, Slippage = {slippage_rate*10000:.1f} bps")
    print("=" * 95 + "\n")

    storage_mgr = StorageManager()

    # Step 1: Load dataset & compute features on full chronological series
    logger.info(f"Loading processed dataset for {symbol}...")
    if not storage_mgr.dataset_exists(symbol, start_date, end_date, interval, tier="processed"):
        print(f"[ERROR]: Processed dataset not found for {symbol}. Please run Step 2/3 validation first.")
        sys.exit(1)

    df_raw = storage_mgr.load_dataset(symbol, start_date, end_date, interval, tier="processed")
    df_feat = add_features(df_raw)

    # Step 2: Generate signals on full chronological series (Fixed Parameters)
    logger.info("Generating strategy signals on full series (Fixed Pre-Specified Parameters)...")
    ma_signals = generate_ma_crossover_signals(df_feat, fast_window=20, slow_window=50)
    mom_signals = generate_momentum_signals(df_feat, lookback=20, threshold=0.0)
    bo_signals = generate_breakout_signals(df_feat, lookback=20)

    # Print Chronological Partition Summary
    partitions = split_chronologically(df_feat, ma_signals)
    print("=" * 80)
    print(" DATASET CHRONOLOGICAL PARTITION SUMMARY ")
    print("=" * 80)
    for name, part in partitions.items():
        start_str = part.start_date.strftime("%Y-%m-%d") if part.rows_count > 0 else "N/A"
        end_str = part.end_date.strftime("%Y-%m-%d") if part.rows_count > 0 else "N/A"
        print(f"{name:<16} : {start_str} to {end_str} ({part.rows_count} trading days)")
    print("=" * 80 + "\n")

    # Step 3: Run Multi-Period Backtests for Strategies & Benchmark
    backtester = Backtester(
        initial_capital=initial_capital,
        position_size=1.0,
        transaction_cost_rate=cost_rate,
        slippage_rate=slippage_rate,
        allow_fractional=False,
    )

    logger.info("1. Running multi-period backtest for MA Crossover (20/50)...")
    ma_multi = run_multi_period_backtest(df_feat, ma_signals, "MA Crossover (20/50)", backtester)

    logger.info("2. Running multi-period backtest for Momentum (20d)...")
    mom_multi = run_multi_period_backtest(df_feat, mom_signals, "Momentum (20d)", backtester)

    logger.info("3. Running multi-period backtest for 20-Day Breakout...")
    bo_multi = run_multi_period_backtest(df_feat, bo_signals, "20-Day Breakout", backtester)

    logger.info("4. Running multi-period backtest for Buy & Hold Benchmark...")
    bnh_multi = run_multi_period_backtest(df_feat, ma_signals, "Buy & Hold Benchmark", backtester, is_benchmark=True)

    multi_results = {
        "MA Crossover (20/50)": ma_multi,
        "Momentum (20d)": mom_multi,
        "20-Day Breakout": bo_multi,
        "Buy & Hold Benchmark": bnh_multi,
    }

    # Print Out-of-Sample Performance Table
    print("=" * 125)
    print(" OUT-OF-SAMPLE (OOS: 2024-2026) PERFORMANCE SUMMARY ")
    print("=" * 125)
    oos_rows = []
    for name, mres in multi_results.items():
        oos = mres.oos_metrics
        oos_rows.append(
            {
                "Strategy": name,
                "OOS Total Return": f"{oos.total_return*100:.2f}%",
                "OOS CAGR": f"{oos.cagr*100:.2f}%",
                "OOS Sharpe": f"{oos.sharpe_ratio:.2f}",
                "OOS Sortino": f"{oos.sortino_ratio:.2f}",
                "OOS MaxDD": f"{oos.max_drawdown*100:.2f}%",
                "OOS Trades": oos.total_executed_trades,
                "OOS WinRate": f"{oos.win_rate*100:.1f}%",
                "OOS ProfitFactor": f"{oos.profit_factor:.2f}",
                "OOS Costs": f"${oos.total_transaction_costs:,.2f}",
            }
        )
    print(pd.DataFrame(oos_rows).to_string(index=False))
    print("=" * 125 + "\n")

    # Print Multi-Period Robustness Comparison Table
    print("=" * 135)
    print(" MULTI-PERIOD ROBUSTNESS COMPARISON (DEV vs VAL vs OOS) ")
    print("=" * 135)
    robust_df = compare_robustness(multi_results)
    print(robust_df.to_string(index=False))
    print("=" * 135 + "\n")

    # Print Degradation Breakdown
    print("=" * 90)
    print(" PERFORMANCE DEGRADATION ANALYSIS (OOS vs DEVELOPMENT) ")
    print("=" * 90)
    for name, mres in multi_results.items():
        deg = mres.degradation
        print(f"\n[{name}]")
        print(f"  Delta CAGR (OOS - Dev)         : {deg['delta_cagr']*100:+.2f}%")
        print(f"  Delta Sharpe (OOS - Dev)       : {deg['delta_sharpe']:+.2f}")
        print(f"  Delta Max Drawdown (OOS - Dev) : {deg['delta_max_drawdown']*100:+.2f}%")
        print(f"  Delta Win Rate (OOS - Dev)     : {deg['delta_win_rate']*100:+.1f}%")
        print(f"  Delta Profit Factor            : {deg['delta_profit_factor']:+.2f}")
    print("=" * 90 + "\n")

    # Print Signal Stability & Market Exposure Summary
    print("=" * 90)
    print(" SIGNAL STABILITY & MARKET EXPOSURE STATISTICS ")
    print("=" * 90)
    for name, mres in multi_results.items():
        print(f"\n[{name}]")
        for p_name in ["Development", "Validation", "Out-of-Sample"]:
            stab = mres.stability[p_name]
            print(f"  {p_name:<15} : Entries = {stab['entries_count']:<3d}, Exits = {stab['exits_count']:<3d}, Time in Market = {stab['market_exposure_pct']*100:.1f}%")
    print("=" * 90 + "\n")

    print("=" * 90)
    print(" RESEARCH INTEGRITY STATEMENT ")
    print("=" * 90)
    print("The out-of-sample period (2024-2026) is treated as held-out evaluation data under a fixed,")
    print("pre-specified strategy specification. Strategy parameters were NOT optimized or tuned using")
    print("observations from the out-of-sample period. Results remain descriptive historical evidence.")
    print("=" * 90 + "\n")

    return multi_results


if __name__ == "__main__":
    run_oos_validation()
