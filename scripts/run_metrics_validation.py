"""
Performance and Risk Analytics Validation Script for Step 6.

Loads real processed NIFTY 50 market data, generates technical features, strategy signals,
and backtest outputs, computes complete PerformanceMetrics, and reports risk-adjusted
returns, drawdown recovery, and round-trip trade statistics.
"""

import datetime
import logging
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest import Backtester, run_buy_and_hold_benchmark
from src.features import add_features
from src.metrics import compare_performance, evaluate_backtest
from src.storage import StorageManager
from src.strategies.breakout import generate_breakout_signals
from src.strategies.momentum import generate_momentum_signals
from src.strategies.moving_average import generate_ma_crossover_signals

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("MetricsValidation")


def run_metrics_validation(
    symbol: str = "^NSEI",
    start_date: str = "2015-01-01",
    end_date: str = None,
    interval: str = "1d",
    initial_capital: float = 100000.0,
    cost_rate: float = 0.0005,
    slippage_rate: float = 0.0005,
):
    """Execute performance and risk analytics validation on real NIFTY 50 backtests.

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
        Starting capital.
    cost_rate : float, default 0.0005
        Transaction cost rate.
    slippage_rate : float, default 0.0005
        Slippage rate.
    """
    if end_date is None:
        end_date = datetime.date.today().strftime("%Y-%m-%d")

    print("\n" + "=" * 90)
    print(" QUANTMARKET STEP 6: PERFORMANCE & RISK ANALYTICS VALIDATION ")
    print("=" * 90)
    print(f"Target Instrument : NIFTY 50 ({symbol})")
    print(f"Dataset Range     : {start_date} to {end_date}")
    print(f"Initial Capital   : ${initial_capital:,.2f}")
    print(f"Frictions         : Costs = {cost_rate*10000:.1f} bps, Slippage = {slippage_rate*10000:.1f} bps")
    print("=" * 90 + "\n")

    storage_mgr = StorageManager()

    # Step 1: Load dataset & compute features
    logger.info(f"Loading processed dataset for {symbol}...")
    if not storage_mgr.dataset_exists(symbol, start_date, end_date, interval, tier="processed"):
        print(f"[ERROR]: Processed dataset not found for {symbol}. Please run Step 2/3 validation first.")
        sys.exit(1)

    df_raw = storage_mgr.load_dataset(symbol, start_date, end_date, interval, tier="processed")
    df_feat = add_features(df_raw)

    # Step 2: Execute Backtests from Step 5
    backtester = Backtester(
        initial_capital=initial_capital,
        position_size=1.0,
        transaction_cost_rate=cost_rate,
        slippage_rate=slippage_rate,
        allow_fractional=False,
    )

    logger.info("Executing Strategy 1: Moving-Average Crossover (20/50)...")
    ma_signals = generate_ma_crossover_signals(df_feat, fast_window=20, slow_window=50)
    ma_res = backtester.run(df_feat, ma_signals)

    logger.info("Executing Strategy 2: Momentum (20d)...")
    mom_signals = generate_momentum_signals(df_feat, lookback=20, threshold=0.0)
    mom_res = backtester.run(df_feat, mom_signals)

    logger.info("Executing Strategy 3: 20-Day Breakout...")
    bo_signals = generate_breakout_signals(df_feat, lookback=20)
    bo_res = backtester.run(df_feat, bo_signals)

    logger.info("Executing Buy & Hold Benchmark...")
    bnh_res = run_buy_and_hold_benchmark(
        df_feat,
        initial_capital=initial_capital,
        transaction_cost_rate=cost_rate,
        slippage_rate=slippage_rate,
        allow_fractional=False,
    )

    # Step 3: Evaluate Performance Metrics
    logger.info("Evaluating PerformanceMetrics objects...")
    ma_metrics = evaluate_backtest(ma_res)
    mom_metrics = evaluate_backtest(mom_res)
    bo_metrics = evaluate_backtest(bo_res)
    bnh_metrics = evaluate_backtest(bnh_res)

    metrics_map = {
        "MA Crossover (20/50)": ma_metrics,
        "Momentum (20d, Thresh=0)": mom_metrics,
        "20-Day Breakout": bo_metrics,
        "Buy & Hold Benchmark": bnh_metrics,
    }

    # Print Summary Comparison Table
    print("\n" + "=" * 115)
    print(" QUANTMARKET STRATEGY PERFORMANCE & RISK COMPARISON ")
    print("=" * 115)
    comp_df = compare_performance(metrics_map)
    print(comp_df.to_string(index=False))
    print("=" * 115 + "\n")

    # Print Drawdown Peak, Trough, and Recovery Analysis
    print("=" * 90)
    print(" MAXIMUM DRAWDOWN & RECOVERY ANALYSIS ")
    print("=" * 90)
    for name, m in metrics_map.items():
        peak_str = m.drawdown_peak_date.strftime("%Y-%m-%d") if m.drawdown_peak_date else "N/A"
        trough_str = m.drawdown_trough_date.strftime("%Y-%m-%d") if m.drawdown_trough_date else "N/A"
        rec_str = m.drawdown_recovery_date.strftime("%Y-%m-%d") if m.drawdown_recovery_date else "UNRECOVERED"

        print(f"\n[{name}]")
        print(f"  Max Drawdown : {m.max_drawdown*100:.2f}%")
        print(f"  Peak Date    : {peak_str}")
        print(f"  Trough Date  : {trough_str}")
        print(f"  Recovery Date: {rec_str}")

    print("\n" + "=" * 90)

    # Print Descriptive Findings
    best_cagr_name = max(metrics_map.keys(), key=lambda k: metrics_map[k].cagr)
    best_sharpe_name = max(metrics_map.keys(), key=lambda k: metrics_map[k].sharpe_ratio)
    lowest_dd_name = max(metrics_map.keys(), key=lambda k: metrics_map[k].max_drawdown)  # Closest to 0.0
    highest_turnover_name = max(metrics_map.keys(), key=lambda k: metrics_map[k].total_executed_trades)

    print(" DESCRIPTIVE FINDINGS & HIGHLIGHTS ")
    print("=" * 90)
    print(f"  - Highest CAGR Strategy     : {best_cagr_name} ({metrics_map[best_cagr_name].cagr*100:.2f}%)")
    print(f"  - Highest Sharpe Ratio      : {best_sharpe_name} ({metrics_map[best_sharpe_name].sharpe_ratio:.2f})")
    print(f"  - Lowest Maximum Drawdown   : {lowest_dd_name} ({metrics_map[lowest_dd_name].max_drawdown*100:.2f}%)")
    print(f"  - Highest Turnover Strategy : {highest_turnover_name} ({metrics_map[highest_turnover_name].total_executed_trades} trades, ${metrics_map[highest_turnover_name].total_transaction_costs:,.2f} costs)")
    print("=" * 90 + "\n")

    print("[SUCCESS]: Performance and risk analytics evaluation completed with zero parameter optimization.")
    return metrics_map


if __name__ == "__main__":
    run_metrics_validation()
