"""
Monte Carlo Risk Analysis Validation Script for Step 8.

Executes 5,000 empirical bootstrap simulations for MA Crossover, Momentum, 20-Day Breakout,
and Buy & Hold Benchmark across Full History (2015-2026) and Out-of-Sample (2024-2026) scopes.
Generates summary tables and visual figure charts saved under reports/figures/.
"""

import datetime
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest import Backtester, run_buy_and_hold_benchmark
from src.features import add_features
from src.monte_carlo import (
    MonteCarloConfig,
    MonteCarloResult,
    compare_monte_carlo_results,
    run_monte_carlo_simulation,
)
from src.oos import split_chronologically
from src.storage import StorageManager
from src.strategies.breakout import generate_breakout_signals
from src.strategies.momentum import generate_momentum_signals
from src.strategies.moving_average import generate_ma_crossover_signals

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("MonteCarloValidation")


def generate_monte_carlo_charts(
    mc_results: dict[str, MonteCarloResult],
    output_dir: Path,
):
    """Generate and save publication-quality Monte Carlo charts.

    Parameters
    ----------
    mc_results : dict[str, MonteCarloResult]
        Dictionary of MonteCarloResult objects.
    output_dir : Path
        Directory to save figures.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Chart 1: Sample Equity Paths & Percentile Bands
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (name, mc) in enumerate(mc_results.items()):
        if idx >= 4:
            break
        ax = axes[idx]
        paths = mc.simulated_paths  # shape (n_vis, path_len)

        # Plot 50 light background paths
        for p_i in range(min(50, paths.shape[0])):
            ax.plot(paths[p_i], color="lightgray", alpha=0.3, linewidth=0.8)

        # Calculate median, 5th, and 95th percentile paths over time
        median_path = np.median(paths, axis=0)
        p5_path = np.percentile(paths, 5, axis=0)
        p95_path = np.percentile(paths, 95, axis=0)

        ax.plot(median_path, color="#1f77b4", linewidth=2.0, label="Simulated Median Path")
        ax.plot(p5_path, color="#d62728", linestyle="--", linewidth=1.5, label="5th Percentile Outcome")
        ax.plot(p95_path, color="#2ca02c", linestyle="--", linewidth=1.5, label="95th Percentile Outcome")
        ax.axhline(mc.initial_capital, color="black", linestyle=":", label="Initial Capital ($100k)")

        ax.set_title(f"{name} ({mc.analysis_scope} - {mc.simulation_mode.upper()})\nSample Size = {mc.empirical_sample_size} Obs", fontsize=11, fontweight="bold")
        ax.set_xlabel("Sequence Step (Trades / Days)")
        ax.set_ylabel("Portfolio Value ($)")
        ax.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    paths_chart_path = output_dir / "monte_carlo_equity_paths.png"
    plt.savefig(paths_chart_path, dpi=300)
    plt.close()
    logger.info(f"Saved figure: {paths_chart_path}")

    # Chart 2: Terminal Wealth Distribution Histogram
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (name, mc) in enumerate(mc_results.items()):
        if idx >= 4:
            break
        ax = axes[idx]
        term_wealth = mc.all_terminal_wealth / 1000.0  # In thousands

        ax.hist(term_wealth, bins=50, color="#4c72b0", edgecolor="white", alpha=0.8)
        ax.axvline(mc.historical_final_equity / 1000.0, color="orange", linewidth=2.0, label=f"Hist Equity (${mc.historical_final_equity/1000:.1f}k)")
        ax.axvline(mc.terminal_wealth_median / 1000.0, color="blue", linewidth=2.0, linestyle="-", label=f"Sim Median (${mc.terminal_wealth_median/1000:.1f}k)")
        ax.axvline(mc.terminal_wealth_percentiles["5th"] / 1000.0, color="red", linewidth=1.5, linestyle="--", label=f"Sim 5th% (${mc.terminal_wealth_percentiles['5th']/1000:.1f}k)")
        ax.axvline(mc.terminal_wealth_percentiles["95th"] / 1000.0, color="green", linewidth=1.5, linestyle="--", label=f"Sim 95th% (${mc.terminal_wealth_percentiles['95th']/1000:.1f}k)")

        ax.set_title(f"{name}: Terminal Wealth Distribution (5,000 Runs)\nLoss Prob = {mc.prob_of_loss*100:.1f}%", fontsize=11, fontweight="bold")
        ax.set_xlabel("Terminal Portfolio Value ($ Thousands)")
        ax.set_ylabel("Frequency")
        ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    wealth_chart_path = output_dir / "monte_carlo_terminal_wealth.png"
    plt.savefig(wealth_chart_path, dpi=300)
    plt.close()
    logger.info(f"Saved figure: {wealth_chart_path}")

    # Chart 3: Maximum Drawdown Distribution Histogram
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (name, mc) in enumerate(mc_results.items()):
        if idx >= 4:
            break
        ax = axes[idx]
        max_dds = mc.all_max_drawdowns * 100.0  # Percentage

        ax.hist(max_dds, bins=50, color="#c44e52", edgecolor="white", alpha=0.8)
        ax.axvline(mc.historical_max_drawdown * 100.0, color="black", linewidth=2.0, label=f"Hist MaxDD ({mc.historical_max_drawdown*100:.1f}%)")
        ax.axvline(mc.max_drawdown_median * 100.0, color="blue", linewidth=2.0, label=f"Sim Median DD ({mc.max_drawdown_median*100:.1f}%)")
        ax.axvline(mc.max_drawdown_percentiles["5th"] * 100.0, color="red", linewidth=1.5, linestyle="--", label=f"Sim 5th% DD ({mc.max_drawdown_percentiles['5th']*100:.1f}%)")

        ax.set_title(f"{name}: Max Drawdown Distribution\nProb Worse DD = {mc.prob_drawdown_worse_than_historical*100:.1f}%", fontsize=11, fontweight="bold")
        ax.set_xlabel("Maximum Drawdown (%)")
        ax.set_ylabel("Frequency")
        ax.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    dd_chart_path = output_dir / "monte_carlo_max_drawdown.png"
    plt.savefig(dd_chart_path, dpi=300)
    plt.close()
    logger.info(f"Saved figure: {dd_chart_path}")


def run_monte_carlo_validation(
    symbol: str = "^NSEI",
    start_date: str = "2015-01-01",
    end_date: str = None,
    interval: str = "1d",
    initial_capital: float = 100000.0,
    cost_rate: float = 0.0005,
    slippage_rate: float = 0.0005,
):
    """Execute 5,000 Monte Carlo risk simulations on real NIFTY 50 backtests.

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
        Starting capital per simulation.
    cost_rate : float, default 0.0005
        Transaction cost rate.
    slippage_rate : float, default 0.0005
        Slippage rate.
    """
    if end_date is None:
        end_date = datetime.date.today().strftime("%Y-%m-%d")

    print("\n" + "=" * 95)
    print(" QUANTMARKET STEP 8: MONTE CARLO RISK ANALYSIS VALIDATION ")
    print("=" * 95)
    print(f"Target Instrument     : NIFTY 50 ({symbol})")
    print(f"Full Dataset Range    : {start_date} to {end_date}")
    print(f"Simulations Count     : 5,000 Iterations")
    print(f"Random Seed           : 42 (Reproducible)")
    print(f"Initial Capital       : ${initial_capital:,.2f}")
    print("=" * 95 + "\n")

    storage_mgr = StorageManager()

    # Step 1: Load dataset & compute features
    logger.info(f"Loading processed dataset for {symbol}...")
    if not storage_mgr.dataset_exists(symbol, start_date, end_date, interval, tier="processed"):
        print(f"[ERROR]: Processed dataset not found for {symbol}. Please run Step 2/3 validation first.")
        sys.exit(1)

    df_raw = storage_mgr.load_dataset(symbol, start_date, end_date, interval, tier="processed")
    df_feat = add_features(df_raw)

    # Step 2: Generate strategy signals
    ma_signals = generate_ma_crossover_signals(df_feat, fast_window=20, slow_window=50)
    mom_signals = generate_momentum_signals(df_feat, lookback=20, threshold=0.0)
    bo_signals = generate_breakout_signals(df_feat, lookback=20)

    backtester = Backtester(
        initial_capital=initial_capital,
        position_size=1.0,
        transaction_cost_rate=cost_rate,
        slippage_rate=slippage_rate,
        allow_fractional=False,
    )

    # Full History Backtests
    logger.info("Executing Full-History Backtests (2015-2026)...")
    ma_res_full = backtester.run(df_feat, ma_signals)
    mom_res_full = backtester.run(df_feat, mom_signals)
    bo_res_full = backtester.run(df_feat, bo_signals)
    bnh_res_full = run_buy_and_hold_benchmark(df_feat, initial_capital=initial_capital, transaction_cost_rate=cost_rate, slippage_rate=slippage_rate)

    # Out-of-Sample Partition Backtests (2024-2026)
    logger.info("Executing Out-of-Sample Partition Backtests (2024-2026)...")
    partitions = split_chronologically(df_feat, ma_signals)
    oos_df = partitions["Out-of-Sample"].df

    ma_oos_sig = partitions["Out-of-Sample"].signals
    mom_oos_sig = split_chronologically(df_feat, mom_signals)["Out-of-Sample"].signals
    bo_oos_sig = split_chronologically(df_feat, bo_signals)["Out-of-Sample"].signals

    ma_res_oos = backtester.run(oos_df, ma_oos_sig)
    mom_res_oos = backtester.run(oos_df, mom_oos_sig)
    bo_res_oos = backtester.run(oos_df, bo_oos_sig)

    # Step 3: Execute Monte Carlo Simulations (5,000 runs, Seed=42)
    mc_cfg = MonteCarloConfig(n_simulations=5000, initial_capital=initial_capital, random_seed=42)

    logger.info("Running 5,000 Monte Carlo simulations (Trade-Level Bootstrap)...")
    mc_ma_full = run_monte_carlo_simulation(ma_res_full, "MA Crossover (20/50)", mc_cfg, mode="trade_level", scope="Full History")
    mc_mom_full = run_monte_carlo_simulation(mom_res_full, "Momentum (20d)", mc_cfg, mode="trade_level", scope="Full History")
    mc_bo_full = run_monte_carlo_simulation(bo_res_full, "20-Day Breakout", mc_cfg, mode="trade_level", scope="Full History")

    # OOS Monte Carlo simulations
    mc_ma_oos = run_monte_carlo_simulation(ma_res_oos, "MA Crossover (20/50)", mc_cfg, mode="trade_level", scope="Out-of-Sample")
    mc_mom_oos = run_monte_carlo_simulation(mom_res_oos, "Momentum (20d)", mc_cfg, mode="trade_level", scope="Out-of-Sample")
    mc_bo_oos = run_monte_carlo_simulation(bo_res_oos, "20-Day Breakout", mc_cfg, mode="trade_level", scope="Out-of-Sample")

    mc_full_dict = {
        "MA Crossover (20/50)": mc_ma_full,
        "Momentum (20d)": mc_mom_full,
        "20-Day Breakout": mc_bo_full,
    }

    mc_oos_dict = {
        "MA Crossover (20/50) [OOS]": mc_ma_oos,
        "Momentum (20d) [OOS]": mc_mom_oos,
        "20-Day Breakout [OOS]": mc_bo_oos,
    }

    # Step 4: Print Comparative Tables
    print("=" * 135)
    print(" MONTE CARLO SIMULATION RESULTS : FULL HISTORY (2015-2026) - TRADE-LEVEL BOOTSTRAP ")
    print("=" * 135)
    comp_full = compare_monte_carlo_results(mc_full_dict)
    print(comp_full.to_string(index=False))
    print("=" * 135 + "\n")

    print("=" * 135)
    print(" MONTE CARLO SIMULATION RESULTS : OUT-OF-SAMPLE (2024-2026) - TRADE-LEVEL BOOTSTRAP ")
    print("=" * 135)
    comp_oos = compare_monte_carlo_results(mc_oos_dict)
    print(comp_oos.to_string(index=False))
    print("=" * 135 + "\n")

    # Print Sample Size Warnings
    print("=" * 110)
    print(" EMPIRICAL SAMPLE SIZE & LIMITATION REPORT (RESEARCH INTEGRITY AUDIT) ")
    print("=" * 110)
    print("This report distinguishes between Full-History and Out-of-Sample (OOS) Monte Carlo analyses.")
    print("The OOS period (2024-2026) represents a held-out dataset. Completed trades in this period serve")
    print("as the empirical basis for bootstrapping. Below is the validation of sample sufficiency:")
    print("-" * 110)

    for name, mc in mc_oos_dict.items():
        print(f"[{name}]")
        print(f"  Empirical Sample Size (Completed Trades) : {mc.empirical_sample_size}")
        if mc.empirical_sample_size < 30:
            print(f"  WARNING: Small empirical sample size ({mc.empirical_sample_size} completed trades) in Out-of-Sample period.")
            print(f"           The 5,000 simulations do NOT increase the underlying historical sample size.")
            print(f"           Resampling a small set of observations does not generate new historical information.")
            if "Breakout" in name:
                print(f"           CRITICAL LIMITATION: For the 20-Day Breakout strategy OOS sample ({mc.empirical_sample_size} trades),")
                print(f"           inference is highly constrained. 5,000 runs MUST NOT be presented as a larger dataset.")
            elif "Crossover" in name:
                print(f"           CRITICAL LIMITATION: For the MA Crossover strategy OOS sample ({mc.empirical_sample_size} trades),")
                print(f"           inference is virtually meaningless due to severe lack of independent trade observations.")
        else:
            print("  Sample size sufficient (>= 30 trades) for empirical resampling.")
        print()
    print("=" * 110 + "\n")

    # Step 5: Generate & Save Charts under reports/figures/
    output_dir = PROJECT_ROOT / "reports" / "figures"
    logger.info("Generating Monte Carlo visualization figures...")
    generate_monte_carlo_charts(mc_full_dict, output_dir)

    print("=" * 110)
    print(" QUANTMARKET RESEARCH INTEGRITY & METHODOLOGICAL DISCLAIMERS ")
    print("=" * 110)
    print("1. Full-History vs. OOS Distinction:")
    print("   - Full-History Monte Carlo analyses are based on the entire backtest range (2015-2026).")
    print("   - Out-of-Sample (OOS) Monte Carlo analyses evaluate risk distributions purely on the held-out")
    print("     and unseen OOS backtest period (2024-2026). OOS results are strictly limited by OOS trade counts.")
    print("2. Empirical Sample Size Boundedness:")
    print("   - Generating 5,000 bootstrap simulations does NOT increase the amount of historical information.")
    print("   - The simulations are resamples of the existing observations; the underlying sample size remains")
    print("     exactly equal to the empirical trade count (reported alongside all results).")
    print("3. Simulated Outcome Percentiles:")
    print("   - Percentile values (e.g., 5th, 50th, 95th) are simulated outcome percentiles under the bootstrap")
    print("     distribution. They are NOT statistical confidence intervals for future performance.")
    print("4. Serial Dependence & Bootstrap Assumptions:")
    print("   - Simple trade-level and daily return bootstrap resampling assumes independent and identically")
    print("     distributed (i.i.d.) observations. This method destroys and does NOT preserve serial dependence")
    print("     or autocorrelation present in historical returns or trade sequences.")
    print("5. Parameter Stability & Parameter Selection:")
    print("   - Random seed is fixed at 42 and simulation run count is fixed at 5,000 for strict reproducibility.")
    print("   - Strategy parameters are fixed and have NOT been modified or tuned using Monte Carlo outcomes.")
    print("   - Monte Carlo results are used strictly for risk assessment, NOT for selecting or tuning strategies.")
    print("=" * 110 + "\n")

    return mc_full_dict, mc_oos_dict


if __name__ == "__main__":
    run_monte_carlo_validation()
