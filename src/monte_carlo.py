"""
Monte Carlo Risk Analysis Module.

Provides empirical bootstrap resampling (trade-level and daily return) and block bootstrap
resampling to evaluate portfolio path distributions, terminal wealth outcome percentiles,
and maximum drawdown risk without assuming normal return distributions.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.backtest import BacktestResult
from src.metrics import RoundTripTrade, match_round_trip_trades

logger = logging.getLogger(__name__)


@dataclass
class MonteCarloConfig:
    """Configuration container for Monte Carlo simulations.

    Attributes
    ----------
    n_simulations : int, default 5000
        Number of bootstrap simulation iterations.
    initial_capital : float, default 100000.0
        Starting portfolio capital.
    random_seed : int, default 42
        Explicit random seed for strict numerical reproducibility.
    block_size : Optional[int], default None
        Block size for block bootstrap resampling (None for simple empirical bootstrap).
    n_visualize : int, default 100
        Number of simulated paths to store for chart plotting.
    """

    n_simulations: int = 5000
    initial_capital: float = 100000.0
    random_seed: int = 42
    block_size: Optional[int] = None
    n_visualize: int = 100


@dataclass
class MonteCarloResult:
    """Dataclass holding Monte Carlo simulation summary statistics and risk metrics.

    Attributes
    ----------
    strategy_name : str
        Strategy name identifier.
    analysis_scope : str
        Scope identifier ('Full History' or 'Out-of-Sample').
    simulation_mode : str
        Resampling mode ('trade_level' or 'daily_return').
    n_simulations : int
        Number of simulations executed.
    initial_capital : float
        Starting portfolio capital.
    random_seed : int
        Random seed used.
    empirical_sample_size : int
        Number of empirical observations (completed trades or return days) resampled.
    sample_size_warning : Optional[str]
        Warning text if empirical sample size is small (< 30 observations).
    historical_final_equity : float
        Actual realized final equity from backtest.
    historical_total_return : float
        Actual realized total return percentage from backtest.
    historical_max_drawdown : float
        Actual realized maximum drawdown from backtest.
    terminal_wealth_mean : float
        Mean simulated terminal portfolio value.
    terminal_wealth_median : float
        Median simulated terminal portfolio value.
    terminal_wealth_std : float
        Standard deviation of simulated terminal wealth values.
    terminal_wealth_percentiles : Dict[str, float]
        Simulated outcome percentiles ('5th', '25th', '50th', '75th', '95th').
    prob_of_loss : float
        Probability of portfolio loss (terminal wealth < initial capital).
    max_drawdown_mean : float
        Mean simulated maximum drawdown.
    max_drawdown_median : float
        Median simulated maximum drawdown.
    max_drawdown_percentiles : Dict[str, float]
        Simulated outcome percentiles for max drawdown ('5th', '25th', '50th', '75th', '95th').
    prob_drawdown_worse_than_historical : float
        Probability of experiencing a drawdown worse (more negative) than historical max DD.
    simulated_paths : np.ndarray
        Array of sample equity paths for visual rendering, shape (n_visualize, path_len).
    all_terminal_wealth : np.ndarray
        Array of terminal wealth values across all simulations, shape (n_simulations,).
    all_max_drawdowns : np.ndarray
        Array of max drawdown values across all simulations, shape (n_simulations,).
    """

    strategy_name: str
    analysis_scope: str
    simulation_mode: str
    n_simulations: int
    initial_capital: float
    random_seed: int
    empirical_sample_size: int
    sample_size_warning: Optional[str]
    historical_final_equity: float
    historical_total_return: float
    historical_max_drawdown: float
    terminal_wealth_mean: float
    terminal_wealth_median: float
    terminal_wealth_std: float
    terminal_wealth_percentiles: Dict[str, float]
    prob_of_loss: float
    max_drawdown_mean: float
    max_drawdown_median: float
    max_drawdown_percentiles: Dict[str, float]
    prob_drawdown_worse_than_historical: float
    simulated_paths: np.ndarray
    all_terminal_wealth: np.ndarray
    all_max_drawdowns: np.ndarray


def calculate_path_max_drawdown(equity_path: np.ndarray) -> float:
    """Calculate maximum drawdown for a single 1D equity path.

    Parameters
    ----------
    equity_path : np.ndarray
        1D array of equity values over time.

    Returns
    -------
    float
        Maximum drawdown (<= 0.0).
    """
    if len(equity_path) == 0:
        return 0.0
    running_peak = np.maximum.accumulate(equity_path)
    drawdowns = (equity_path / running_peak) - 1.0
    return float(np.min(drawdowns))


def bootstrap_trade_returns(
    round_trips: List[RoundTripTrade],
    config: MonteCarloConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample realized round-trip trade returns with replacement (Trade-Level Bootstrap).

    Parameters
    ----------
    round_trips : List[RoundTripTrade]
        List of matched round-trip trade objects.
    config : MonteCarloConfig
        Monte Carlo configuration settings.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (all_terminal_wealth, all_max_drawdowns, sample_equity_paths)
    """
    rng = np.random.default_rng(config.random_seed)
    n_sims = config.n_simulations
    init_cap = config.initial_capital
    n_vis = min(config.n_visualize, n_sims)

    if not round_trips:
        # Handling zero completed trades
        all_term = np.full(n_sims, init_cap)
        all_dd = np.zeros(n_sims)
        paths = np.full((n_vis, 1), init_cap)
        return all_term, all_dd, paths

    # Empirical array of percentage returns per round-trip trade
    returns_arr = np.array([rt.net_return for rt in round_trips], dtype=np.float64)
    k_trades = len(returns_arr)

    # Vectorized resampling: shape (n_sims, k_trades)
    resampled_idx = rng.choice(k_trades, size=(n_sims, k_trades), replace=True)
    resampled_rets = returns_arr[resampled_idx]  # shape (n_sims, k_trades)

    # Compute equity paths: cumulative product of (1 + r)
    growth_factors = 1.0 + resampled_rets
    cum_growth = np.cumprod(growth_factors, axis=1)  # shape (n_sims, k_trades)

    # Prepend initial capital to equity paths
    initial_col = np.ones((n_sims, 1), dtype=np.float64)
    norm_paths = np.hstack([initial_col, cum_growth])  # shape (n_sims, k_trades + 1)
    equity_paths = norm_paths * init_cap

    # Terminal wealth
    all_terminal = equity_paths[:, -1]

    # Max Drawdowns per path
    running_peaks = np.maximum.accumulate(equity_paths, axis=1)
    drawdowns = (equity_paths / running_peaks) - 1.0
    all_max_dds = np.min(drawdowns, axis=1)

    sample_paths = equity_paths[:n_vis, :]
    return all_terminal, all_max_dds, sample_paths


def bootstrap_daily_returns(
    daily_returns: pd.Series,
    config: MonteCarloConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample daily portfolio returns with replacement (Daily Return Bootstrap).

    Supports both Simple Empirical Bootstrap and Block Bootstrap.

    Parameters
    ----------
    daily_returns : pd.Series
        Daily portfolio returns time-series.
    config : MonteCarloConfig
        Monte Carlo configuration settings.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (all_terminal_wealth, all_max_drawdowns, sample_equity_paths)
    """
    rng = np.random.default_rng(config.random_seed)
    clean_rets = daily_returns.dropna().to_numpy(dtype=np.float64)
    n_sims = config.n_simulations
    init_cap = config.initial_capital
    n_vis = min(config.n_visualize, n_sims)
    t_days = len(clean_rets)

    if t_days == 0:
        all_term = np.full(n_sims, init_cap)
        all_dd = np.zeros(n_sims)
        paths = np.full((n_vis, 1), init_cap)
        return all_term, all_dd, paths

    block_size = config.block_size

    if block_size is None or block_size <= 1:
        # Simple Empirical Bootstrap
        resampled_idx = rng.choice(t_days, size=(n_sims, t_days), replace=True)
        resampled_rets = clean_rets[resampled_idx]
    else:
        # Block Bootstrap
        n_blocks_needed = int(np.ceil(t_days / block_size))
        max_start_idx = max(1, t_days - block_size + 1)
        resampled_rets_list = []

        for _ in range(n_sims):
            sim_rets = []
            for _ in range(n_blocks_needed):
                start_i = rng.integers(0, max_start_idx)
                block = clean_rets[start_i : start_i + block_size]
                sim_rets.extend(block)
            resampled_rets_list.append(sim_rets[:t_days])

        resampled_rets = np.array(resampled_rets_list, dtype=np.float64)

    # Compute equity paths
    growth_factors = 1.0 + resampled_rets
    cum_growth = np.cumprod(growth_factors, axis=1)

    initial_col = np.ones((n_sims, 1), dtype=np.float64)
    norm_paths = np.hstack([initial_col, cum_growth])
    equity_paths = norm_paths * init_cap

    all_terminal = equity_paths[:, -1]

    running_peaks = np.maximum.accumulate(equity_paths, axis=1)
    drawdowns = (equity_paths / running_peaks) - 1.0
    all_max_dds = np.min(drawdowns, axis=1)

    sample_paths = equity_paths[:n_vis, :]
    return all_terminal, all_max_dds, sample_paths


def run_monte_carlo_simulation(
    backtest_result: BacktestResult,
    strategy_name: str,
    config: Optional[MonteCarloConfig] = None,
    mode: str = "trade_level",
    scope: str = "Full History",
) -> MonteCarloResult:
    """Run Monte Carlo simulation on backtest outcomes and calculate outcome distributions.

    Parameters
    ----------
    backtest_result : BacktestResult
        Result object produced by Backtester.run() or run_buy_and_hold_benchmark().
    strategy_name : str
        Strategy identifier.
    config : Optional[MonteCarloConfig], default None
        Configuration settings (defaults to 5000 simulations, seed=42, initial_capital=100000.0).
    mode : str, default "trade_level"
        Resampling mode: "trade_level" or "daily_return".
    scope : str, default "Full History"
        Analysis scope description ("Full History" or "Out-of-Sample").

    Returns
    -------
    MonteCarloResult
        Monte Carlo simulation result container.
    """
    if config is None:
        config = MonteCarloConfig()

    hist_final_eq = float(backtest_result.portfolio_history["total_equity"].iloc[-1]) if not backtest_result.portfolio_history.empty else config.initial_capital
    hist_tot_ret = backtest_result.total_return
    hist_max_dd = float(((backtest_result.portfolio_history["total_equity"] / backtest_result.portfolio_history["total_equity"].cummax()) - 1.0).min()) if not backtest_result.portfolio_history.empty else 0.0

    round_trips = match_round_trip_trades(backtest_result.trades)

    if mode == "trade_level":
        emp_sample_size = len(round_trips)
        all_term, all_dd, sample_paths = bootstrap_trade_returns(round_trips, config)
    elif mode == "daily_return":
        daily_rets = backtest_result.portfolio_history.get("daily_return", pd.Series())
        emp_sample_size = len(daily_rets.dropna())
        all_term, all_dd, sample_paths = bootstrap_daily_returns(daily_rets, config)
    else:
        raise ValueError(f"Invalid mode '{mode}'. Choose 'trade_level' or 'daily_return'.")

    # Sample Size Warning Logic
    if emp_sample_size < 30:
        sample_warning = (
            f"WARNING: Small empirical sample size ({emp_sample_size} observations). "
            f"Bootstrapping 5,000 simulations resamples existing data and DOES NOT increase underlying historical sample size!"
        )
        logger.warning(f"[{strategy_name} - {scope}] {sample_warning}")
    else:
        sample_warning = None

    # Terminal Wealth Percentiles
    term_mean = float(np.mean(all_term))
    term_median = float(np.median(all_term))
    term_std = float(np.std(all_term, ddof=1)) if len(all_term) > 1 else 0.0

    term_percentiles = {
        "5th": float(np.percentile(all_term, 5)),
        "25th": float(np.percentile(all_term, 25)),
        "50th": float(np.percentile(all_term, 50)),
        "75th": float(np.percentile(all_term, 75)),
        "95th": float(np.percentile(all_term, 95)),
    }

    prob_loss = float(np.mean(all_term < config.initial_capital))

    # Max Drawdown Percentiles (Note: drawdowns are <= 0.0)
    dd_mean = float(np.mean(all_dd))
    dd_median = float(np.median(all_dd))

    dd_percentiles = {
        "5th": float(np.percentile(all_dd, 5)),    # Severe downside drawdown percentile
        "25th": float(np.percentile(all_dd, 25)),
        "50th": float(np.percentile(all_dd, 50)),
        "75th": float(np.percentile(all_dd, 75)),
        "95th": float(np.percentile(all_dd, 95)),
    }

    # Probability of drawdown worse than historical max drawdown
    prob_worse_dd = float(np.mean(all_dd < hist_max_dd))

    return MonteCarloResult(
        strategy_name=strategy_name,
        analysis_scope=scope,
        simulation_mode=mode,
        n_simulations=config.n_simulations,
        initial_capital=config.initial_capital,
        random_seed=config.random_seed,
        empirical_sample_size=emp_sample_size,
        sample_size_warning=sample_warning,
        historical_final_equity=hist_final_eq,
        historical_total_return=hist_tot_ret,
        historical_max_drawdown=hist_max_dd,
        terminal_wealth_mean=term_mean,
        terminal_wealth_median=term_median,
        terminal_wealth_std=term_std,
        terminal_wealth_percentiles=term_percentiles,
        prob_of_loss=prob_loss,
        max_drawdown_mean=dd_mean,
        max_drawdown_median=dd_median,
        max_drawdown_percentiles=dd_percentiles,
        prob_drawdown_worse_than_historical=prob_worse_dd,
        simulated_paths=sample_paths,
        all_terminal_wealth=all_term,
        all_max_drawdowns=all_dd,
    )


def compare_monte_carlo_results(
    mc_results_dict: Dict[str, MonteCarloResult]
) -> pd.DataFrame:
    """Format a summary comparison DataFrame across MonteCarloResult objects.

    Parameters
    ----------
    mc_results_dict : Dict[str, MonteCarloResult]
        Dictionary mapping strategy names to MonteCarloResult objects.

    Returns
    -------
    pd.DataFrame
        Formatted Monte Carlo comparison table.
    """
    rows = []
    for name, mc in mc_results_dict.items():
        tp = mc.terminal_wealth_percentiles
        dp = mc.max_drawdown_percentiles
        rows.append(
            {
                "Strategy": name,
                "Scope": mc.analysis_scope,
                "Mode": mc.simulation_mode,
                "Empirical Sample Size (Obs)": mc.empirical_sample_size,
                "Hist Final Eq": f"${mc.historical_final_equity:,.2f}",
                "Hist MaxDD": f"{mc.historical_max_drawdown*100:.2f}%",
                "Sim Median Eq": f"${mc.terminal_wealth_median:,.2f}",
                "Sim 5th% Eq": f"${tp['5th']:,.2f}",
                "Sim 95th% Eq": f"${tp['95th']:,.2f}",
                "Prob Loss": f"{mc.prob_of_loss*100:.1f}%",
                "Sim Median MaxDD": f"{mc.max_drawdown_median*100:.2f}%",
                "Sim 5th% MaxDD": f"{dp['5th']*100:.2f}%",
                "Prob Worse DD": f"{mc.prob_drawdown_worse_than_historical*100:.1f}%",
            }
        )
    return pd.DataFrame(rows)
