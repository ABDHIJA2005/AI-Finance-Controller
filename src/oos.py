"""
Out-of-Sample Testing and Robustness Analysis Module.

Provides chronological time-series splitting, period-isolated backtesting, performance
degradation analysis, and signal stability tracking across Development (2015-2021),
Validation (2022-2023), and Out-of-Sample (2024-2026) periods.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.backtest import Backtester, BacktestResult, run_buy_and_hold_benchmark
from src.metrics import PerformanceMetrics, evaluate_backtest, match_round_trip_trades

logger = logging.getLogger(__name__)


@dataclass
class PeriodPartition:
    """Dataclass holding a chronologically partitioned time-series slice.

    Attributes
    ----------
    name : str
        Partition name ('Development', 'Validation', 'Out-of-Sample').
    start_date : pd.Timestamp
        First timestamp in partition.
    end_date : pd.Timestamp
        Last timestamp in partition.
    rows_count : int
        Number of trading days in partition.
    df : pd.DataFrame
        Market OHLCV DataFrame slice.
    signals : pd.DataFrame
        Strategy signals DataFrame slice.
    """

    name: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    rows_count: int
    df: pd.DataFrame
    signals: pd.DataFrame


@dataclass
class MultiPeriodBacktestResult:
    """Dataclass holding strategy evaluation across Dev, Val, and OOS periods.

    Attributes
    ----------
    strategy_name : str
        Strategy name identifier.
    dev_metrics : PerformanceMetrics
        Performance metrics for Development period (2015-2021).
    val_metrics : PerformanceMetrics
        Performance metrics for Validation period (2022-2023).
    oos_metrics : PerformanceMetrics
        Performance metrics for Out-of-Sample period (2024-2026).
    degradation : Dict[str, float]
        Deltas between OOS and Development metrics (delta_cagr, delta_sharpe, etc.).
    stability : Dict[str, Dict[str, Union[int, float]]]
        Signal event and market exposure statistics per period.
    """

    strategy_name: str
    dev_metrics: PerformanceMetrics
    val_metrics: PerformanceMetrics
    oos_metrics: PerformanceMetrics
    degradation: Dict[str, float]
    stability: Dict[str, Dict[str, Union[int, float]]]


def split_chronologically(
    df: pd.DataFrame, signals_df: pd.DataFrame
) -> Dict[str, PeriodPartition]:
    """Partition market price data and signals into 3 non-overlapping chronological periods.

    Period Definitions
    ------------------
    - Development : 2015-01-01 through 2021-12-31
    - Validation  : 2022-01-01 through 2023-12-31
    - Out-of-Sample: 2024-01-01 through latest available date

    Parameters
    ----------
    df : pd.DataFrame
        Full market OHLCV DataFrame with DatetimeIndex.
    signals_df : pd.DataFrame
        Full strategy signals DataFrame with DatetimeIndex.

    Returns
    -------
    Dict[str, PeriodPartition]
        Dictionary mapping partition names ('Development', 'Validation', 'Out-of-Sample')
        to PeriodPartition objects.

    Raises
    ------
    ValueError
        If index alignment is invalid or DataFrame is empty.
    """
    if not df.index.equals(signals_df.index):
        raise ValueError("Price DataFrame index and Signals DataFrame index must match exactly.")

    # Masks for chronological periods
    dev_mask = (df.index >= "2015-01-01") & (df.index <= "2021-12-31")
    val_mask = (df.index >= "2022-01-01") & (df.index <= "2023-12-31")
    oos_mask = df.index >= "2024-01-01"

    partitions: Dict[str, PeriodPartition] = {}
    specs = [
        ("Development", dev_mask),
        ("Validation", val_mask),
        ("Out-of-Sample", oos_mask),
    ]

    for name, mask in specs:
        sub_df = df.loc[mask].copy()
        sub_sig = signals_df.loc[mask].copy()

        if sub_df.empty:
            logger.warning(f"Partition '{name}' is empty.")
            start_ts = pd.Timestamp("1970-01-01")
            end_ts = pd.Timestamp("1970-01-01")
            count = 0
        else:
            start_ts = sub_df.index[0]
            end_ts = sub_df.index[-1]
            count = len(sub_df)

        partitions[name] = PeriodPartition(
            name=name,
            start_date=start_ts,
            end_date=end_ts,
            rows_count=count,
            df=sub_df,
            signals=sub_sig,
        )

    return partitions


def calculate_signal_stability(
    signals_df: pd.DataFrame, partition_df: pd.DataFrame
) -> Dict[str, Union[int, float]]:
    """Calculate signal stability and market exposure metrics for a period partition.

    Parameters
    ----------
    signals_df : pd.DataFrame
        Signals DataFrame slice.
    partition_df : pd.DataFrame
        Price DataFrame slice.

    Returns
    -------
    Dict[str, Union[int, float]]
        Stability dictionary (entries, exits, market_exposure_pct, long_days).
    """
    total_days = len(partition_df)
    if total_days == 0 or "position_state" not in signals_df.columns:
        return {
            "entries_count": 0,
            "exits_count": 0,
            "market_exposure_pct": 0.0,
            "long_days": 0,
        }

    pos_states = signals_df["position_state"].fillna(0.0)
    long_days = int((pos_states == 1.0).sum())
    exposure_pct = float(long_days / total_days)

    sig_events = signals_df.get("signal_event", pd.Series())
    entries_cnt = int((sig_events == 1.0).sum())
    exits_cnt = int((sig_events == -1.0).sum())

    return {
        "entries_count": entries_cnt,
        "exits_count": exits_cnt,
        "market_exposure_pct": exposure_pct,
        "long_days": long_days,
    }


def calculate_degradation(
    dev_metrics: PerformanceMetrics, oos_metrics: PerformanceMetrics
) -> Dict[str, float]:
    """Calculate performance degradation metrics between Development and OOS periods.

    Parameters
    ----------
    dev_metrics : PerformanceMetrics
        Metrics for Development period.
    oos_metrics : PerformanceMetrics
        Metrics for Out-of-Sample period.

    Returns
    -------
    Dict[str, float]
        Degradation deltas (delta_cagr, delta_sharpe, delta_max_drawdown, delta_win_rate, delta_profit_factor).
    """
    return {
        "delta_cagr": float(oos_metrics.cagr - dev_metrics.cagr),
        "delta_sharpe": float(oos_metrics.sharpe_ratio - dev_metrics.sharpe_ratio),
        "delta_max_drawdown": float(oos_metrics.max_drawdown - dev_metrics.max_drawdown),
        "delta_win_rate": float(oos_metrics.win_rate - dev_metrics.win_rate),
        "delta_profit_factor": float(oos_metrics.profit_factor - dev_metrics.profit_factor),
    }


def run_multi_period_backtest(
    df: pd.DataFrame,
    signals_df: pd.DataFrame,
    strategy_name: str,
    backtester: Optional[Backtester] = None,
    is_benchmark: bool = False,
) -> MultiPeriodBacktestResult:
    """Run independent backtests across Development, Validation, and OOS periods.

    Portfolio Boundary Rule
    -----------------------
    Each period (Dev, Val, OOS) is evaluated as an independent window with a fresh
    initial capital balance ($100,000). Positions, cash, or equity are NOT carried
    over period boundaries. Feature warm-up is preserved by slicing pre-computed features/signals.

    Parameters
    ----------
    df : pd.DataFrame
        Full market OHLCV DataFrame.
    signals_df : pd.DataFrame
        Full strategy signals DataFrame.
    strategy_name : str
        Name identifier for the strategy.
    backtester : Optional[Backtester], default None
        Backtester instance for strategy backtests.
    is_benchmark : bool, default False
        If True, uses Buy & Hold benchmark execution model.

    Returns
    -------
    MultiPeriodBacktestResult
        Multi-period evaluation summary container.
    """
    if backtester is None and not is_benchmark:
        backtester = Backtester(initial_capital=100000.0)

    partitions = split_chronologically(df, signals_df)

    period_results: Dict[str, Tuple[BacktestResult, PerformanceMetrics]] = {}
    stability_map: Dict[str, Dict[str, Union[int, float]]] = {}

    for name in ["Development", "Validation", "Out-of-Sample"]:
        part = partitions[name]
        p_df = part.df
        p_sig = part.signals

        if p_df.empty:
            raise ValueError(f"Partition '{name}' contains no data rows.")

        # Fresh initial capital simulation per period
        if is_benchmark:
            init_cap = backtester.initial_capital if backtester else 100000.0
            cost_r = backtester.transaction_cost_rate if backtester else 0.0005
            slip_r = backtester.slippage_rate if backtester else 0.0005
            res = run_buy_and_hold_benchmark(
                p_df,
                initial_capital=init_cap,
                transaction_cost_rate=cost_r,
                slippage_rate=slip_r,
            )
        else:
            # Re-instantiate backtester to ensure clean initial capital state
            fresh_bt = Backtester(
                initial_capital=backtester.initial_capital,
                position_size=backtester.position_size,
                transaction_cost_rate=backtester.transaction_cost_rate,
                slippage_rate=backtester.slippage_rate,
                allow_fractional=backtester.allow_fractional,
            )
            res = fresh_bt.run(p_df, p_sig)

        metrics = evaluate_backtest(res)
        period_results[name] = (res, metrics)
        stability_map[name] = calculate_signal_stability(p_sig, p_df)

    dev_res, dev_met = period_results["Development"]
    val_res, val_met = period_results["Validation"]
    oos_res, oos_met = period_results["Out-of-Sample"]

    degradation = calculate_degradation(dev_met, oos_met)

    logger.info(
        f"Multi-period backtest completed for '{strategy_name}': "
        f"Dev CAGR={dev_met.cagr*100:.2f}%, Val CAGR={val_met.cagr*100:.2f}%, OOS CAGR={oos_met.cagr*100:.2f}%"
    )

    return MultiPeriodBacktestResult(
        strategy_name=strategy_name,
        dev_metrics=dev_met,
        val_metrics=val_met,
        oos_metrics=oos_met,
        degradation=degradation,
        stability=stability_map,
    )


def compare_robustness(
    multi_results: Dict[str, MultiPeriodBacktestResult]
) -> pd.DataFrame:
    """Format a summary robustness comparison DataFrame across strategies and periods.

    Parameters
    ----------
    multi_results : Dict[str, MultiPeriodBacktestResult]
        Dictionary mapping strategy names to MultiPeriodBacktestResult objects.

    Returns
    -------
    pd.DataFrame
        Robustness summary table.
    """
    rows = []
    for name, mres in multi_results.items():
        dev = mres.dev_metrics
        val = mres.val_metrics
        oos = mres.oos_metrics
        deg = mres.degradation

        rows.append(
            {
                "Strategy": name,
                "Dev CAGR (%)": f"{dev.cagr*100:.2f}%",
                "Val CAGR (%)": f"{val.cagr*100:.2f}%",
                "OOS CAGR (%)": f"{oos.cagr*100:.2f}%",
                "Dev Sharpe": f"{dev.sharpe_ratio:.2f}",
                "Val Sharpe": f"{val.sharpe_ratio:.2f}",
                "OOS Sharpe": f"{oos.sharpe_ratio:.2f}",
                "Dev MaxDD (%)": f"{dev.max_drawdown*100:.2f}%",
                "Val MaxDD (%)": f"{val.max_drawdown*100:.2f}%",
                "OOS MaxDD (%)": f"{oos.max_drawdown*100:.2f}%",
                "Delta Sharpe (OOS-Dev)": f"{deg['delta_sharpe']:+.2f}",
                "OOS Trades": oos.total_executed_trades,
                "OOS WinRate (%)": f"{oos.win_rate*100:.1f}%",
                "OOS ProfitFactor": f"{oos.profit_factor:.2f}",
            }
        )
    return pd.DataFrame(rows)
