"""
Performance and Risk Analytics Module.

Provides transparent, reproducible quantitative metrics evaluating portfolio equity curves,
daily return series, drawdown recovery behavior, round-trip trade statistics, and transaction
costs from BacktestResult objects.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.backtest import BacktestResult

logger = logging.getLogger(__name__)


@dataclass
class RoundTripTrade:
    """Dataclass holding matched entry/exit round-trip trade metrics.

    Attributes
    ----------
    round_trip_id : int
        Sequential index of the completed round-trip trade.
    buy_trade_id : int
        Trade ID of the BUY entry.
    sell_trade_id : int
        Trade ID of the SELL exit.
    entry_timestamp : pd.Timestamp
        Execution timestamp of the BUY entry.
    exit_timestamp : pd.Timestamp
        Execution timestamp of the SELL exit.
    entry_price : float
        Effective execution price of BUY entry.
    exit_price : float
        Effective execution price of SELL exit.
    quantity : float
        Executed unit quantity.
    buy_cash_flow : float
        Net cash flow of entry (negative value).
    sell_cash_flow : float
        Net cash flow of exit (positive value).
    net_pnl : float
        Net profit/loss in currency (sell_cash_flow + buy_cash_flow).
    net_return : float
        Net percentage return of the round trip (net_pnl / abs(buy_cash_flow)).
    holding_days : float
        Calendar days held between entry and exit.
    """

    round_trip_id: int
    buy_trade_id: int
    sell_trade_id: int
    entry_timestamp: pd.Timestamp
    exit_timestamp: pd.Timestamp
    entry_price: float
    exit_price: float
    quantity: float
    buy_cash_flow: float
    sell_cash_flow: float
    net_pnl: float
    net_return: float
    holding_days: float


@dataclass
class PerformanceMetrics:
    """Dataclass holding comprehensive performance and risk analytics.

    Attributes
    ----------
    total_return : float
        Total return percentage ((final_equity / initial_capital) - 1).
    cagr : float
        Annualized Compound Growth Rate over actual calendar time span.
    annualized_volatility : float
        Annualized standard deviation of daily portfolio returns (std * sqrt(252)).
    max_drawdown : float
        Maximum peak-to-trough equity decline percentage (always <= 0).
    drawdown_peak_date : Optional[pd.Timestamp]
        Timestamp of the portfolio equity peak prior to max drawdown.
    drawdown_trough_date : Optional[pd.Timestamp]
        Timestamp of the maximum drawdown trough.
    drawdown_recovery_date : Optional[pd.Timestamp]
        Timestamp when equity recovered back to peak level (None if unrecovered).
    sharpe_ratio : float
        Annualized Sharpe Ratio (assumes Risk-Free Rate Rf = 0.0).
    sortino_ratio : float
        Annualized Sortino Ratio (assumes MAR = 0.0).
    calmar_ratio : float
        Ratio of CAGR to absolute max drawdown (CAGR / abs(max_drawdown)).
    total_executed_trades : int
        Total count of executed trade orders (BUY + SELL).
    completed_round_trips : int
        Total count of paired BUY-to-SELL completed round-trip trades.
    winning_trades : int
        Count of completed round-trip trades with net_pnl > 0.
    losing_trades : int
        Count of completed round-trip trades with net_pnl <= 0.
    win_rate : float
        Proportion of winning round-trip trades (winning_trades / completed_round_trips).
    average_win : float
        Mean PnL of winning round-trip trades.
    average_loss : float
        Mean PnL of losing round-trip trades.
    profit_factor : float
        Ratio of gross winning PnL to absolute gross losing PnL.
    total_transaction_costs : float
        Total transaction costs paid across all executed trades.
    avg_cost_per_trade : float
        Mean transaction cost per executed trade order.
    cost_to_capital_ratio : float
        Transaction costs as a fraction of initial capital.
    execution_convention : str
        Documented execution model used by the strategy or benchmark.
    """

    total_return: float
    cagr: float
    annualized_volatility: float
    max_drawdown: float
    drawdown_peak_date: Optional[pd.Timestamp]
    drawdown_trough_date: Optional[pd.Timestamp]
    drawdown_recovery_date: Optional[pd.Timestamp]
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    total_executed_trades: int
    completed_round_trips: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    average_win: float
    average_loss: float
    profit_factor: float
    total_transaction_costs: float
    avg_cost_per_trade: float
    cost_to_capital_ratio: float
    execution_convention: str


def calculate_cagr(portfolio_history: pd.DataFrame, initial_capital: float) -> float:
    """Calculate Compound Annual Growth Rate (CAGR) based on actual elapsed calendar time.

    Mathematical Definition
    ------------------------
    years = (T_end - T_start)_days / 365.25
    CAGR = (final_equity / initial_capital) ^ (1 / years) - 1

    Parameters
    ----------
    portfolio_history : pd.DataFrame
        DataFrame indexed by DatetimeIndex containing 'total_equity'.
    initial_capital : float
        Starting capital.

    Returns
    -------
    float
        Annualized CAGR.
    """
    if portfolio_history.empty or len(portfolio_history) < 2:
        return 0.0

    start_date = portfolio_history.index[0]
    end_date = portfolio_history.index[-1]

    elapsed_days = (end_date - start_date).days
    if elapsed_days <= 0:
        return 0.0

    years = elapsed_days / 365.25
    final_equity = float(portfolio_history["total_equity"].iloc[-1])

    if final_equity <= 0 or initial_capital <= 0:
        return -1.0  # Total loss

    return float((final_equity / initial_capital) ** (1.0 / years) - 1.0)


def calculate_annualized_volatility(
    daily_returns: pd.Series, annualization_factor: float = np.sqrt(252)
) -> float:
    """Calculate annualized standard deviation of daily portfolio returns.

    Parameters
    ----------
    daily_returns : pd.Series
        Daily returns time-series.
    annualization_factor : float, default sqrt(252)
        Factor assuming 252 trading days per year.

    Returns
    -------
    float
        Annualized volatility.
    """
    clean_returns = daily_returns.dropna()
    if len(clean_returns) < 2:
        return 0.0
    return float(clean_returns.std(ddof=1) * annualization_factor)


def calculate_drawdown_metrics(
    portfolio_history: pd.DataFrame,
) -> Tuple[float, Optional[pd.Timestamp], Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """Calculate Maximum Drawdown, Peak Date, Trough Date, and Recovery Date.

    Mathematical Definition
    ------------------------
    running_peak_t = max(equity_0 ... equity_t)
    drawdown_t = (equity_t / running_peak_t) - 1.0
    max_drawdown = min(drawdown_t)

    Parameters
    ----------
    portfolio_history : pd.DataFrame
        DataFrame indexed by DatetimeIndex containing 'total_equity'.

    Returns
    -------
    Tuple[float, Optional[pd.Timestamp], Optional[pd.Timestamp], Optional[pd.Timestamp]]
        (max_drawdown, peak_date, trough_date, recovery_date)
    """
    if portfolio_history.empty:
        return 0.0, None, None, None

    equity = portfolio_history["total_equity"]
    running_peak = equity.cummax()
    drawdown = (equity / running_peak) - 1.0

    max_dd = float(drawdown.min())
    if max_dd >= 0.0 or drawdown.isna().all():
        return 0.0, None, None, None

    # Trough date: timestamp where drawdown is minimum
    trough_date = drawdown.idxmin()

    # Peak date: timestamp of running_peak prior to trough_date
    equity_up_to_trough = equity.loc[:trough_date]
    peak_date = equity_up_to_trough.idxmax()
    peak_equity = float(equity.loc[peak_date])

    # Recovery date: first timestamp after trough_date where equity >= peak_equity
    equity_after_trough = equity.loc[trough_date:]
    recovered = equity_after_trough[equity_after_trough >= peak_equity]

    if not recovered.empty and recovered.index[0] != trough_date:
        recovery_date = recovered.index[0]
    else:
        recovery_date = None  # Unrecovered by end of dataset

    return max_dd, peak_date, trough_date, recovery_date


def calculate_sharpe_ratio(
    daily_returns: pd.Series,
    risk_free_rate: float = 0.0,
    annualization_factor: float = np.sqrt(252),
) -> float:
    """Calculate annualized Sharpe Ratio using daily portfolio returns.

    Parameters
    ----------
    daily_returns : pd.Series
        Daily returns series.
    risk_free_rate : float, default 0.0
        Annualized risk-free rate assumption (default 0.0).
    annualization_factor : float, default sqrt(252)
        Annualization multiplier.

    Returns
    -------
    float
        Annualized Sharpe Ratio. Handles zero volatility safely (returns 0.0).
    """
    clean_returns = daily_returns.dropna()
    if len(clean_returns) < 2:
        return 0.0

    daily_rf = (1.0 + risk_free_rate) ** (1.0 / 252.0) - 1.0
    excess_returns = clean_returns - daily_rf

    std_dev = excess_returns.std(ddof=1)
    if std_dev <= 1e-12:
        return 0.0  # Zero volatility safety check

    daily_sharpe = excess_returns.mean() / std_dev
    return float(daily_sharpe * annualization_factor)


def calculate_sortino_ratio(
    daily_returns: pd.Series,
    mar: float = 0.0,
    annualization_factor: float = np.sqrt(252),
) -> float:
    """Calculate annualized Sortino Ratio measuring downside-adjusted return.

    Parameters
    ----------
    daily_returns : pd.Series
        Daily returns series.
    mar : float, default 0.0
        Minimum Acceptable Return assumption.
    annualization_factor : float, default sqrt(252)
        Annualization multiplier.

    Returns
    -------
    float
        Annualized Sortino Ratio. Handles zero downside deviation safely.
    """
    clean_returns = daily_returns.dropna()
    if len(clean_returns) < 2:
        return 0.0

    excess_returns = clean_returns - mar
    downside_diffs = np.minimum(excess_returns, 0.0)

    # Downside deviation (root mean square of downside differences)
    downside_dev = np.sqrt(np.mean(downside_diffs**2))
    if downside_dev <= 1e-12:
        return 0.0 if excess_returns.mean() <= 0 else 999.0  # Safe handling

    sortino = (excess_returns.mean() / downside_dev) * annualization_factor
    return float(sortino)


def calculate_calmar_ratio(cagr: float, max_drawdown: float) -> float:
    """Calculate Calmar Ratio (CAGR / abs(max_drawdown)).

    Parameters
    ----------
    cagr : float
        Annualized Compound Growth Rate.
    max_drawdown : float
        Maximum drawdown (<= 0).

    Returns
    -------
    float
        Calmar Ratio. Handles zero drawdown safely.
    """
    abs_dd = abs(max_drawdown)
    if abs_dd <= 1e-12:
        return 0.0 if cagr <= 0 else 999.0
    return float(cagr / abs_dd)


def match_round_trip_trades(trades_df: pd.DataFrame) -> List[RoundTripTrade]:
    """Match BUY entries with SELL exits to form completed round-trip trades.

    Matching Algorithm
    ------------------
    Iterates chronologically over trade execution logs.
    Pairs each BUY entry with the subsequent SELL exit.
    Calculates net round-trip PnL = sell_net_cash_flow + buy_net_cash_flow.

    Parameters
    ----------
    trades_df : pd.DataFrame
        Trade log DataFrame containing 'side', 'net_cash_flow', 'execution_price', 'quantity', 'execution_timestamp'.

    Returns
    -------
    List[RoundTripTrade]
        List of matched RoundTripTrade objects.
    """
    if trades_df.empty:
        return []

    sorted_trades = trades_df.sort_values("execution_timestamp").reset_index(drop=True)
    round_trips: List[RoundTripTrade] = []

    pending_buy: Optional[pd.Series] = None
    round_trip_counter = 0

    for _, trade in sorted_trades.iterrows():
        if trade["side"] == "BUY":
            pending_buy = trade
        elif trade["side"] == "SELL" and pending_buy is not None:
            round_trip_counter += 1
            buy_cash = float(pending_buy["net_cash_flow"])  # Negative value
            sell_cash = float(trade["net_cash_flow"])        # Positive value
            net_pnl = sell_cash + buy_cash

            buy_ts = pd.to_datetime(pending_buy["execution_timestamp"])
            sell_ts = pd.to_datetime(trade["execution_timestamp"])
            holding_days = (sell_ts - buy_ts).total_seconds() / (24.0 * 3600.0)

            net_return = (net_pnl / abs(buy_cash)) if abs(buy_cash) > 0 else 0.0

            round_trips.append(
                RoundTripTrade(
                    round_trip_id=round_trip_counter,
                    buy_trade_id=int(pending_buy["trade_id"]),
                    sell_trade_id=int(trade["trade_id"]),
                    entry_timestamp=buy_ts,
                    exit_timestamp=sell_ts,
                    entry_price=float(pending_buy["execution_price"]),
                    exit_price=float(trade["execution_price"]),
                    quantity=float(trade["quantity"]),
                    buy_cash_flow=buy_cash,
                    sell_cash_flow=sell_cash,
                    net_pnl=net_pnl,
                    net_return=net_return,
                    holding_days=holding_days,
                )
            )
            pending_buy = None  # Reset after matching exit

    return round_trips


def evaluate_backtest(backtest_result: BacktestResult) -> PerformanceMetrics:
    """Evaluate a BacktestResult object and compute complete performance and risk analytics.

    Parameters
    ----------
    backtest_result : BacktestResult
        Result object produced by Backtester.run() or run_buy_and_hold_benchmark().

    Returns
    -------
    PerformanceMetrics
        Comprehensive performance metrics container.
    """
    history = backtest_result.portfolio_history
    trades_df = backtest_result.trades
    initial_capital = (
        float(history["total_equity"].iloc[0]) if not history.empty else 100000.0
    )

    # 1. Total Return & CAGR
    tot_ret = backtest_result.total_return
    cagr = calculate_cagr(history, initial_capital)

    # 2. Volatility
    daily_returns = history["daily_return"] if "daily_return" in history.columns else pd.Series()
    ann_vol = calculate_annualized_volatility(daily_returns)

    # 3. Maximum Drawdown & Recovery
    max_dd, peak_dt, trough_dt, rec_dt = calculate_drawdown_metrics(history)

    # 4. Sharpe, Sortino, Calmar Ratios
    sharpe = calculate_sharpe_ratio(daily_returns)
    sortino = calculate_sortino_ratio(daily_returns)
    calmar = calculate_calmar_ratio(cagr, max_dd)

    # 5. Round-Trip Trade Matching & Statistics
    round_trips = match_round_trip_trades(trades_df)
    completed_cnt = len(round_trips)

    if completed_cnt > 0:
        pnls = [rt.net_pnl for rt in round_trips]
        winning_pnls = [p for p in pnls if p > 0]
        losing_pnls = [p for p in pnls if p <= 0]

        winning_cnt = len(winning_pnls)
        losing_cnt = len(losing_pnls)
        win_rate = winning_cnt / completed_cnt

        avg_win = float(np.mean(winning_pnls)) if winning_cnt > 0 else 0.0
        avg_loss = float(np.mean(losing_pnls)) if losing_cnt > 0 else 0.0

        gross_win = sum(winning_pnls) if winning_cnt > 0 else 0.0
        gross_loss = abs(sum(losing_pnls)) if losing_cnt > 0 else 0.0

        if gross_loss > 1e-12:
            profit_factor = float(gross_win / gross_loss)
        elif gross_win > 0:
            profit_factor = 999.0  # All winning trades
        else:
            profit_factor = 0.0
    else:
        winning_cnt = 0
        losing_cnt = 0
        win_rate = 0.0
        avg_win = 0.0
        avg_loss = 0.0
        profit_factor = 0.0

    # 6. Transaction Cost Analysis
    tot_costs = backtest_result.total_transaction_costs
    tot_trades_cnt = backtest_result.total_trades_count
    avg_cost = tot_costs / tot_trades_cnt if tot_trades_cnt > 0 else 0.0
    cost_capital_pct = tot_costs / initial_capital

    return PerformanceMetrics(
        total_return=tot_ret,
        cagr=cagr,
        annualized_volatility=ann_vol,
        max_drawdown=max_dd,
        drawdown_peak_date=peak_dt,
        drawdown_trough_date=trough_dt,
        drawdown_recovery_date=rec_dt,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        total_executed_trades=tot_trades_cnt,
        completed_round_trips=completed_cnt,
        winning_trades=winning_cnt,
        losing_trades=losing_cnt,
        win_rate=win_rate,
        average_win=avg_win,
        average_loss=avg_loss,
        profit_factor=profit_factor,
        total_transaction_costs=tot_costs,
        avg_cost_per_trade=avg_cost,
        cost_to_capital_ratio=cost_capital_pct,
        execution_convention=backtest_result.execution_convention,
    )


def compare_performance(metrics_dict: Dict[str, PerformanceMetrics]) -> pd.DataFrame:
    """Format a summary comparison DataFrame across multiple strategy PerformanceMetrics.

    Parameters
    ----------
    metrics_dict : Dict[str, PerformanceMetrics]
        Dictionary mapping strategy names to PerformanceMetrics objects.

    Returns
    -------
    pd.DataFrame
        Formatted comparison DataFrame.
    """
    rows = []
    for name, m in metrics_dict.items():
        rows.append(
            {
                "Strategy": name,
                "Total Return (%)": f"{m.total_return*100:.2f}%",
                "CAGR (%)": f"{m.cagr*100:.2f}%",
                "Ann. Volatility (%)": f"{m.annualized_volatility*100:.2f}%",
                "Max Drawdown (%)": f"{m.max_drawdown*100:.2f}%",
                "Sharpe Ratio": f"{m.sharpe_ratio:.2f}",
                "Sortino Ratio": f"{m.sortino_ratio:.2f}",
                "Calmar Ratio": f"{m.calmar_ratio:.2f}",
                "Executed Trades": m.total_executed_trades,
                "Completed Round-Trips": m.completed_round_trips,
                "Win Rate (%)": f"{m.win_rate*100:.1f}%",
                "Profit Factor": f"{m.profit_factor:.2f}",
                "Total Costs ($)": f"${m.total_transaction_costs:,.2f}",
            }
        )
    return pd.DataFrame(rows)
