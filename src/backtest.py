"""
Backtesting and Trade Execution Engine Module.

Provides a robust, deterministic backtesting engine that simulates portfolio evolution
from strategy signals, incorporating next-day Open execution (t+1), proportional transaction
costs, execution slippage, cash/position accounting, and trade logging.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from src.signal_engine import validate_required_columns

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """Dataclass holding details of an executed trade.

    Attributes
    ----------
    trade_id : int
        Sequential unique trade identifier.
    signal_timestamp : pd.Timestamp
        Timestamp at the Close of day t when the signal was generated.
    execution_timestamp : pd.Timestamp
        Timestamp at the Open of day t+1 when the trade was executed.
    side : str
        Trade side: 'BUY' or 'SELL'.
    execution_price : float
        Effective execution price including slippage.
    quantity : float
        Number of units/shares executed.
    gross_value : float
        Gross trade value (quantity * execution_price).
    transaction_cost : float
        Proportional transaction cost paid on the trade.
    slippage_amount : float
        Per-unit slippage adjustment.
    net_cash_flow : float
        Net cash change resulting from trade (-outflow for BUY, +inflow for SELL).
    position_after_trade : float
        Position quantity held after trade completion.
    """

    trade_id: int
    signal_timestamp: pd.Timestamp
    execution_timestamp: pd.Timestamp
    side: str
    execution_price: float
    quantity: float
    gross_value: float
    transaction_cost: float
    slippage_amount: float
    net_cash_flow: float
    position_after_trade: float


@dataclass
class BacktestResult:
    """Dataclass holding structured backtest simulation output.

    Attributes
    ----------
    equity_curve : pd.Series
        Daily portfolio total equity time-series.
    portfolio_history : pd.DataFrame
        Daily portfolio accounting DataFrame (cash, quantity, price, position_value, equity, returns).
    trades : pd.DataFrame
        DataFrame representation of the trade log.
    final_equity : float
        Ending portfolio total equity.
    total_return : float
        Cumulative portfolio return ((final_equity / initial_capital) - 1).
    total_transaction_costs : float
        Sum of all transaction costs paid across all executed trades.
    total_trades_count : int
        Total count of executed trades (entries + exits).
    entries_count : int
        Count of BUY entry trades.
    exits_count : int
        Count of SELL exit trades.
    execution_convention : str
        Documentation string stating the execution timing model used.
    """

    equity_curve: pd.Series
    portfolio_history: pd.DataFrame
    trades: pd.DataFrame
    final_equity: float
    total_return: float
    total_transaction_costs: float
    total_trades_count: int
    entries_count: int
    exits_count: int
    execution_convention: str


class Backtester:
    """Chronological backtesting and trade execution engine.

    Simulates portfolio equity, cash, and positions from strategy signals with
    explicit next-day Open execution (t+1), transaction costs, and slippage.
    """

    EXECUTION_CONVENTION_STR = (
        "Strategy Execution: Signal generated at Close(t) -> Executed at Open(t+1)"
    )

    def __init__(
        self,
        initial_capital: float = 100000.0,
        position_size: float = 1.0,
        transaction_cost_rate: float = 0.0005,
        slippage_rate: float = 0.0005,
        allow_fractional: bool = False,
    ) -> None:
        """Initialize Backtester.

        Parameters
        ----------
        initial_capital : float, default 100000.0
            Starting cash capital.
        position_size : float, default 1.0
            Percentage of total portfolio equity allocated to new positions (1.0 = 100%).
        transaction_cost_rate : float, default 0.0005
            Proportional transaction cost rate (e.g., 0.0005 = 5 bps).
        slippage_rate : float, default 0.0005
            Proportional execution slippage rate (e.g., 0.0005 = 5 bps).
        allow_fractional : bool, default False
            If True, allows fractional unit quantities; if False, floors to integer units.
        """
        if initial_capital <= 0:
            raise ValueError(f"initial_capital ({initial_capital}) must be positive.")
        if not (0.0 < position_size <= 1.0):
            raise ValueError(f"position_size ({position_size}) must be in range (0, 1.0].")
        if transaction_cost_rate < 0 or slippage_rate < 0:
            raise ValueError("transaction_cost_rate and slippage_rate cannot be negative.")

        self.initial_capital = initial_capital
        self.position_size = position_size
        self.transaction_cost_rate = transaction_cost_rate
        self.slippage_rate = slippage_rate
        self.allow_fractional = allow_fractional

    def run(self, df: pd.DataFrame, signals_df: pd.DataFrame) -> BacktestResult:
        """Run chronological backtest simulation.

        Parameters
        ----------
        df : pd.DataFrame
            OHLC market DataFrame containing 'open' and 'close' price columns.
        signals_df : pd.DataFrame
            Strategy signals DataFrame containing 'position_state' and/or 'signal_event'.

        Returns
        -------
        BacktestResult
            Structured backtest results container.

        Raises
        ------
        ValueError
            If required price or signal columns are missing.
        """
        validate_required_columns(df, ["open", "close"])

        # Work on copies to prevent input DataFrame mutation
        price_df = df.copy()
        sig_df = signals_df.copy()

        # Ensure index alignment
        if not price_df.index.equals(sig_df.index):
            raise ValueError("Price DataFrame index and Signals DataFrame index must match exactly.")

        timestamps = price_df.index
        n_days = len(timestamps)

        # Portfolio accounting storage arrays
        cash_series = np.zeros(n_days, dtype=np.float64)
        position_qty_series = np.zeros(n_days, dtype=np.float64)
        position_val_series = np.zeros(n_days, dtype=np.float64)
        total_equity_series = np.zeros(n_days, dtype=np.float64)

        current_cash = float(self.initial_capital)
        current_qty = 0.0
        trade_records: List[TradeRecord] = []
        trade_counter = 0

        # Pending signal event from previous day Close
        pending_signal_event: Optional[float] = None
        pending_signal_ts: Optional[pd.Timestamp] = None

        for t in range(n_days):
            current_ts = timestamps[t]
            open_price = float(price_df["open"].iloc[t])
            close_price = float(price_df["close"].iloc[t])

            # STEP A: Execute pending trade at Open of day t if pending signal exists
            if pending_signal_event is not None and not np.isnan(pending_signal_event):

                # 1. Pending BUY Entry (Signal = 1)
                if pending_signal_event == 1.0 and current_qty == 0.0:
                    exec_price = open_price * (1.0 + self.slippage_rate)
                    slippage_unit = open_price * self.slippage_rate
                    alloc_capital = current_cash * self.position_size

                    # Calculate quantity based on execution price (including slippage)
                    if self.allow_fractional:
                        qty = alloc_capital / (exec_price * (1.0 + self.transaction_cost_rate))
                    else:
                        qty = np.floor(
                            alloc_capital / (exec_price * (1.0 + self.transaction_cost_rate))
                        )

                    if qty > 0:
                        gross_val = qty * exec_price
                        cost = gross_val * self.transaction_cost_rate
                        total_outflow = gross_val + cost

                        # Handle floating-point rounding tolerance (e.g. 1000.0000000000001 vs 1000.0)
                        if total_outflow <= current_cash + 1e-8:
                            total_outflow = min(total_outflow, current_cash)
                            current_cash -= total_outflow
                            current_qty += qty
                            trade_counter += 1

                            trade_records.append(
                                TradeRecord(
                                    trade_id=trade_counter,
                                    signal_timestamp=pending_signal_ts,
                                    execution_timestamp=current_ts,
                                    side="BUY",
                                    execution_price=exec_price,
                                    quantity=qty,
                                    gross_value=gross_val,
                                    transaction_cost=cost,
                                    slippage_amount=slippage_unit,
                                    net_cash_flow=-total_outflow,
                                    position_after_trade=current_qty,
                                )
                            )

                # 2. Pending SELL Exit (Signal = -1)
                elif pending_signal_event == -1.0 and current_qty > 0.0:
                    exec_price = open_price * (1.0 - self.slippage_rate)
                    slippage_unit = open_price * self.slippage_rate
                    qty = current_qty

                    gross_val = qty * exec_price
                    cost = gross_val * self.transaction_cost_rate
                    net_inflow = gross_val - cost

                    current_cash += net_inflow
                    current_qty = 0.0
                    trade_counter += 1

                    trade_records.append(
                        TradeRecord(
                            trade_id=trade_counter,
                            signal_timestamp=pending_signal_ts,
                            execution_timestamp=current_ts,
                            side="SELL",
                            execution_price=exec_price,
                            quantity=qty,
                            gross_value=gross_val,
                            transaction_cost=cost,
                            slippage_amount=slippage_unit,
                            net_cash_flow=net_inflow,
                            position_after_trade=current_qty,
                        )
                    )

            # Clear pending execution state after attempting execution
            pending_signal_event = None
            pending_signal_ts = None

            # STEP B: Valuation at day t Close
            pos_val = current_qty * close_price
            tot_eq = current_cash + pos_val

            cash_series[t] = current_cash
            position_qty_series[t] = current_qty
            position_val_series[t] = pos_val
            total_equity_series[t] = tot_eq

            # STEP C: Capture signal event generated at day t Close for execution at day t+1 Open
            if "signal_event" in sig_df.columns:
                sig_event_val = sig_df["signal_event"].iloc[t]
                if not np.isnan(sig_event_val) and sig_event_val in (1.0, -1.0):
                    pending_signal_event = sig_event_val
                    pending_signal_ts = current_ts

        # Build Portfolio History DataFrame
        history_df = pd.DataFrame(
            {
                "cash": cash_series,
                "position_quantity": position_qty_series,
                "market_price": price_df["close"].values,
                "position_value": position_val_series,
                "total_equity": total_equity_series,
            },
            index=timestamps,
        )
        history_df["daily_return"] = history_df["total_equity"].pct_change()
        history_df["cumulative_return"] = (
            history_df["total_equity"] / self.initial_capital
        ) - 1.0

        # Build Trades DataFrame
        if trade_records:
            trades_df = pd.DataFrame([t.__dict__ for t in trade_records])
        else:
            trades_df = pd.DataFrame(
                columns=[
                    "trade_id",
                    "signal_timestamp",
                    "execution_timestamp",
                    "side",
                    "execution_price",
                    "quantity",
                    "gross_value",
                    "transaction_cost",
                    "slippage_amount",
                    "net_cash_flow",
                    "position_after_trade",
                ]
            )

        final_equity = float(history_df["total_equity"].iloc[-1])
        total_return = float(history_df["cumulative_return"].iloc[-1])
        total_costs = (
            float(trades_df["transaction_cost"].sum()) if not trades_df.empty else 0.0
        )
        entries_cnt = (
            int((trades_df["side"] == "BUY").sum()) if not trades_df.empty else 0
        )
        exits_cnt = (
            int((trades_df["side"] == "SELL").sum()) if not trades_df.empty else 0
        )

        logger.info(
            f"Backtest completed: Initial=${self.initial_capital:,.2f}, Final=${final_equity:,.2f}, "
            f"Return={total_return:.2%}, Trades={len(trades_df)}, Costs=${total_costs:,.2f}"
        )

        return BacktestResult(
            equity_curve=history_df["total_equity"],
            portfolio_history=history_df,
            trades=trades_df,
            final_equity=final_equity,
            total_return=total_return,
            total_transaction_costs=total_costs,
            total_trades_count=len(trades_df),
            entries_count=entries_cnt,
            exits_count=exits_cnt,
            execution_convention=self.EXECUTION_CONVENTION_STR,
        )


def run_buy_and_hold_benchmark(
    df: pd.DataFrame,
    initial_capital: float = 100000.0,
    transaction_cost_rate: float = 0.0005,
    slippage_rate: float = 0.0005,
    allow_fractional: bool = False,
) -> BacktestResult:
    """Run simple Buy & Hold Benchmark for comparison.

    Benchmark Execution Convention
    ------------------------------
    Enter at the first available Open (t=0) -> Hold position through end of test.
    (Note: This is distinct from strategy signals which execute at Open t+1).

    Parameters
    ----------
    df : pd.DataFrame
        OHLC market DataFrame.
    initial_capital : float, default 100000.0
        Starting cash capital.
    transaction_cost_rate : float, default 0.0005
        Proportional transaction cost rate.
    slippage_rate : float, default 0.0005
        Proportional slippage rate.
    allow_fractional : bool, default False
        Allow fractional units.

    Returns
    -------
    BacktestResult
        Benchmark backtest result.
    """
    validate_required_columns(df, ["open", "close"])
    benchmark_convention = (
        "Benchmark Execution: Enter at first available Open (t=0) -> Hold through test period"
    )

    timestamps = df.index
    n_days = len(timestamps)

    first_open = float(df["open"].iloc[0])
    first_ts = timestamps[0]

    # Calculate execution price with BUY slippage
    exec_price = first_open * (1.0 + slippage_rate)
    slippage_unit = first_open * slippage_rate

    if allow_fractional:
        qty = initial_capital / (exec_price * (1.0 + transaction_cost_rate))
    else:
        qty = np.floor(initial_capital / (exec_price * (1.0 + transaction_cost_rate)))

    gross_val = qty * exec_price
    cost = gross_val * transaction_cost_rate
    total_outflow = gross_val + cost
    remaining_cash = initial_capital - total_outflow

    trade = TradeRecord(
        trade_id=1,
        signal_timestamp=first_ts,
        execution_timestamp=first_ts,
        side="BUY",
        execution_price=exec_price,
        quantity=qty,
        gross_value=gross_val,
        transaction_cost=cost,
        slippage_amount=slippage_unit,
        net_cash_flow=-total_outflow,
        position_after_trade=qty,
    )

    trades_df = pd.DataFrame([trade.__dict__])

    # Portfolio daily tracking
    closes = df["close"].values
    cash_array = np.full(n_days, remaining_cash)
    qty_array = np.full(n_days, qty)
    pos_val_array = qty * closes
    equity_array = cash_array + pos_val_array

    history_df = pd.DataFrame(
        {
            "cash": cash_array,
            "position_quantity": qty_array,
            "market_price": closes,
            "position_value": pos_val_array,
            "total_equity": equity_array,
        },
        index=timestamps,
    )
    history_df["daily_return"] = history_df["total_equity"].pct_change()
    history_df["cumulative_return"] = (
        history_df["total_equity"] / initial_capital
    ) - 1.0

    final_eq = float(history_df["total_equity"].iloc[-1])
    tot_ret = float(history_df["cumulative_return"].iloc[-1])

    return BacktestResult(
        equity_curve=history_df["total_equity"],
        portfolio_history=history_df,
        trades=trades_df,
        final_equity=final_eq,
        total_return=tot_ret,
        total_transaction_costs=cost,
        total_trades_count=1,
        entries_count=1,
        exits_count=0,
        execution_convention=benchmark_convention,
    )
