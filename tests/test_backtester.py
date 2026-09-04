"""
Unit tests for Backtester engine, accounting, friction modeling, and timing.
"""

import numpy as np
import pandas as pd
import pytest

from src.backtest import Backtester, run_buy_and_hold_benchmark


@pytest.fixture
def synthetic_ohlc_5d() -> pd.DataFrame:
    """Fixture providing a synthetic 5-day OHLC DataFrame for backtester unit testing."""
    dates = pd.date_range(start="2024-01-01", periods=5, freq="D")
    df = pd.DataFrame(
        {
            "open": [100.0, 102.0, 104.0, 103.0, 105.0],
            "high": [105.0, 106.0, 107.0, 108.0, 110.0],
            "low": [98.0, 100.0, 102.0, 101.0, 104.0],
            "close": [103.0, 105.0, 102.0, 106.0, 108.0],
            "volume": [1000] * 5,
        },
        index=dates,
    )
    df.index.name = "timestamp"
    return df


def test_1_initial_portfolio_state(synthetic_ohlc_5d: pd.DataFrame):
    """1. Test initial portfolio state before any signal events occur."""
    bt = Backtester(initial_capital=100000.0)
    signals = pd.DataFrame({"signal_event": [0.0] * 5}, index=synthetic_ohlc_5d.index)
    res = bt.run(synthetic_ohlc_5d, signals)

    assert res.final_equity == 100000.0
    assert res.total_return == 0.0
    assert len(res.trades) == 0
    assert (res.portfolio_history["cash"] == 100000.0).all()
    assert (res.portfolio_history["position_quantity"] == 0.0).all()


def test_2_single_entry_execution(synthetic_ohlc_5d: pd.DataFrame):
    """2. Test single entry execution at t+1 Open."""
    bt = Backtester(
        initial_capital=100000.0,
        position_size=1.0,
        transaction_cost_rate=0.0,
        slippage_rate=0.0,
        allow_fractional=False,
    )
    # Signal at Close of day 0 (t=0)
    signals = pd.DataFrame({"signal_event": [1.0, 0.0, 0.0, 0.0, 0.0]}, index=synthetic_ohlc_5d.index)
    res = bt.run(synthetic_ohlc_5d, signals)

    assert len(res.trades) == 1
    trade = res.trades.iloc[0]

    # Signal generated at day 0 Close (2024-01-01), executed at day 1 Open (2024-01-02)
    assert trade["signal_timestamp"] == synthetic_ohlc_5d.index[0]
    assert trade["execution_timestamp"] == synthetic_ohlc_5d.index[1]
    assert trade["side"] == "BUY"
    assert trade["execution_price"] == 102.0  # Day 1 Open price


def test_3_single_entry_exit_cycle(synthetic_ohlc_5d: pd.DataFrame):
    """3. Test a complete single entry + exit trade cycle."""
    bt = Backtester(
        initial_capital=100000.0,
        transaction_cost_rate=0.0,
        slippage_rate=0.0,
        allow_fractional=False,
    )
    # Entry signal at t=0 Close, Exit signal at t=2 Close
    signals = pd.DataFrame({"signal_event": [1.0, 0.0, -1.0, 0.0, 0.0]}, index=synthetic_ohlc_5d.index)
    res = bt.run(synthetic_ohlc_5d, signals)

    assert len(res.trades) == 2
    buy_trade = res.trades.iloc[0]
    sell_trade = res.trades.iloc[1]

    # BUY at t=1 Open (102.0), SELL at t=3 Open (103.0)
    assert buy_trade["execution_timestamp"] == synthetic_ohlc_5d.index[1]
    assert buy_trade["execution_price"] == 102.0

    assert sell_trade["execution_timestamp"] == synthetic_ohlc_5d.index[3]
    assert sell_trade["execution_price"] == 103.0
    assert res.portfolio_history["position_quantity"].iloc[3] == 0.0


def test_4_multiple_trades(synthetic_ohlc_5d: pd.DataFrame):
    """4. Test multiple sequential entry/exit trade cycles."""
    bt = Backtester(initial_capital=100000.0, transaction_cost_rate=0.0, slippage_rate=0.0)
    # Entry at t=0, Exit at t=1, Entry at t=2, Exit at t=3
    signals = pd.DataFrame({"signal_event": [1.0, -1.0, 1.0, -1.0, 0.0]}, index=synthetic_ohlc_5d.index)
    res = bt.run(synthetic_ohlc_5d, signals)

    assert len(res.trades) == 4
    assert res.entries_count == 2
    assert res.exits_count == 2


def test_5_next_day_execution_timing(synthetic_ohlc_5d: pd.DataFrame):
    """5. Test that signal at Close t executes at Open t+1, NOT Close t or Open t."""
    bt = Backtester(transaction_cost_rate=0.0, slippage_rate=0.0)
    signals = pd.DataFrame({"signal_event": [0.0, 1.0, 0.0, 0.0, 0.0]}, index=synthetic_ohlc_5d.index)
    res = bt.run(synthetic_ohlc_5d, signals)

    trade = res.trades.iloc[0]
    # Signal at t=1 Close (2024-01-02), Executed at t=2 Open (2024-01-03, price=104.0)
    assert trade["signal_timestamp"] == synthetic_ohlc_5d.index[1]
    assert trade["execution_timestamp"] == synthetic_ohlc_5d.index[2]
    assert trade["execution_price"] == 104.0


def test_6_transaction_cost_calculation(synthetic_ohlc_5d: pd.DataFrame):
    """6. Test transaction cost calculation (cost = gross_value * cost_rate)."""
    cost_rate = 0.01  # 1%
    bt = Backtester(
        initial_capital=10000.0,
        position_size=1.0,
        transaction_cost_rate=cost_rate,
        slippage_rate=0.0,
        allow_fractional=True,
    )
    signals = pd.DataFrame({"signal_event": [1.0, 0.0, 0.0, 0.0, 0.0]}, index=synthetic_ohlc_5d.index)
    res = bt.run(synthetic_ohlc_5d, signals)

    trade = res.trades.iloc[0]
    expected_cost = trade["gross_value"] * cost_rate
    np.testing.assert_almost_equal(trade["transaction_cost"], expected_cost)
    assert res.total_transaction_costs == expected_cost


def test_7_buy_slippage_direction(synthetic_ohlc_5d: pd.DataFrame):
    """7. Test BUY slippage increases execution price (Open * (1 + slippage))."""
    slippage = 0.02  # 2%
    bt = Backtester(transaction_cost_rate=0.0, slippage_rate=slippage)
    signals = pd.DataFrame({"signal_event": [1.0, 0.0, 0.0, 0.0, 0.0]}, index=synthetic_ohlc_5d.index)
    res = bt.run(synthetic_ohlc_5d, signals)

    trade = res.trades.iloc[0]
    raw_open = synthetic_ohlc_5d["open"].iloc[1]  # 102.0
    expected_price = raw_open * (1.0 + slippage)
    np.testing.assert_almost_equal(trade["execution_price"], expected_price)


def test_8_sell_slippage_direction(synthetic_ohlc_5d: pd.DataFrame):
    """8. Test SELL slippage decreases execution price (Open * (1 - slippage))."""
    slippage = 0.02  # 2%
    bt = Backtester(transaction_cost_rate=0.0, slippage_rate=slippage)
    signals = pd.DataFrame({"signal_event": [1.0, 0.0, -1.0, 0.0, 0.0]}, index=synthetic_ohlc_5d.index)
    res = bt.run(synthetic_ohlc_5d, signals)

    sell_trade = res.trades.iloc[1]
    raw_open = synthetic_ohlc_5d["open"].iloc[3]  # 103.0
    expected_price = raw_open * (1.0 - slippage)
    np.testing.assert_almost_equal(sell_trade["execution_price"], expected_price)


def test_9_position_sizing(synthetic_ohlc_5d: pd.DataFrame):
    """9. Test integer vs fractional position sizing."""
    bt_int = Backtester(initial_capital=1000.0, position_size=1.0, transaction_cost_rate=0.0, slippage_rate=0.0, allow_fractional=False)
    bt_frac = Backtester(initial_capital=1000.0, position_size=1.0, transaction_cost_rate=0.0, slippage_rate=0.0, allow_fractional=True)

    signals = pd.DataFrame({"signal_event": [1.0, 0.0, 0.0, 0.0, 0.0]}, index=synthetic_ohlc_5d.index)
    
    res_int = bt_int.run(synthetic_ohlc_5d, signals)
    res_frac = bt_frac.run(synthetic_ohlc_5d, signals)

    # Day 1 Open = 102.0. Integer qty = floor(1000/102) = 9.0
    assert res_int.trades.iloc[0]["quantity"] == 9.0
    # Fractional qty = 1000/102 = 9.8039...
    assert res_frac.trades.iloc[0]["quantity"] > 9.5


def test_10_insufficient_capital(synthetic_ohlc_5d: pd.DataFrame):
    """10. Test that no trade occurs if capital is insufficient."""
    bt = Backtester(initial_capital=50.0, allow_fractional=False)  # 50 < 102.0
    signals = pd.DataFrame({"signal_event": [1.0, 0.0, 0.0, 0.0, 0.0]}, index=synthetic_ohlc_5d.index)
    res = bt.run(synthetic_ohlc_5d, signals)

    assert len(res.trades) == 0


def test_11_end_of_data_open_position(synthetic_ohlc_5d: pd.DataFrame):
    """11. Test that signal on final day N-1 Close does not execute."""
    bt = Backtester(transaction_cost_rate=0.0, slippage_rate=0.0)
    # Signal at final day t=4 Close
    signals = pd.DataFrame({"signal_event": [0.0, 0.0, 0.0, 0.0, 1.0]}, index=synthetic_ohlc_5d.index)
    res = bt.run(synthetic_ohlc_5d, signals)

    # No day t=5 Open exists, so trade must NOT execute
    assert len(res.trades) == 0


def test_12_cash_accounting(synthetic_ohlc_5d: pd.DataFrame):
    """12. Test cash balance updates after entry and exit."""
    bt = Backtester(initial_capital=10000.0, transaction_cost_rate=0.001, slippage_rate=0.0, allow_fractional=False)
    signals = pd.DataFrame({"signal_event": [1.0, 0.0, -1.0, 0.0, 0.0]}, index=synthetic_ohlc_5d.index)
    res = bt.run(synthetic_ohlc_5d, signals)

    buy_trade = res.trades.iloc[0]
    sell_trade = res.trades.iloc[1]

    # Cash after buy = 10000 + buy_net_cash_flow
    expected_cash_after_buy = 10000.0 + buy_trade["net_cash_flow"]
    np.testing.assert_almost_equal(res.portfolio_history["cash"].iloc[1], expected_cash_after_buy)

    # Final cash = expected_cash_after_buy + sell_net_cash_flow
    expected_final_cash = expected_cash_after_buy + sell_trade["net_cash_flow"]
    np.testing.assert_almost_equal(res.portfolio_history["cash"].iloc[3], expected_final_cash)


def test_13_portfolio_equity_calculation(synthetic_ohlc_5d: pd.DataFrame):
    """13. Test daily total equity = cash + position_quantity * close_t."""
    bt = Backtester(initial_capital=10000.0, transaction_cost_rate=0.0, slippage_rate=0.0)
    signals = pd.DataFrame({"signal_event": [1.0, 0.0, 0.0, 0.0, 0.0]}, index=synthetic_ohlc_5d.index)
    res = bt.run(synthetic_ohlc_5d, signals)

    hist = res.portfolio_history
    for t in range(len(hist)):
        expected_eq = hist["cash"].iloc[t] + hist["position_quantity"].iloc[t] * hist["market_price"].iloc[t]
        np.testing.assert_almost_equal(hist["total_equity"].iloc[t], expected_eq)


def test_14_signal_vs_execution_timestamp_distinct(synthetic_ohlc_5d: pd.DataFrame):
    """14. Test that signal_timestamp and execution_timestamp are distinct in trade log."""
    bt = Backtester()
    signals = pd.DataFrame({"signal_event": [1.0, 0.0, 0.0, 0.0, 0.0]}, index=synthetic_ohlc_5d.index)
    res = bt.run(synthetic_ohlc_5d, signals)

    trade = res.trades.iloc[0]
    assert trade["signal_timestamp"] != trade["execution_timestamp"]
    assert trade["signal_timestamp"] < trade["execution_timestamp"]


def test_15_no_look_ahead_execution_test(synthetic_ohlc_5d: pd.DataFrame):
    """15. Test that signal generated on day t Close cannot execute before day t+1 Open."""
    bt = Backtester()
    signals = pd.DataFrame({"signal_event": [0.0, 0.0, 1.0, 0.0, 0.0]}, index=synthetic_ohlc_5d.index)
    res = bt.run(synthetic_ohlc_5d, signals)

    # Position at day 2 Close (t=2) must still be 0.0
    assert res.portfolio_history["position_quantity"].iloc[2] == 0.0
    # Position at day 3 Open (t=3) becomes active
    assert res.portfolio_history["position_quantity"].iloc[3] > 0.0


def test_16_non_mutation_of_inputs(synthetic_ohlc_5d: pd.DataFrame):
    """16. Test that backtester does not mutate input DataFrames."""
    cols_price = list(synthetic_ohlc_5d.columns)
    signals = pd.DataFrame({"signal_event": [1.0, 0.0, -1.0, 0.0, 0.0]}, index=synthetic_ohlc_5d.index)
    cols_sig = list(signals.columns)

    bt = Backtester()
    bt.run(synthetic_ohlc_5d, signals)

    assert list(synthetic_ohlc_5d.columns) == cols_price
    assert list(signals.columns) == cols_sig


def test_buy_and_hold_benchmark(synthetic_ohlc_5d: pd.DataFrame):
    """Test Buy & Hold benchmark execution at first available Open (t=0)."""
    res = run_buy_and_hold_benchmark(
        synthetic_ohlc_5d,
        initial_capital=100000.0,
        transaction_cost_rate=0.0,
        slippage_rate=0.0,
    )
    assert len(res.trades) == 1
    trade = res.trades.iloc[0]
    assert trade["execution_timestamp"] == synthetic_ohlc_5d.index[0]
    assert trade["execution_price"] == 100.0  # Day 0 Open price
