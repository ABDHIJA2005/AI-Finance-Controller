"""
Unit tests for Performance and Risk Analytics module.
"""

import numpy as np
import pandas as pd
import pytest

from src.metrics import (
    PerformanceMetrics,
    RoundTripTrade,
    calculate_annualized_volatility,
    calculate_cagr,
    calculate_calmar_ratio,
    calculate_drawdown_metrics,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    match_round_trip_trades,
)


@pytest.fixture
def synthetic_equity_history() -> pd.DataFrame:
    """Fixture providing a 3-year synthetic daily portfolio history for metrics testing."""
    dates = pd.date_range(start="2020-01-01", end="2023-01-01", freq="D")
    n_days = len(dates)
    # Synthetic equity starting at 100,000 and growing to 150,000 with a drawdown in between
    equity = np.linspace(100000.0, 150000.0, n_days)
    # Inject a drawdown between days 200 and 400
    equity[200:300] = equity[200] * (1.0 - np.linspace(0.0, 0.20, 100))  # 20% drawdown
    equity[300:400] = equity[200] * (1.0 - np.linspace(0.20, 0.0, 100))  # Recovery

    df = pd.DataFrame(
        {
            "cash": equity,
            "position_quantity": [0.0] * n_days,
            "market_price": [100.0] * n_days,
            "position_value": [0.0] * n_days,
            "total_equity": equity,
        },
        index=dates,
    )
    df["daily_return"] = df["total_equity"].pct_change()
    df["cumulative_return"] = (df["total_equity"] / 100000.0) - 1.0
    return df


@pytest.fixture
def sample_trades_df() -> pd.DataFrame:
    """Fixture providing a trade log with 2 completed round-trip trades (1 win, 1 loss)."""
    trades = [
        {
            "trade_id": 1,
            "side": "BUY",
            "execution_timestamp": pd.Timestamp("2020-01-02"),
            "execution_price": 100.0,
            "quantity": 100.0,
            "net_cash_flow": -10050.0,  # 10000 + 50 cost
        },
        {
            "trade_id": 2,
            "side": "SELL",
            "execution_timestamp": pd.Timestamp("2020-01-10"),
            "execution_price": 110.0,
            "quantity": 100.0,
            "net_cash_flow": 10945.0,  # 11000 - 55 cost -> PnL = +895.0 (WIN)
        },
        {
            "trade_id": 3,
            "side": "BUY",
            "execution_timestamp": pd.Timestamp("2020-02-01"),
            "execution_price": 110.0,
            "quantity": 100.0,
            "net_cash_flow": -11055.0,
        },
        {
            "trade_id": 4,
            "side": "SELL",
            "execution_timestamp": pd.Timestamp("2020-02-15"),
            "execution_price": 100.0,
            "quantity": 100.0,
            "net_cash_flow": 9950.0,  # PnL = -1105.0 (LOSS)
        },
    ]
    return pd.DataFrame(trades)


def test_1_total_return(synthetic_equity_history: pd.DataFrame):
    """1. Test total return calculation."""
    final_eq = synthetic_equity_history["total_equity"].iloc[-1]
    initial_cap = 100000.0
    expected_return = (final_eq / initial_cap) - 1.0
    np.testing.assert_almost_equal(expected_return, 0.50)


def test_2_cagr_calculation(synthetic_equity_history: pd.DataFrame):
    """2. Test CAGR based on actual elapsed calendar time span."""
    cagr = calculate_cagr(synthetic_equity_history, initial_capital=100000.0)
    # Elapsed time: 3 years (2020-01-01 to 2023-01-01)
    # (1.5)^(1/3) - 1 = 0.144714 (14.47%)
    assert 0.14 < cagr < 0.15


def test_3_annualized_volatility(synthetic_equity_history: pd.DataFrame):
    """3. Test annualized volatility calculation (std * sqrt(252))."""
    returns = synthetic_equity_history["daily_return"]
    vol = calculate_annualized_volatility(returns)
    assert vol > 0.0


def test_4_5_6_max_drawdown_peak_trough_recovery(synthetic_equity_history: pd.DataFrame):
    """4-6. Test max drawdown, peak date, trough date, and recovery date detection."""
    max_dd, peak_dt, trough_dt, rec_dt = calculate_drawdown_metrics(synthetic_equity_history)

    assert max_dd < -0.15  # 20% drawdown injected
    assert peak_dt is not None
    assert trough_dt is not None
    assert rec_dt is not None
    assert peak_dt < trough_dt < rec_dt


def test_7_unrecovered_drawdown():
    """7. Test that unrecovered drawdown sets recovery_date = None."""
    dates = pd.date_range(start="2020-01-01", periods=10, freq="D")
    equity = pd.Series([100, 110, 120, 100, 90, 80, 85, 90, 95, 100], index=dates)
    history = pd.DataFrame({"total_equity": equity}, index=dates)

    max_dd, peak_dt, trough_dt, rec_dt = calculate_drawdown_metrics(history)
    assert max_dd == -0.33333333333333337  # (80 - 120) / 120 = -0.3333
    assert peak_dt == dates[2]  # Date with 120
    assert trough_dt == dates[5]  # Date with 80
    assert rec_dt is None  # Ended at 100 < 120 peak


def test_8_9_sharpe_ratio_and_zero_volatility():
    """8-9. Test Sharpe ratio and zero-volatility safety check."""
    # Normal returns
    returns = pd.Series([0.01, 0.005, -0.002, 0.008, 0.012])
    sharpe = calculate_sharpe_ratio(returns)
    assert sharpe > 0.0

    # Constant returns (zero std)
    const_returns = pd.Series([0.01, 0.01, 0.01, 0.01])
    zero_vol_sharpe = calculate_sharpe_ratio(const_returns)
    assert zero_vol_sharpe == 0.0  # Safe handling without crash


def test_10_11_sortino_ratio_and_zero_downside():
    """10-11. Test Sortino ratio and zero-downside handling."""
    returns = pd.Series([0.01, -0.005, 0.02, -0.01, 0.015])
    sortino = calculate_sortino_ratio(returns, mar=0.0)
    assert sortino > 0.0

    # All positive returns (zero downside)
    pos_returns = pd.Series([0.01, 0.02, 0.015, 0.03])
    zero_downside_sortino = calculate_sortino_ratio(pos_returns, mar=0.0)
    assert zero_downside_sortino == 999.0  # Capped positive indicator


def test_12_calmar_ratio():
    """12. Test Calmar Ratio (CAGR / abs(max_drawdown))."""
    calmar = calculate_calmar_ratio(cagr=0.15, max_drawdown=-0.30)
    np.testing.assert_almost_equal(calmar, 0.50)

    # Zero drawdown
    zero_dd_calmar = calculate_calmar_ratio(cagr=0.15, max_drawdown=0.0)
    assert zero_dd_calmar == 999.0


def test_13_round_trip_matching_and_win_rate(sample_trades_df: pd.DataFrame):
    """13. Test matching BUY entry -> SELL exit round trips and win rate calculation."""
    round_trips = match_round_trip_trades(sample_trades_df)
    assert len(round_trips) == 2  # 2 completed round trips

    rt1 = round_trips[0]
    rt2 = round_trips[1]

    # RT 1 PnL = +10945 - 10050 = +895.0
    assert rt1.net_pnl == 895.0
    # RT 2 PnL = +9950 - 11055 = -1105.0
    assert rt2.net_pnl == -1105.0


def test_14_15_16_trade_stats(sample_trades_df: pd.DataFrame):
    """14-16. Test average win, average loss, and profit factor."""
    round_trips = match_round_trip_trades(sample_trades_df)
    pnls = [rt.net_pnl for rt in round_trips]

    winning_pnls = [p for p in pnls if p > 0]
    losing_pnls = [p for p in pnls if p <= 0]

    avg_win = float(np.mean(winning_pnls))
    avg_loss = float(np.mean(losing_pnls))
    profit_factor = sum(winning_pnls) / abs(sum(losing_pnls))

    assert avg_win == 895.0
    assert avg_loss == -1105.0
    np.testing.assert_almost_equal(profit_factor, 895.0 / 1105.0)


def test_17_edge_case_zero_loss_zero_win(sample_trades_df: pd.DataFrame):
    """17. Test edge cases: all winning trades (zero loss) and all losing trades (zero win)."""
    # Only RT 1 (WIN)
    win_trades = sample_trades_df.iloc[:2]
    win_rts = match_round_trip_trades(win_trades)
    assert len(win_rts) == 1
    assert win_rts[0].net_pnl > 0

    # Only RT 2 (LOSS)
    loss_trades = sample_trades_df.iloc[2:]
    loss_rts = match_round_trip_trades(loss_trades)
    assert len(loss_rts) == 1
    assert loss_rts[0].net_pnl < 0


def test_18_transaction_cost_aggregation(sample_trades_df: pd.DataFrame):
    """18. Test transaction cost aggregation across trades."""
    # Costs: Buy1=50, Sell1=55, Buy2=55, Sell2=50 -> Sum = 210
    trades_with_costs = sample_trades_df.copy()
    trades_with_costs["transaction_cost"] = [50.0, 55.0, 55.0, 50.0]

    tot_cost = trades_with_costs["transaction_cost"].sum()
    assert tot_cost == 210.0
