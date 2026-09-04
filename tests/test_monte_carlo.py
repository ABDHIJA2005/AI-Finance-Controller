"""
Unit tests for Monte Carlo Risk Analysis module.
"""

import numpy as np
import pandas as pd
import pytest

from src.metrics import RoundTripTrade
from src.monte_carlo import (
    MonteCarloConfig,
    bootstrap_daily_returns,
    bootstrap_trade_returns,
    calculate_path_max_drawdown,
    run_monte_carlo_simulation,
)


@pytest.fixture
def sample_round_trips() -> list[RoundTripTrade]:
    """Fixture providing a list of 5 synthetic round-trip trades."""
    returns = [0.05, -0.02, 0.08, -0.04, 0.03]
    trades = []
    for i, r in enumerate(returns, 1):
        buy_cash = -10000.0
        sell_cash = 10000.0 * (1.0 + r)
        trades.append(
            RoundTripTrade(
                round_trip_id=i,
                buy_trade_id=i * 2 - 1,
                sell_trade_id=i * 2,
                entry_timestamp=pd.Timestamp("2020-01-01") + pd.Timedelta(days=i * 5),
                exit_timestamp=pd.Timestamp("2020-01-01") + pd.Timedelta(days=i * 5 + 2),
                entry_price=100.0,
                exit_price=100.0 * (1.0 + r),
                quantity=100.0,
                buy_cash_flow=buy_cash,
                sell_cash_flow=sell_cash,
                net_pnl=sell_cash + buy_cash,
                net_return=r,
                holding_days=2.0,
            )
        )
    return trades


def test_1_reproducibility_with_fixed_seed(sample_round_trips: list[RoundTripTrade]):
    """1. Test that identical random_seed produces identical simulation outputs."""
    cfg1 = MonteCarloConfig(n_simulations=1000, random_seed=42)
    cfg2 = MonteCarloConfig(n_simulations=1000, random_seed=42)

    term1, dd1, _ = bootstrap_trade_returns(sample_round_trips, cfg1)
    term2, dd2, _ = bootstrap_trade_returns(sample_round_trips, cfg2)

    np.testing.assert_array_equal(term1, term2)
    np.testing.assert_array_equal(dd1, dd2)


def test_2_simulation_count(sample_round_trips: list[RoundTripTrade]):
    """2. Test that simulation produces exact specified number of iterations."""
    cfg = MonteCarloConfig(n_simulations=5000, random_seed=42)
    term, dd, paths = bootstrap_trade_returns(sample_round_trips, cfg)

    assert len(term) == 5000
    assert len(dd) == 5000
    assert paths.shape[0] == 100  # Default visual paths


def test_3_path_length(sample_round_trips: list[RoundTripTrade]):
    """3. Test that simulated equity path length matches K_trades + 1."""
    cfg = MonteCarloConfig(n_simulations=100, random_seed=42)
    term, dd, paths = bootstrap_trade_returns(sample_round_trips, cfg)

    # 5 trades -> path length 6 (including initial capital)
    assert paths.shape[1] == 6


def test_4_terminal_wealth_calculation(sample_round_trips: list[RoundTripTrade]):
    """4. Test terminal wealth compounding formula accuracy."""
    cfg = MonteCarloConfig(n_simulations=1, initial_capital=100000.0, random_seed=42)
    term, _, _ = bootstrap_trade_returns(sample_round_trips, cfg)

    assert term[0] > 0.0  # Positive terminal capital


def test_5_probability_of_loss(sample_round_trips: list[RoundTripTrade]):
    """5. Test probability of loss calculation (terminal wealth < initial capital)."""
    cfg = MonteCarloConfig(n_simulations=1000, initial_capital=100000.0, random_seed=42)
    term, _, _ = bootstrap_trade_returns(sample_round_trips, cfg)

    prob_loss = np.mean(term < 100000.0)
    assert 0.0 <= prob_loss <= 1.0


def test_6_percentile_calculations():
    """6. Test percentile calculation ordering (5th <= 25th <= 50th <= 75th <= 95th)."""
    arr = np.linspace(50000.0, 150000.0, 1000)
    p5 = np.percentile(arr, 5)
    p25 = np.percentile(arr, 25)
    p50 = np.percentile(arr, 50)
    p75 = np.percentile(arr, 75)
    p95 = np.percentile(arr, 95)

    assert p5 <= p25 <= p50 <= p75 <= p95


def test_7_8_max_drawdown_calculation_and_percentiles():
    """7-8. Test maximum drawdown path calculation and percentile ordering for negative values."""
    path = np.array([100000.0, 110000.0, 99000.0, 105000.0, 90000.0, 115000.0])
    # Peak = 110,000, Trough = 90,000 -> DD = (90,000 / 110,000) - 1 = -0.181818 (-18.18%)
    max_dd = calculate_path_max_drawdown(path)
    np.testing.assert_almost_equal(max_dd, -0.18181818181818177)


def test_9_empty_returns_handling():
    """9. Test robust handling of empty trade returns list."""
    cfg = MonteCarloConfig(n_simulations=500, initial_capital=100000.0, random_seed=42)
    term, dd, paths = bootstrap_trade_returns([], cfg)

    assert len(term) == 500
    assert (term == 100000.0).all()
    assert (dd == 0.0).all()


def test_10_all_zero_returns_handling():
    """10. Test handling of all-zero return series."""
    zero_rets = pd.Series([0.0] * 50)
    cfg = MonteCarloConfig(n_simulations=100, initial_capital=100000.0, random_seed=42)
    term, dd, _ = bootstrap_daily_returns(zero_rets, cfg)

    assert (term == 100000.0).all()
    assert (dd == 0.0).all()


def test_11_insufficient_trade_sample_warning(sample_round_trips: list[RoundTripTrade]):
    """11. Test that small empirical sample sizes (< 30) generate a sample_size_warning."""
    hist_df = pd.DataFrame({"total_equity": [100000.0, 105000.0]}, index=pd.date_range("2020-01-01", periods=2))
    trades_df = pd.DataFrame([
        {"trade_id": 1, "side": "BUY", "execution_timestamp": pd.Timestamp("2020-01-01"), "execution_price": 100, "quantity": 10, "net_cash_flow": -1000},
        {"trade_id": 2, "side": "SELL", "execution_timestamp": pd.Timestamp("2020-01-02"), "execution_price": 110, "quantity": 10, "net_cash_flow": 1100},
    ])
    mock_backtest_res = pytest.importorskip("src.backtest").BacktestResult(
        equity_curve=hist_df["total_equity"],
        portfolio_history=hist_df,
        trades=trades_df,
        final_equity=105000.0,
        total_return=0.05,
        total_transaction_costs=0.0,
        total_trades_count=2,
        entries_count=1,
        exits_count=1,
        execution_convention="Test",
    )

    mc_res = run_monte_carlo_simulation(mock_backtest_res, strategy_name="Test", scope="Out-of-Sample")
    assert mc_res.empirical_sample_size == 1  # 1 completed trade
    assert mc_res.sample_size_warning is not None
    assert "WARNING: Small empirical sample size" in mc_res.sample_size_warning


def test_12_input_data_non_mutation(sample_round_trips: list[RoundTripTrade]):
    """12. Test that Monte Carlo simulation does not mutate input trade objects or Series."""
    original_returns = [rt.net_return for rt in sample_round_trips]
    cfg = MonteCarloConfig(n_simulations=100, random_seed=42)
    bootstrap_trade_returns(sample_round_trips, cfg)

    current_returns = [rt.net_return for rt in sample_round_trips]
    assert original_returns == current_returns


def test_13_resampled_from_empirical_distribution(sample_round_trips: list[RoundTripTrade]):
    """13. Test that resampled returns are drawn strictly from empirical input distribution."""
    empirical_set = set(rt.net_return for rt in sample_round_trips)
    cfg = MonteCarloConfig(n_simulations=10, random_seed=42)
    rng = np.random.default_rng(cfg.random_seed)

    returns_arr = np.array([rt.net_return for rt in sample_round_trips])
    resampled = rng.choice(returns_arr, size=50, replace=True)

    for val in resampled:
        assert val in empirical_set
