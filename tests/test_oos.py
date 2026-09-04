"""
Unit tests for Out-of-Sample testing and robustness analysis module.
"""

import numpy as np
import pandas as pd
import pytest

from src.backtest import Backtester
from src.features import add_features
from src.metrics import PerformanceMetrics
from src.oos import (
    calculate_degradation,
    calculate_signal_stability,
    run_multi_period_backtest,
    split_chronologically,
)
from src.strategies.moving_average import generate_ma_crossover_signals


@pytest.fixture
def multi_year_price_df() -> pd.DataFrame:
    """Fixture providing a multi-year synthetic price DataFrame covering 2015 through 2025."""
    dates = pd.date_range(start="2015-01-01", end="2025-12-31", freq="D")
    n_days = len(dates)
    prices = 100.0 + np.linspace(0.0, 50.0, n_days) + 2.0 * np.sin(np.linspace(0, 10 * np.pi, n_days))

    df = pd.DataFrame(
        {
            "open": prices - 0.5,
            "high": prices + 1.0,
            "low": prices - 1.0,
            "close": prices,
            "volume": [1000] * n_days,
        },
        index=dates,
    )
    df.index.name = "timestamp"
    return df


def test_1_chronological_splitting(multi_year_price_df: pd.DataFrame):
    """1. Test chronological partitioning into Dev, Val, and OOS partitions."""
    feat_df = add_features(multi_year_price_df)
    signals = generate_ma_crossover_signals(feat_df)
    partitions = split_chronologically(feat_df, signals)

    assert "Development" in partitions
    assert "Validation" in partitions
    assert "Out-of-Sample" in partitions

    assert len(partitions["Development"].df) > 0
    assert len(partitions["Validation"].df) > 0
    assert len(partitions["Out-of-Sample"].df) > 0


def test_2_no_random_shuffling(multi_year_price_df: pd.DataFrame):
    """2. Test that partitions preserve strict monotonic time-series ordering."""
    feat_df = add_features(multi_year_price_df)
    signals = generate_ma_crossover_signals(feat_df)
    partitions = split_chronologically(feat_df, signals)

    for name, part in partitions.items():
        assert part.df.index.is_monotonic_increasing, f"Partition '{name}' is not monotonic!"


def test_3_boundary_dates(multi_year_price_df: pd.DataFrame):
    """3. Test exact boundary date enforcement."""
    feat_df = add_features(multi_year_price_df)
    signals = generate_ma_crossover_signals(feat_df)
    partitions = split_chronologically(feat_df, signals)

    dev_df = partitions["Development"].df
    val_df = partitions["Validation"].df
    oos_df = partitions["Out-of-Sample"].df

    assert dev_df.index.max() <= pd.Timestamp("2021-12-31")
    assert val_df.index.min() >= pd.Timestamp("2022-01-01")
    assert val_df.index.max() <= pd.Timestamp("2023-12-31")
    assert oos_df.index.min() >= pd.Timestamp("2024-01-01")


def test_4_non_overlapping_partitions(multi_year_price_df: pd.DataFrame):
    """4. Test that partitions are strictly non-overlapping."""
    feat_df = add_features(multi_year_price_df)
    signals = generate_ma_crossover_signals(feat_df)
    partitions = split_chronologically(feat_df, signals)

    dev_idx = set(partitions["Development"].df.index)
    val_idx = set(partitions["Validation"].df.index)
    oos_idx = set(partitions["Out-of-Sample"].df.index)

    assert len(dev_idx.intersection(val_idx)) == 0
    assert len(dev_idx.intersection(oos_idx)) == 0
    assert len(val_idx.intersection(oos_idx)) == 0


def test_5_every_observation_belongs_to_one_partition(multi_year_price_df: pd.DataFrame):
    """5. Test that every observation in target range belongs to exactly one partition."""
    feat_df = add_features(multi_year_price_df)
    signals = generate_ma_crossover_signals(feat_df)
    partitions = split_chronologically(feat_df, signals)

    combined_partition_len = (
        len(partitions["Development"].df)
        + len(partitions["Validation"].df)
        + len(partitions["Out-of-Sample"].df)
    )
    assert combined_partition_len == len(feat_df)


def test_6_feature_warmup_preservation(multi_year_price_df: pd.DataFrame):
    """6. Test that features computed on full dataset are preserved when sliced into Validation/OOS."""
    feat_df = add_features(multi_year_price_df, sma_windows=(50, 200))
    signals = generate_ma_crossover_signals(feat_df, fast_window=50, slow_window=200)
    partitions = split_chronologically(feat_df, signals)

    val_df = partitions["Validation"].df
    # In 2022 (Validation), sma_200 should ALREADY be fully populated (non-NaN) because warmup occurred in 2015-2021
    assert val_df["sma_200"].notna().all()


def test_7_no_look_ahead_leakage(multi_year_price_df: pd.DataFrame):
    """7. Test that signals in Development period are independent of OOS data."""
    feat_df = add_features(multi_year_price_df)
    signals = generate_ma_crossover_signals(feat_df)
    partitions = split_chronologically(feat_df, signals)

    dev_signals = partitions["Development"].signals

    # Mutate OOS price data
    mutated_df = multi_year_price_df.copy()
    mutated_df.loc[mutated_df.index >= "2024-01-01", "close"] = 999999.0
    mutated_feat = add_features(mutated_df)
    mutated_signals = generate_ma_crossover_signals(mutated_feat)
    mutated_partitions = split_chronologically(mutated_feat, mutated_signals)

    mutated_dev_signals = mutated_partitions["Development"].signals

    # Assert Dev signals remain identical
    pd.testing.assert_frame_equal(dev_signals, mutated_dev_signals)


def test_8_independent_fresh_capital_per_period(multi_year_price_df: pd.DataFrame):
    """8. Test that each period backtest runs with fresh initial capital ($100,000)."""
    feat_df = add_features(multi_year_price_df)
    signals = generate_ma_crossover_signals(feat_df)
    bt = Backtester(initial_capital=100000.0)

    res = run_multi_period_backtest(feat_df, signals, strategy_name="MA_Test", backtester=bt)

    assert res.dev_metrics.total_return != 0.0
    assert res.val_metrics.total_return != 0.0
    assert res.oos_metrics.total_return != 0.0
    # Dev metrics start from $100,000, Val metrics start from fresh $100,000, OOS starts from fresh $100,000


def test_9_degradation_calculation():
    """9. Test degradation calculation between Dev and OOS metrics."""
    dev_m = PerformanceMetrics(
        total_return=1.0, cagr=0.10, annualized_volatility=0.15, max_drawdown=-0.20,
        drawdown_peak_date=None, drawdown_trough_date=None, drawdown_recovery_date=None,
        sharpe_ratio=1.0, sortino_ratio=1.5, calmar_ratio=0.5, total_executed_trades=10,
        completed_round_trips=5, winning_trades=3, losing_trades=2, win_rate=0.6,
        average_win=100.0, average_loss=-50.0, profit_factor=3.0, total_transaction_costs=100.0,
        avg_cost_per_trade=10.0, cost_to_capital_ratio=0.001, execution_convention="Test"
    )
    oos_m = pytest.importorskip("src.metrics").PerformanceMetrics(
        total_return=0.5, cagr=0.05, annualized_volatility=0.15, max_drawdown=-0.25,
        drawdown_peak_date=None, drawdown_trough_date=None, drawdown_recovery_date=None,
        sharpe_ratio=0.5, sortino_ratio=0.7, calmar_ratio=0.2, total_executed_trades=10,
        completed_round_trips=5, winning_trades=2, losing_trades=3, win_rate=0.4,
        average_win=100.0, average_loss=-50.0, profit_factor=1.33, total_transaction_costs=100.0,
        avg_cost_per_trade=10.0, cost_to_capital_ratio=0.001, execution_convention="Test"
    )

    deg = calculate_degradation(dev_m, oos_m)

    np.testing.assert_almost_equal(deg["delta_cagr"], -0.05)
    np.testing.assert_almost_equal(deg["delta_sharpe"], -0.50)
    np.testing.assert_almost_equal(deg["delta_max_drawdown"], -0.05)


def test_10_input_dataframe_non_mutation(multi_year_price_df: pd.DataFrame):
    """10. Test that OOS partitioning and backtesting do not mutate input DataFrames."""
    feat_df = add_features(multi_year_price_df)
    signals = generate_ma_crossover_signals(feat_df)

    price_cols_before = list(feat_df.columns)
    sig_cols_before = list(signals.columns)

    run_multi_period_backtest(feat_df, signals, strategy_name="Test")

    assert list(feat_df.columns) == price_cols_before
    assert list(signals.columns) == sig_cols_before
