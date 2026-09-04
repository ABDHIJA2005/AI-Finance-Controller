"""
Unit tests for Trading Strategies and Signal Generation logic.
"""

import numpy as np
import pandas as pd
import pytest

from src.features import add_features
from src.strategies.breakout import generate_breakout_signals
from src.strategies.momentum import generate_momentum_signals
from src.strategies.moving_average import generate_ma_crossover_signals


@pytest.fixture
def synthetic_price_df() -> pd.DataFrame:
    """Fixture providing a synthetic 60-day price DataFrame for strategy tests."""
    dates = pd.date_range(start="2024-01-01", periods=60, freq="D")
    # Sine wave price oscillation around 100
    x = np.linspace(0, 4 * np.pi, 60)
    prices = 100.0 + 10.0 * np.sin(x)

    df = pd.DataFrame(
        {
            "open": prices - 0.5,
            "high": prices + 1.0,
            "low": prices - 1.0,
            "close": prices,
            "volume": [1000] * 60,
        },
        index=dates,
    )
    df.index.name = "timestamp"
    return df


# --- Moving Average Crossover Tests ---

def test_ma_crossover_buy_and_exit_conditions(synthetic_price_df: pd.DataFrame):
    """Test MA crossover BUY (fast > slow) and EXIT (fast < slow) conditions."""
    feat_df = add_features(synthetic_price_df, sma_windows=(5, 10))
    signals = generate_ma_crossover_signals(feat_df, fast_window=5, slow_window=10)

    assert "position_state" in signals.columns
    assert "signal_event" in signals.columns

    # Verify that whenever fast_ma > slow_ma, position_state is 1
    valid_mask = signals["fast_ma"].notna() & signals["slow_ma"].notna()
    long_mask = valid_mask & (signals["fast_ma"] > signals["slow_ma"])
    flat_mask = valid_mask & (signals["fast_ma"] <= signals["slow_ma"])

    assert (signals.loc[long_mask, "position_state"] == 1.0).all()
    assert (signals.loc[flat_mask, "position_state"] == 0.0).all()


def test_ma_crossover_unavailable_history(synthetic_price_df: pd.DataFrame):
    """Test that MA crossover signals are NaN when SMA history is unavailable."""
    signals = generate_ma_crossover_signals(synthetic_price_df, fast_window=10, slow_window=20)
    # First 19 rows should have NaN position_state
    assert signals["position_state"].iloc[:19].isna().all()


def test_ma_crossover_non_mutation(synthetic_price_df: pd.DataFrame):
    """Test that MA crossover does not mutate the input DataFrame."""
    cols_before = list(synthetic_price_df.columns)
    generate_ma_crossover_signals(synthetic_price_df, fast_window=5, slow_window=10)
    assert list(synthetic_price_df.columns) == cols_before


# --- Momentum Strategy Tests ---

def test_momentum_formula_and_threshold(synthetic_price_df: pd.DataFrame):
    """Test momentum formula (Close_t / Close_{t-N} - 1) and threshold comparison."""
    lookback = 10
    threshold = 0.02
    signals = generate_momentum_signals(synthetic_price_df, lookback=lookback, threshold=threshold)

    # Check momentum formula at index 15
    close_15 = synthetic_price_df["close"].iloc[15]
    close_5 = synthetic_price_df["close"].iloc[5]
    expected_mom_15 = (close_15 / close_5) - 1.0

    np.testing.assert_almost_equal(signals["momentum"].iloc[15], expected_mom_15)

    # Check state threshold comparison
    valid_mask = signals["momentum"].notna()
    entry_mask = valid_mask & (signals["momentum"] > threshold)
    exit_mask = valid_mask & (signals["momentum"] <= threshold)

    assert (signals.loc[entry_mask, "position_state"] == 1.0).all()
    assert (signals.loc[exit_mask, "position_state"] == 0.0).all()


def test_momentum_insufficient_history(synthetic_price_df: pd.DataFrame):
    """Test that momentum signals are NaN during lookback warmup."""
    signals = generate_momentum_signals(synthetic_price_df, lookback=15)
    assert signals["position_state"].iloc[:15].isna().all()


# --- N-Day Breakout Strategy Tests ---

def test_breakout_entry_and_exit_conditions(synthetic_price_df: pd.DataFrame):
    """Test Breakout entry (Close > prev_high) and exit (Close < prev_low)."""
    feat_df = add_features(synthetic_price_df, breakout_window=10)
    signals = generate_breakout_signals(feat_df, lookback=10)

    assert "previous_high" in signals.columns
    assert "previous_low" in signals.columns

    # Inspect entry event rows
    entry_events = signals[signals["signal_event"] == 1.0]
    for idx in entry_events.index:
        close_val = signals.loc[idx, "close"]
        prev_high = signals.loc[idx, "previous_high"]
        assert close_val > prev_high, f"Breakout Entry triggered without Close ({close_val}) > Prev High ({prev_high})"

    # Inspect exit event rows
    exit_events = signals[signals["signal_event"] == -1.0]
    for idx in exit_events.index:
        close_val = signals.loc[idx, "close"]
        prev_low = signals.loc[idx, "previous_low"]
        assert close_val < prev_low, f"Breakout Exit triggered without Close ({close_val}) < Prev Low ({prev_low})"


def test_breakout_look_ahead_bias_prevention(synthetic_price_df: pd.DataFrame):
    """CRITICAL BREAKOUT LOOK-AHEAD TEST.

    Constructs a controlled DataFrame, mutates today's Close/High/Low, and verifies
    that today's previous-window threshold (previous_high_N) used for today's signal
    REMAINS EXACTLY UNCHANGED.
    """
    lookback = 10
    feat_df = add_features(synthetic_price_df, breakout_window=lookback)
    original_signals = generate_breakout_signals(feat_df, lookback=lookback)

    # Pick a target test index t = 25
    t_idx = 25
    target_date = synthetic_price_df.index[t_idx]

    prev_high_before = original_signals.loc[target_date, "previous_high"]
    prev_low_before = original_signals.loc[target_date, "previous_low"]

    # Mutate today's OHLC values at index t to extreme outlier values
    mutated_price_df = synthetic_price_df.copy()
    mutated_price_df.loc[target_date, "open"] = 888888.0
    mutated_price_df.loc[target_date, "high"] = 999999.0
    mutated_price_df.loc[target_date, "low"] = 111111.0
    mutated_price_df.loc[target_date, "close"] = 999999.0

    mutated_feat_df = add_features(mutated_price_df, breakout_window=lookback)
    mutated_signals = generate_breakout_signals(mutated_feat_df, lookback=lookback)

    prev_high_after = mutated_signals.loc[target_date, "previous_high"]
    prev_low_after = mutated_signals.loc[target_date, "previous_low"]

    # Assert that today's previous_high and previous_low thresholds used for today's signal DID NOT CHANGE
    assert prev_high_before == prev_high_after, "LOOK-AHEAD LEAK: previous_high changed when today's OHLC mutated!"
    assert prev_low_before == prev_low_after, "LOOK-AHEAD LEAK: previous_low changed when today's OHLC mutated!"
