"""
Unit tests for Signal Engine utilities and signal event processing.
"""

import numpy as np
import pandas as pd
import pytest

from src.signal_engine import (
    detect_signal_events,
    validate_required_columns,
    validate_signal_values,
)


def test_validate_signal_values():
    """Test validation of allowed signal values {1, 0, -1}."""
    valid_signals = pd.Series([1.0, 0.0, -1.0, np.nan, 1.0])
    assert validate_signal_values(valid_signals) is True

    invalid_signals = pd.Series([1.0, 2.0, 0.0])
    assert validate_signal_values(invalid_signals) is False


def test_detect_signal_events_entry_and_exit():
    """Test detection of new entry (+1) and exit (-1) transition events."""
    # State sequence: [0, 0, 1, 1, 1, 0, 0]
    states = pd.Series([0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0])
    events = detect_signal_events(states)

    # Expected events: [0, 0, 1, 0, 0, -1, 0]
    expected = pd.Series([0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0])
    pd.testing.assert_series_equal(events, expected)


def test_detect_signal_events_initial_state_handling():
    """Test explicit handling of initial state when state starts as long (1)."""
    # State sequence: [1, 1, 0]
    states = pd.Series([1.0, 1.0, 0.0])
    events = detect_signal_events(states)

    # Initial state is 1 -> flagged as initial Entry (1)
    expected = pd.Series([1.0, 0.0, -1.0])
    pd.testing.assert_series_equal(events, expected)


def test_validate_required_columns_raises():
    """Test that missing required columns raise ValueError."""
    df = pd.DataFrame({"open": [100.0], "close": [105.0]})
    with pytest.raises(ValueError, match="missing required feature columns"):
        validate_required_columns(df, ["open", "high", "low", "close"])
