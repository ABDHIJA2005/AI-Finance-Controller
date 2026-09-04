"""
Unit tests for Market Features and Feature Engineering module.
"""

import numpy as np
import pandas as pd
import pytest

from src.features import (
    add_features,
    calculate_atr,
    calculate_ema,
    calculate_returns,
    calculate_rolling_high_low,
    calculate_rolling_volatility,
    calculate_rsi,
    calculate_sma,
)


@pytest.fixture
def sample_ohlc_df() -> pd.DataFrame:
    """Fixture providing a synthetic 30-day OHLC DataFrame for feature testing."""
    dates = pd.date_range(start="2024-01-01", periods=30, freq="D")
    # Linear trend with some variation
    close_prices = np.array([100.0 + i + (i % 3) * 0.5 for i in range(30)])
    high_prices = close_prices + 2.0
    low_prices = close_prices - 2.0
    open_prices = close_prices - 0.5

    df = pd.DataFrame(
        {
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
            "volume": [1000] * 30,
        },
        index=dates,
    )
    df.index.name = "timestamp"
    return df


def test_1_daily_simple_returns(sample_ohlc_df: pd.DataFrame):
    """1. Test daily simple return calculation against manual calculation."""
    res = calculate_returns(sample_ohlc_df)
    assert "return" in res.columns
    assert np.isnan(res["return"].iloc[0])  # First row is NaN

    # Check second row manually: (Close_1 / Close_0) - 1
    expected_return_1 = (sample_ohlc_df["close"].iloc[1] / sample_ohlc_df["close"].iloc[0]) - 1.0
    np.testing.assert_almost_equal(res["return"].iloc[1], expected_return_1)


def test_2_log_returns(sample_ohlc_df: pd.DataFrame):
    """2. Test log return calculation against manual calculation."""
    res = calculate_returns(sample_ohlc_df)
    assert "log_return" in res.columns
    assert np.isnan(res["log_return"].iloc[0])

    expected_log_return_1 = np.log(sample_ohlc_df["close"].iloc[1] / sample_ohlc_df["close"].iloc[0])
    np.testing.assert_almost_equal(res["log_return"].iloc[1], expected_log_return_1)


def test_3_sma_calculation(sample_ohlc_df: pd.DataFrame):
    """3. Test SMA calculation against manually computed rolling average."""
    res = calculate_sma(sample_ohlc_df, windows=(5, 20))
    assert "sma_5" in res.columns
    assert "sma_20" in res.columns

    # First 4 rows of sma_5 should be NaN
    assert res["sma_5"].iloc[:4].isna().all()
    
    # 5th row should equal mean of first 5 close prices
    expected_sma_5 = sample_ohlc_df["close"].iloc[:5].mean()
    np.testing.assert_almost_equal(res["sma_5"].iloc[4], expected_sma_5)


def test_4_ema_calculation(sample_ohlc_df: pd.DataFrame):
    """4. Test EMA calculation for known data using ewm(span=N, adjust=False)."""
    res = calculate_ema(sample_ohlc_df, windows=(5,))
    assert "ema_5" in res.columns

    # Verify manual recurrence formula for EMA: EMA_t = alpha * Price_t + (1-alpha) * EMA_{t-1}
    alpha = 2.0 / (5.0 + 1.0)
    prices = sample_ohlc_df["close"].values
    expected_ema = np.zeros_like(prices)
    expected_ema[0] = prices[0]
    for i in range(1, len(prices)):
        expected_ema[i] = alpha * prices[i] + (1 - alpha) * expected_ema[i - 1]

    np.testing.assert_allclose(res["ema_5"].values, expected_ema)


def test_5_rolling_volatility(sample_ohlc_df: pd.DataFrame):
    """5. Test rolling annualized volatility calculation."""
    res = calculate_rolling_volatility(sample_ohlc_df, windows=(10,))
    assert "volatility_10" in res.columns

    # Manual check for row 10
    returns = (sample_ohlc_df["close"] / sample_ohlc_df["close"].shift(1) - 1.0).iloc[1:11]
    expected_vol = returns.std(ddof=1) * np.sqrt(252)
    np.testing.assert_almost_equal(res["volatility_10"].iloc[10], expected_vol)


def test_6_rsi_calculation(sample_ohlc_df: pd.DataFrame):
    """6. Test RSI calculation on controlled dataset (bounded [0, 100])."""
    res = calculate_rsi(sample_ohlc_df, period=14)
    assert "rsi_14" in res.columns

    rsi_vals = res["rsi_14"].dropna()
    assert (rsi_vals >= 0.0).all()
    assert (rsi_vals <= 100.0).all()


def test_7_true_range_calculation(sample_ohlc_df: pd.DataFrame):
    """7. Test True Range (TR) calculation."""
    res = calculate_atr(sample_ohlc_df, period=14)
    assert "true_range" in res.columns

    # Check row 1 TR manually
    high_1 = sample_ohlc_df["high"].iloc[1]
    low_1 = sample_ohlc_df["low"].iloc[1]
    prev_close_0 = sample_ohlc_df["close"].iloc[0]
    expected_tr_1 = max(high_1 - low_1, abs(high_1 - prev_close_0), abs(low_1 - prev_close_0))
    np.testing.assert_almost_equal(res["true_range"].iloc[1], expected_tr_1)


def test_8_atr_calculation(sample_ohlc_df: pd.DataFrame):
    """8. Test ATR calculation (must be non-negative)."""
    res = calculate_atr(sample_ohlc_df, period=14)
    assert "atr_14" in res.columns
    valid_atr = res["atr_14"].dropna()
    assert (valid_atr >= 0.0).all()


def test_9_rolling_high_low(sample_ohlc_df: pd.DataFrame):
    """9. Test rolling high/low calculation."""
    res = calculate_rolling_high_low(sample_ohlc_df, window=10)
    assert "rolling_high_10" in res.columns
    assert "rolling_low_10" in res.columns

    expected_high_9 = sample_ohlc_df["close"].iloc[:10].max()
    np.testing.assert_almost_equal(res["rolling_high_10"].iloc[9], expected_high_9)


def test_10_previous_window_high_low_exclusion(sample_ohlc_df: pd.DataFrame):
    """10. Test that previous_high_20 excludes the current observation t."""
    res = calculate_rolling_high_low(sample_ohlc_df, window=10)
    assert "previous_high_10" in res.columns

    # Row index 10: previous_high_10 should equal max of close from row index 0 to 9 (10 items excluding row 10)
    expected_prev_high = sample_ohlc_df["close"].iloc[0:10].max()
    np.testing.assert_almost_equal(res["previous_high_10"].iloc[10], expected_prev_high)


def test_11_preserves_original_ohlc(sample_ohlc_df: pd.DataFrame):
    """11. Test that add_features preserves original OHLC columns."""
    res = add_features(sample_ohlc_df)
    for col in ["open", "high", "low", "close", "volume"]:
        assert col in res.columns
        pd.testing.assert_series_equal(sample_ohlc_df[col], res[col])


def test_12_preserves_datetime_index(sample_ohlc_df: pd.DataFrame):
    """12. Test that add_features preserves original DatetimeIndex."""
    res = add_features(sample_ohlc_df)
    pd.testing.assert_index_equal(sample_ohlc_df.index, res.index)


def test_13_does_not_mutate_input_df(sample_ohlc_df: pd.DataFrame):
    """13. Test that add_features does not mutate the input DataFrame."""
    cols_before = list(sample_ohlc_df.columns)
    add_features(sample_ohlc_df)
    cols_after = list(sample_ohlc_df.columns)
    assert cols_before == cols_after


def test_14_missing_required_columns_raises_error():
    """14. Test that missing required OHLC columns raise ValueError."""
    bad_df = pd.DataFrame({"open": [100.0], "high": [105.0]})
    with pytest.raises(ValueError, match="missing required OHLC columns"):
        add_features(bad_df)


def test_look_ahead_bias_prevention(sample_ohlc_df: pd.DataFrame):
    """CRITICAL LOOK-AHEAD BIAS TEST.

    Demonstrates that modifying today's (t) Close price does NOT change
    `previous_high_20` or `previous_low_20` at index t.
    """
    original_features = add_features(sample_ohlc_df, breakout_window=10)

    # Pick a test index t = 15
    t_idx = 15
    target_date = sample_ohlc_df.index[t_idx]

    prev_high_before = original_features.loc[target_date, "previous_high_10"]
    prev_low_before = original_features.loc[target_date, "previous_low_10"]

    # Mutate Close and High at index t to extreme outlier values
    modified_df = sample_ohlc_df.copy()
    modified_df.loc[target_date, "close"] = 999999.0
    modified_df.loc[target_date, "high"] = 999999.0

    modified_features = add_features(modified_df, breakout_window=10)

    prev_high_after = modified_features.loc[target_date, "previous_high_10"]
    prev_low_after = modified_features.loc[target_date, "previous_low_10"]

    # Assert that previous_high and previous_low at index t remain EXACTLY identical
    assert prev_high_before == prev_high_after, "LOOK-AHEAD LEAK: previous_high_10 changed when today's price changed!"
    assert prev_low_before == prev_low_after, "LOOK-AHEAD LEAK: previous_low_10 changed when today's price changed!"
