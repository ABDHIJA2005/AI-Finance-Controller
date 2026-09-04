"""
Market Features and Feature Engineering Module.

Provides point-in-time calculation of technical market indicators and financial metrics
with zero look-ahead bias, maintaining chronological time-series integrity.
"""

import logging
from typing import Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def validate_ohlc_columns(df: pd.DataFrame) -> None:
    """Validate that required OHLC columns exist in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Raises
    ------
    ValueError
        If required OHLC columns are missing.
    """
    required = ["open", "high", "low", "close"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Input DataFrame is missing required OHLC columns: {missing}"
        )


def calculate_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily simple returns and log returns.

    Mathematical Definitions
    ------------------------
    r_t = (Close_t / Close_{t-1}) - 1
    log_r_t = ln(Close_t / Close_{t-1})

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'close' column.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'return' and 'log_return' columns added.
    """
    result = df.copy()
    prev_close = result["close"].shift(1)
    
    # Simple returns: (Close_t / Close_{t-1}) - 1
    result["return"] = (result["close"] / prev_close) - 1.0
    
    # Log returns: ln(Close_t / Close_{t-1})
    result["log_return"] = np.log(result["close"] / prev_close)
    
    return result


def calculate_sma(
    df: pd.DataFrame, windows: Sequence[int] = (20, 50, 200)
) -> pd.DataFrame:
    """Calculate Simple Moving Averages (SMA) over configurable rolling windows.

    Mathematical Definition
    ------------------------
    SMA_{N, t} = (1 / N) * sum_{i=0}^{N-1} Close_{t-i}

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'close' column.
    windows : Sequence[int], default (20, 50, 200)
        Rolling window sizes.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'sma_{N}' columns added.
    """
    result = df.copy()
    for w in windows:
        col_name = f"sma_{w}"
        result[col_name] = result["close"].rolling(window=w).mean()
    return result


def calculate_ema(
    df: pd.DataFrame, windows: Sequence[int] = (20, 50)
) -> pd.DataFrame:
    """Calculate Exponential Moving Averages (EMA) over configurable spans.

    Pandas EMA Convention
    ---------------------
    Uses `ewm(span=N, adjust=False)`.
    Smoothing multiplier alpha = 2 / (N + 1).
    EMA_t = alpha * Close_t + (1 - alpha) * EMA_{t-1}

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'close' column.
    windows : Sequence[int], default (20, 50)
        EMA span window sizes.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'ema_{N}' columns added.
    """
    result = df.copy()
    for w in windows:
        col_name = f"ema_{w}"
        result[col_name] = result["close"].ewm(span=w, adjust=False).mean()
    return result


def calculate_rolling_volatility(
    df: pd.DataFrame,
    windows: Sequence[int] = (20, 50),
    annualization_factor: float = np.sqrt(252),
) -> pd.DataFrame:
    """Calculate annualized rolling volatility from daily simple returns.

    Mathematical Definition
    ------------------------
    volatility_{N, t} = std(daily_returns_{t-N+1:t}) * sqrt(252)

    Annualization Assumption
    ------------------------
    The factor sqrt(252) assumes 252 trading days per calendar year.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'return' column (or computes it if missing).
    windows : Sequence[int], default (20, 50)
        Rolling window sizes.
    annualization_factor : float, default sqrt(252)
        Annualization multiplier.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'volatility_{N}' columns added.
    """
    result = df.copy()
    if "return" not in result.columns:
        result = calculate_returns(result)

    for w in windows:
        col_name = f"volatility_{w}"
        # Rolling sample standard deviation (ddof=1)
        rolling_std = result["return"].rolling(window=w).std()
        result[col_name] = rolling_std * annualization_factor
    return result


def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Calculate Relative Strength Index (RSI) using Wilder's smoothing.

    Mathematical Methodology
    ------------------------
    1. Delta_t = Close_t - Close_{t-1}
    2. Gain_t = max(Delta_t, 0), Loss_t = max(-Delta_t, 0)
    3. Smoothed AvgGain and AvgLoss using Exponential Moving Average with alpha = 1 / period (Wilder's RMA).
    4. RS_t = AvgGain_t / AvgLoss_t
    5. RSI_t = 100 - (100 / (1 + RS_t))

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'close' column.
    period : int, default 14
        RSI lookback period.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'rsi_{period}' column added (values bounded in [0, 100]).
    """
    result = df.copy()
    delta = result["close"].diff()

    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Wilder's Smoothing (RMA) is equivalent to EMA with alpha = 1 / period
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()

    # Calculate RS handling zero division
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # If avg_loss is zero (all gains), RSI = 100; if avg_gain is zero (all losses), RSI = 0
    rsi = rsi.fillna(100.0)
    
    # Set first period (before enough data) to NaN to preserve initial rolling period behavior
    rsi.iloc[:period] = np.nan

    result[f"rsi_{period}"] = rsi
    return result


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Calculate True Range (TR) and Average True Range (ATR).

    Mathematical Methodology
    ------------------------
    TR_t = max(High_t - Low_t, |High_t - Close_{t-1}|, |Low_t - Close_{t-1}|)
    ATR_t = Wilder's Smoothing of TR over N periods (alpha = 1 / period).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'high', 'low', and 'close' columns.
    period : int, default 14
        ATR lookback period.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'true_range' and 'atr_{period}' columns added.
    """
    result = df.copy()
    prev_close = result["close"].shift(1)

    tr1 = result["high"] - result["low"]
    tr2 = (result["high"] - prev_close).abs()
    tr3 = (result["low"] - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    result["true_range"] = tr

    # Wilder's smoothing for ATR
    atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()
    
    # Set initial period (before period observations) to NaN
    atr.iloc[:period] = np.nan

    result[f"atr_{period}"] = atr
    return result


def calculate_rolling_high_low(
    df: pd.DataFrame, window: int = 20
) -> pd.DataFrame:
    """Calculate rolling highs/lows and shifted previous-window highs/lows.

    Critical Look-Ahead Prevention Design
    --------------------------------------
    - `rolling_high_20`: Rolling max of Close including current observation t.
    - `rolling_low_20`: Rolling min of Close including current observation t.
    - `previous_high_20`: Rolling max of Close over window N shifted by 1,
      representing max(Close_{t-N} ... Close_{t-1}) strictly EXCLUDING current Close_t.
    - `previous_low_20`: Rolling min of Close over window N shifted by 1,
      representing min(Close_{t-N} ... Close_{t-1}) strictly EXCLUDING current Close_t.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'close' column.
    window : int, default 20
        Rolling window size.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'rolling_high_{window}', 'rolling_low_{window}',
        'previous_high_{window}', and 'previous_low_{window}' columns added.
    """
    result = df.copy()

    # Rolling window including today
    result[f"rolling_high_{window}"] = result["close"].rolling(window=window).max()
    result[f"rolling_low_{window}"] = result["close"].rolling(window=window).min()

    # Shifted previous window strictly excluding today
    result[f"previous_high_{window}"] = (
        result["close"].rolling(window=window).max().shift(1)
    )
    result[f"previous_low_{window}"] = (
        result["close"].rolling(window=window).min().shift(1)
    )

    return result


def add_features(
    df: pd.DataFrame,
    sma_windows: Sequence[int] = (20, 50, 200),
    ema_windows: Sequence[int] = (20, 50),
    volatility_windows: Sequence[int] = (20, 50),
    rsi_period: int = 14,
    atr_period: int = 14,
    breakout_window: int = 20,
) -> pd.DataFrame:
    """Main pipeline function to compute all market features.

    Validates schema, preserves input DataFrame, DatetimeIndex, and OHLC columns,
    and returns a new DataFrame containing all computed market features.

    Parameters
    ----------
    df : pd.DataFrame
        Input market DataFrame containing OHLC price columns.
    sma_windows : Sequence[int], default (20, 50, 200)
        SMA window sizes.
    ema_windows : Sequence[int], default (20, 50)
        EMA span sizes.
    volatility_windows : Sequence[int], default (20, 50)
        Rolling volatility window sizes.
    rsi_period : int, default 14
        RSI period.
    atr_period : int, default 14
        ATR period.
    breakout_window : int, default 20
        Breakout rolling high/low window size.

    Returns
    -------
    pd.DataFrame
        New DataFrame with all features added.

    Raises
    ------
    ValueError
        If required OHLC columns are missing.
    """
    validate_ohlc_columns(df)

    # Work on a copy to prevent mutation
    feat_df = df.copy()

    # 1. Returns
    feat_df = calculate_returns(feat_df)

    # 2. Simple Moving Averages
    feat_df = calculate_sma(feat_df, windows=sma_windows)

    # 3. Exponential Moving Averages
    feat_df = calculate_ema(feat_df, windows=ema_windows)

    # 4. Rolling Volatility
    feat_df = calculate_rolling_volatility(feat_df, windows=volatility_windows)

    # 5. Relative Strength Index
    feat_df = calculate_rsi(feat_df, period=rsi_period)

    # 6. Average True Range
    feat_df = calculate_atr(feat_df, period=atr_period)

    # 7. Rolling High / Low & Shifted Previous Window High / Low
    feat_df = calculate_rolling_high_low(feat_df, window=breakout_window)

    logger.info(
        f"Successfully generated {len(feat_df.columns) - len(df.columns)} feature columns for {len(feat_df)} rows."
    )
    return feat_df
