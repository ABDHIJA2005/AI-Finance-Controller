"""
Moving-Average Crossover Strategy Signal Module.

Generates position states and signal events based on fast vs slow Simple Moving Average (SMA)
crossovers, with zero look-ahead bias and no trade execution modeling.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.features import calculate_sma
from src.signal_engine import detect_signal_events, validate_required_columns

logger = logging.getLogger(__name__)


def generate_ma_crossover_signals(
    df: pd.DataFrame, fast_window: int = 20, slow_window: int = 50
) -> pd.DataFrame:
    """Generate moving-average crossover strategy signals.

    Strategy Rule
    -------------
    Long Condition  : SMA_fast > SMA_slow  -> position_state = 1
    Flat Condition  : SMA_fast <= SMA_slow -> position_state = 0

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing OHLC or SMA feature columns.
    fast_window : int, default 20
        Fast moving average window size.
    slow_window : int, default 50
        Slow moving average window size.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['fast_ma', 'slow_ma', 'position_state', 'signal_event'].

    Raises
    ------
    ValueError
        If required price/feature columns are missing or window parameters are invalid.
    """
    if fast_window >= slow_window:
        raise ValueError(
            f"fast_window ({fast_window}) must be strictly smaller than slow_window ({slow_window})."
        )

    # Work on a copy to prevent input DataFrame mutation
    input_df = df.copy()

    fast_col = f"sma_{fast_window}"
    slow_col = f"sma_{slow_window}"

    # Calculate SMAs if not present in input DataFrame
    if fast_col not in input_df.columns or slow_col not in input_df.columns:
        input_df = calculate_sma(input_df, windows=(fast_window, slow_window))

    result = pd.DataFrame(index=input_df.index)
    result["fast_ma"] = input_df[fast_col]
    result["slow_ma"] = input_df[slow_col]

    # Create position state series (1 = long, 0 = flat, NaN when SMAs unavailable)
    valid_mask = result["fast_ma"].notna() & result["slow_ma"].notna()
    
    position_state = pd.Series(index=input_df.index, dtype="float64")
    position_state[valid_mask] = np.where(
        result.loc[valid_mask, "fast_ma"] > result.loc[valid_mask, "slow_ma"],
        1.0,
        0.0,
    )

    result["position_state"] = position_state
    result["signal_event"] = detect_signal_events(position_state)

    logger.info(
        f"Generated MA Crossover signals (Fast={fast_window}, Slow={slow_window}): "
        f"{int((result['signal_event'] == 1).sum())} entries, {int((result['signal_event'] == -1).sum())} exits."
    )
    return result
