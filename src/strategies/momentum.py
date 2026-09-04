"""
Momentum Strategy Signal Module.

Generates position states and signal events based on N-period historical price momentum,
with zero look-ahead bias and configurable momentum thresholding.
"""

import logging

import numpy as np
import pandas as pd

from src.signal_engine import detect_signal_events, validate_required_columns

logger = logging.getLogger(__name__)


def generate_momentum_signals(
    df: pd.DataFrame, lookback: int = 20, threshold: float = 0.0
) -> pd.DataFrame:
    """Generate N-period momentum strategy signals.

    Strategy Rule
    -------------
    Momentum Formula : momentum_N(t) = (Close_t / Close_{t-N}) - 1
    Long Condition   : momentum > threshold  -> position_state = 1
    Flat Condition   : momentum <= threshold -> position_state = 0

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing 'close' price column.
    lookback : int, default 20
        Momentum lookback period N.
    threshold : float, default 0.0
        Minimum momentum threshold required for Long entry.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['momentum', 'threshold', 'position_state', 'signal_event'].

    Raises
    ------
    ValueError
        If required 'close' column is missing or lookback is invalid.
    """
    validate_required_columns(df, ["close"])
    if lookback <= 0:
        raise ValueError(f"lookback period ({lookback}) must be a positive integer.")

    input_df = df.copy()

    # Calculate N-period momentum: (Close_t / Close_{t-N}) - 1
    prev_close = input_df["close"].shift(lookback)
    momentum = (input_df["close"] / prev_close) - 1.0

    result = pd.DataFrame(index=input_df.index)
    result["momentum"] = momentum
    result["threshold"] = float(threshold)

    # Position state: 1 when momentum > threshold, 0 when <= threshold, NaN during warmup
    valid_mask = result["momentum"].notna()
    position_state = pd.Series(index=input_df.index, dtype="float64")
    position_state[valid_mask] = np.where(
        result.loc[valid_mask, "momentum"] > threshold, 1.0, 0.0
    )

    result["position_state"] = position_state
    result["signal_event"] = detect_signal_events(position_state)

    logger.info(
        f"Generated Momentum signals (Lookback={lookback}, Threshold={threshold}): "
        f"{int((result['signal_event'] == 1).sum())} entries, {int((result['signal_event'] == -1).sum())} exits."
    )
    return result
