"""
N-Day Breakout Strategy Signal Module.

Generates position states and signal events based on N-day price breakout thresholds,
strictly utilizing shifted previous-window thresholds (previous_high_N and previous_low_N)
to eliminate look-ahead bias.
"""

import logging

import numpy as np
import pandas as pd

from src.features import calculate_rolling_high_low
from src.signal_engine import detect_signal_events, validate_required_columns

logger = logging.getLogger(__name__)


def generate_breakout_signals(
    df: pd.DataFrame, lookback: int = 20
) -> pd.DataFrame:
    """Generate N-day breakout strategy signals with zero look-ahead bias.

    Strategy Rule (Stateful Breakout)
    ----------------------------------
    - Entry Condition : Close_t > previous_high_N_t -> position_state = 1
    - Exit Condition  : Close_t < previous_low_N_t  -> position_state = 0
    - Continuation    : previous_low_N_t <= Close_t <= previous_high_N_t
                        -> position_state_t = position_state_{t-1}

    CRITICAL LOOK-AHEAD REQUIREMENT:
    ---------------------------------
    `previous_high_N_t` represents max(Close_{t-N} ... Close_{t-1}) and
    `previous_low_N_t` represents min(Close_{t-N} ... Close_{t-1}), explicitly
    shifted by 1 period to EXCLUDE today's observation Close_t.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing 'close' and optionally 'previous_high_N' columns.
    lookback : int, default 20
        Breakout lookback window size N.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['previous_high', 'previous_low', 'close', 'position_state', 'signal_event'].

    Raises
    ------
    ValueError
        If required 'close' column is missing or lookback is invalid.
    """
    validate_required_columns(df, ["close"])
    if lookback <= 0:
        raise ValueError(f"lookback window ({lookback}) must be a positive integer.")

    input_df = df.copy()

    prev_high_col = f"previous_high_{lookback}"
    prev_low_col = f"previous_low_{lookback}"

    # Calculate shifted rolling high/low features if not already present
    if prev_high_col not in input_df.columns or prev_low_col not in input_df.columns:
        input_df = calculate_rolling_high_low(input_df, window=lookback)

    result = pd.DataFrame(index=input_df.index)
    result["close"] = input_df["close"]
    result["previous_high"] = input_df[prev_high_col]
    result["previous_low"] = input_df[prev_low_col]

    # Stateful position propagation
    closes = result["close"].values
    prev_highs = result["previous_high"].values
    prev_lows = result["previous_low"].values

    n_rows = len(result)
    position_state = np.full(n_rows, np.nan)

    current_state = 0.0  # Initial state is flat

    for t in range(n_rows):
        if np.isnan(prev_highs[t]) or np.isnan(prev_lows[t]):
            continue  # Keep NaN during lookback warmup

        if closes[t] > prev_highs[t]:
            current_state = 1.0  # Breakout Above -> Entry Long
        elif closes[t] < prev_lows[t]:
            current_state = 0.0  # Breakdown Below -> Exit Flat
        # Else: maintain current_state unchanged

        position_state[t] = current_state

    result["position_state"] = pd.Series(position_state, index=input_df.index)
    result["signal_event"] = detect_signal_events(result["position_state"])

    logger.info(
        f"Generated Breakout signals (Lookback={lookback}): "
        f"{int((result['signal_event'] == 1).sum())} entries, {int((result['signal_event'] == -1).sum())} exits."
    )
    return result
