"""
Signal Engine Module.

Provides core utility functions and standardized validation for generating position states
and signal events across trading strategies without trade execution or portfolio accounting.
"""

import logging
from typing import Sequence, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ALLOWED_SIGNAL_VALUES = {1, 0, -1}


def validate_required_columns(
    df: pd.DataFrame, required_cols: Sequence[str]
) -> None:
    """Validate that required feature columns exist in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    required_cols : Sequence[str]
        List of required column names.

    Raises
    ------
    ValueError
        If any required column is missing.
    """
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"Input DataFrame is missing required feature columns: {missing}"
        )


def validate_signal_values(signals: pd.Series) -> bool:
    """Validate that a signal Series contains only allowed values {1, 0, -1} or NaN.

    Parameters
    ----------
    signals : pd.Series
        Signal values series.

    Returns
    -------
    bool
        True if all non-NaN values are in {1, 0, -1}, False otherwise.
    """
    non_null_signals = signals.dropna()
    unique_vals = set(non_null_signals.unique())
    is_valid = unique_vals.issubset(ALLOWED_SIGNAL_VALUES)
    if not is_valid:
        logger.warning(
            f"Detected invalid signal values: {unique_vals - ALLOWED_SIGNAL_VALUES}"
        )
    return is_valid


def detect_signal_events(position_state: pd.Series) -> pd.Series:
    """Convert continuous position states (1=long, 0=flat) into signal events.

    Signal Event Conventions
    ------------------------
     1 : New Entry (transition from flat (0) to long (1))
    -1 : Exit (transition from long (1) to flat (0))
     0 : Continuation / No position state transition

    Initial State Handling Policy
    -----------------------------
    For the first valid (non-NaN) observation in `position_state`:
    - If initial state is 1 (long), it is flagged as a new Entry (1).
    - If initial state is 0 (flat), it is flagged as 0 (no event).

    Parameters
    ----------
    position_state : pd.Series
        Series of position states (1 for long, 0 for flat, NaN during warmup).

    Returns
    -------
    pd.Series
        Series of signal events (1, -1, 0, or NaN).
    """
    events = pd.Series(index=position_state.index, dtype="float64")
    
    # Identify non-NaN position state mask
    valid_mask = position_state.notna()
    if not valid_mask.any():
        return events  # Return all NaN if no valid states

    valid_states = position_state[valid_mask]
    prev_states = valid_states.shift(1)

    # Compute raw state diff: position_t - position_{t-1}
    # 1 - 0 = +1 (Entry), 0 - 1 = -1 (Exit), 1 - 1 = 0, 0 - 0 = 0
    diff = valid_states - prev_states

    # Handle the initial state explicitly (first non-NaN row)
    first_valid_idx = valid_states.index[0]
    if valid_states.loc[first_valid_idx] == 1:
        diff.loc[first_valid_idx] = 1.0
    else:
        diff.loc[first_valid_idx] = 0.0

    events.loc[valid_mask] = diff
    return events
