"""
Data Cleaner and Validator Module.

Provides robust, non-destructive data validation and clean-room filtering for
historical market OHLCV time-series data, enforcing strict quantitative rigor
and zero look-ahead bias.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Dataclass holding validation check results for an OHLCV dataset.

    Attributes
    ----------
    is_valid : bool
        True if no critical validation errors were detected, False otherwise.
    issues : List[str]
        List of human-readable description strings for detected issues/warnings.
    missing_counts : Dict[str, int]
        Dictionary mapping column names to count of missing (NaN) values.
    duplicate_timestamps_count : int
        Number of duplicate index timestamps detected.
    invalid_rows_count : int
        Total number of rows violating price bounds or logical OHLC relationships.
    is_chronological : bool
        True if the DatetimeIndex is strictly monotonically increasing.
    """

    is_valid: bool = True
    issues: List[str] = field(default_factory=list)
    missing_counts: Dict[str, int] = field(default_factory=dict)
    duplicate_timestamps_count: int = 0
    invalid_rows_count: int = 0
    is_chronological: bool = True


class DataCleaner:
    """Validator and cleaner for market OHLCV DataFrames.

    Validates schema, price bounds, logical OHLC relationships, timestamp monotonicity,
    and missing values according to strict quantitative financial standards.
    """

    REQUIRED_PRICE_COLUMNS = ["open", "high", "low", "close"]

    def __init__(self, log_level: int = logging.INFO) -> None:
        """Initialize DataCleaner.

        Parameters
        ----------
        log_level : int, default logging.INFO
            Logging verbosity level.
        """
        logger.setLevel(log_level)

    def validate(
        self, df: pd.DataFrame, is_index: bool = False
    ) -> ValidationResult:
        """Validate an OHLCV DataFrame against quantitative hygiene standards.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with DatetimeIndex and normalized lowercase OHLCV columns.
        is_index : bool, default False
            If True, treats volume as optional (e.g. for NIFTY 50 index data).

        Returns
        -------
        ValidationResult
            Detailed validation report.
        """
        result = ValidationResult()

        # 1. Check for empty DataFrame
        if df.empty:
            result.is_valid = False
            result.issues.append("DataFrame is empty.")
            logger.error("Validation failed: DataFrame is empty.")
            return result

        # 2. Check for required price columns
        missing_cols = [
            col for col in self.REQUIRED_PRICE_COLUMNS if col not in df.columns
        ]
        if missing_cols:
            result.is_valid = False
            result.issues.append(
                f"Missing required price columns: {missing_cols}"
            )
            logger.error(
                f"Validation failed: missing required columns {missing_cols}"
            )
            return result

        # 3. Check for missing values (NaN)
        for col in df.columns:
            nan_count = int(df[col].isna().sum())
            result.missing_counts[col] = nan_count
            if nan_count > 0:
                if col in self.REQUIRED_PRICE_COLUMNS:
                    result.is_valid = False
                    result.issues.append(
                        f"Column '{col}' has {nan_count} missing (NaN) values."
                    )
                else:
                    result.issues.append(
                        f"Non-price column '{col}' has {nan_count} missing values."
                    )

        # 4. Check for duplicate timestamps in index
        result.duplicate_timestamps_count = int(df.index.duplicated().sum())
        if result.duplicate_timestamps_count > 0:
            result.is_valid = False
            result.issues.append(
                f"Detected {result.duplicate_timestamps_count} duplicate timestamp(s) in index."
            )
            logger.warning(
                f"Duplicate timestamps found: {result.duplicate_timestamps_count}"
            )

        # 5. Check chronological ordering (monotonic increasing)
        if not df.index.is_monotonic_increasing:
            result.is_chronological = False
            result.is_valid = False
            result.issues.append(
                "DatetimeIndex is not chronologically sorted (monotonic increasing)."
            )

        # 6. Validate price non-positivity (Price <= 0)
        invalid_price_mask = (
            (df["open"] <= 0)
            | (df["high"] <= 0)
            | (df["low"] <= 0)
            | (df["close"] <= 0)
        )

        # 7. Validate OHLC logical relationships
        # High >= max(Open, Close), Low <= min(Open, Close), High >= Low
        invalid_high_mask = (df["high"] < df["open"]) | (
            df["high"] < df["close"]
        )
        invalid_low_mask = (df["low"] > df["open"]) | (df["low"] > df["close"])
        invalid_high_low_mask = df["high"] < df["low"]

        ohlc_invalid_mask = (
            invalid_price_mask
            | invalid_high_mask
            | invalid_low_mask
            | invalid_high_low_mask
        )

        # 8. Check volume if present
        if "volume" in df.columns:
            negative_volume_mask = df["volume"] < 0
            ohlc_invalid_mask = ohlc_invalid_mask | negative_volume_mask
            if negative_volume_mask.any():
                result.issues.append(
                    f"Detected {int(negative_volume_mask.sum())} row(s) with negative volume."
                )

            # Volume optional note for index instruments
            if is_index:
                zero_or_nan_vol = (df["volume"].isna()) | (df["volume"] == 0)
                if zero_or_nan_vol.all():
                    logger.info(
                        "Volume is zero or unavailable for this index dataset, which is expected for index instruments."
                    )

        result.invalid_rows_count = int(ohlc_invalid_mask.sum())
        if result.invalid_rows_count > 0:
            result.is_valid = False
            result.issues.append(
                f"Detected {result.invalid_rows_count} row(s) violating OHLC logical bounds or non-positive price limits."
            )

        return result

    def clean(
        self, df: pd.DataFrame, is_index: bool = False
    ) -> Tuple[pd.DataFrame, ValidationResult]:
        """Clean an OHLCV DataFrame by applying deterministic filtering rules.

        Rules applied:
        - Sorts index chronologically.
        - Removes duplicate timestamps (retains first occurrence).
        - Removes rows with missing required price values.
        - Removes rows violating non-positive price limits or OHLC bounds.
        - Never fabricates data or applies arbitrary forward-filling.

        Parameters
        ----------
        df : pd.DataFrame
            Raw OHLCV DataFrame.
        is_index : bool, default False
            If True, treats volume as optional.

        Returns
        -------
        Tuple[pd.DataFrame, ValidationResult]
            Cleaned DataFrame and final ValidationResult of the cleaned DataFrame.
        """
        if df.empty:
            logger.warning("Attempted to clean empty DataFrame.")
            return df.copy(), self.validate(df, is_index=is_index)

        cleaned_df = df.copy()

        # 1. Sort chronologically
        if not cleaned_df.index.is_monotonic_increasing:
            logger.info("Sorting DataFrame chronologically by DatetimeIndex.")
            cleaned_df = cleaned_df.sort_index(ascending=True)

        # 2. Handle duplicate timestamps (keep first, log warning)
        num_duplicates = int(cleaned_df.index.duplicated().sum())
        if num_duplicates > 0:
            logger.warning(
                f"Dropping {num_duplicates} duplicate timestamp(s), keeping first occurrence."
            )
            cleaned_df = cleaned_df[~cleaned_df.index.duplicated(keep="first")]

        # 3. Drop rows with missing required price values
        missing_price_mask = cleaned_df[self.REQUIRED_PRICE_COLUMNS].isna().any(axis=1)
        if missing_price_mask.any():
            dropped_dates = cleaned_df.index[missing_price_mask].tolist()
            logger.warning(
                f"Dropping {len(dropped_dates)} row(s) due to missing price values: {dropped_dates}"
            )
            cleaned_df = cleaned_df[~missing_price_mask]

        # 4. Filter invalid OHLC values & price bounds
        valid_price_mask = (
            (cleaned_df["open"] > 0)
            & (cleaned_df["high"] > 0)
            & (cleaned_df["low"] > 0)
            & (cleaned_df["close"] > 0)
        )
        valid_high_mask = (cleaned_df["high"] >= cleaned_df["open"]) & (
            cleaned_df["high"] >= cleaned_df["close"]
        )
        valid_low_mask = (cleaned_df["low"] <= cleaned_df["open"]) & (
            cleaned_df["low"] <= cleaned_df["close"]
        )
        valid_high_low_mask = cleaned_df["high"] >= cleaned_df["low"]

        valid_rows = (
            valid_price_mask
            & valid_high_mask
            & valid_low_mask
            & valid_high_low_mask
        )

        if "volume" in cleaned_df.columns:
            valid_volume_mask = (cleaned_df["volume"].isna()) | (
                cleaned_df["volume"] >= 0
            )
            valid_rows = valid_rows & valid_volume_mask

        invalid_rows_mask = ~valid_rows
        if invalid_rows_mask.any():
            invalid_dates = cleaned_df.index[invalid_rows_mask].tolist()
            logger.warning(
                f"Dropping {len(invalid_dates)} row(s) violating price bounds or OHLC relationships: {invalid_dates}"
            )
            cleaned_df = cleaned_df[valid_rows]

        # Final validation check on cleaned data
        final_validation = self.validate(cleaned_df, is_index=is_index)
        return cleaned_df, final_validation
