"""
Unit tests for DataCleaner and Data Validation logic.
"""

import numpy as np
import pandas as pd
import pytest

from src.data_cleaner import DataCleaner, ValidationResult


@pytest.fixture
def cleaner() -> DataCleaner:
    """Fixture providing a DataCleaner instance."""
    return DataCleaner()


@pytest.fixture
def valid_ohlcv_df() -> pd.DataFrame:
    """Fixture generating a valid synthetic OHLCV DataFrame."""
    dates = pd.date_range(start="2024-01-01", periods=5, freq="D")
    df = pd.DataFrame(
        {
            "open": [100.0, 102.0, 101.0, 103.0, 105.0],
            "high": [105.0, 106.0, 104.0, 107.0, 108.0],
            "low": [98.0, 100.0, 99.0, 102.0, 104.0],
            "close": [103.0, 101.0, 103.0, 106.0, 107.0],
            "volume": [1000, 1200, 1100, 1500, 1400],
        },
        index=dates,
    )
    df.index.name = "timestamp"
    return df


def test_valid_dataset(cleaner: DataCleaner, valid_ohlcv_df: pd.DataFrame):
    """Test validation on a perfectly valid OHLCV dataset."""
    res = cleaner.validate(valid_ohlcv_df)
    assert res.is_valid is True
    assert res.invalid_rows_count == 0
    assert res.duplicate_timestamps_count == 0
    assert res.is_chronological is True
    assert len(res.issues) == 0


def test_empty_dataset_validation(cleaner: DataCleaner):
    """Test validation on an empty dataset."""
    empty_df = pd.DataFrame()
    res = cleaner.validate(empty_df)
    assert res.is_valid is False
    assert "DataFrame is empty" in res.issues[0]

    cleaned, clean_res = cleaner.clean(empty_df)
    assert cleaned.empty is True


def test_chronological_sorting_validation(cleaner: DataCleaner, valid_ohlcv_df: pd.DataFrame):
    """Test detection and automated cleaning of unsorted timestamps."""
    # Reverse dates to break chronological ordering
    unsorted_df = valid_ohlcv_df.iloc[::-1].copy()
    
    res = cleaner.validate(unsorted_df)
    assert res.is_chronological is False
    assert res.is_valid is False

    cleaned_df, clean_res = cleaner.clean(unsorted_df)
    assert cleaned_df.index.is_monotonic_increasing is True
    assert clean_res.is_chronological is True


def test_duplicate_timestamp_detection(cleaner: DataCleaner, valid_ohlcv_df: pd.DataFrame):
    """Test detection and removal of duplicate timestamps."""
    dup_df = valid_ohlcv_df.copy()
    # Duplicate the second row
    dup_row = dup_df.iloc[[1]].copy()
    dup_df = pd.concat([dup_df.iloc[:2], dup_row, dup_df.iloc[2:]])

    res = cleaner.validate(dup_df)
    assert res.duplicate_timestamps_count == 1
    assert res.is_valid is False

    cleaned_df, clean_res = cleaner.clean(dup_df)
    assert cleaned_df.index.duplicated().sum() == 0
    assert len(cleaned_df) == len(valid_ohlcv_df)


def test_ohlc_relationship_validation(cleaner: DataCleaner, valid_ohlcv_df: pd.DataFrame):
    """Test detection of invalid OHLC relationships (e.g. High < Low, High < Close, Low > Open)."""
    invalid_df = valid_ohlcv_df.copy()
    # High < Low in row 0
    invalid_df.iloc[0, invalid_df.columns.get_loc("high")] = 90.0
    # Low > Open in row 2
    invalid_df.iloc[2, invalid_df.columns.get_loc("low")] = 110.0

    res = cleaner.validate(invalid_df)
    assert res.is_valid is False
    assert res.invalid_rows_count >= 2

    cleaned_df, clean_res = cleaner.clean(invalid_df)
    assert len(cleaned_df) == len(valid_ohlcv_df) - 2
    assert clean_res.is_valid is True


def test_invalid_non_positive_prices(cleaner: DataCleaner, valid_ohlcv_df: pd.DataFrame):
    """Test detection of non-positive prices (<= 0)."""
    invalid_df = valid_ohlcv_df.copy()
    # Zero price in open
    invalid_df.iloc[1, invalid_df.columns.get_loc("open")] = 0.0
    # Negative price in close
    invalid_df.iloc[3, invalid_df.columns.get_loc("close")] = -5.0

    res = cleaner.validate(invalid_df)
    assert res.is_valid is False
    assert res.invalid_rows_count >= 2

    cleaned_df, clean_res = cleaner.clean(invalid_df)
    assert len(cleaned_df) == len(valid_ohlcv_df) - 2


def test_missing_value_validation(cleaner: DataCleaner, valid_ohlcv_df: pd.DataFrame):
    """Test detection and dropping of NaN missing values in price columns."""
    missing_df = valid_ohlcv_df.copy()
    missing_df.iloc[1, missing_df.columns.get_loc("close")] = np.nan

    res = cleaner.validate(missing_df)
    assert res.is_valid is False
    assert res.missing_counts["close"] == 1

    cleaned_df, clean_res = cleaner.clean(missing_df)
    assert len(cleaned_df) == len(valid_ohlcv_df) - 1
    assert clean_res.missing_counts["close"] == 0


def test_optional_volume_for_index(cleaner: DataCleaner, valid_ohlcv_df: pd.DataFrame):
    """Test that volume missing/zero is treated as optional for index datasets."""
    index_df = valid_ohlcv_df.copy()
    index_df["volume"] = 0  # Zero volume common for index

    res = cleaner.validate(index_df, is_index=True)
    assert res.is_valid is True

    # Test negative volume (must fail even for index)
    index_df.iloc[0, index_df.columns.get_loc("volume")] = -100
    res_neg = cleaner.validate(index_df, is_index=True)
    assert res_neg.is_valid is False
