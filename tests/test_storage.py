"""
Unit tests for StorageManager and Parquet persistence.
"""

from pathlib import Path
import pandas as pd
import pytest

from src.storage import StorageManager


@pytest.fixture
def tmp_storage(tmp_path: Path) -> StorageManager:
    """Fixture providing a StorageManager with temporary directory."""
    return StorageManager(base_data_dir=tmp_path)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Sample DataFrame for testing storage functionality."""
    dates = pd.date_range(start="2024-01-01", periods=3, freq="D")
    df = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0],
            "high": [105.0, 106.0, 107.0],
            "low": [99.0, 100.0, 101.0],
            "close": [104.0, 105.0, 106.0],
            "volume": [1000, 1100, 1200],
        },
        index=dates,
    )
    df.index.name = "timestamp"
    return df


def test_deterministic_path(tmp_storage: StorageManager):
    """Test deterministic filename and path generation."""
    filename = tmp_storage.get_filename("^NSEI", "2020-01-01", "2024-01-01", "1d")
    assert filename == "NSEI_20200101_20240101_1d.parquet"

    raw_path = tmp_storage.get_filepath("^NSEI", "2020-01-01", "2024-01-01", "1d", tier="raw")
    processed_path = tmp_storage.get_filepath("^NSEI", "2020-01-01", "2024-01-01", "1d", tier="processed")

    assert raw_path.parent.name == "raw"
    assert processed_path.parent.name == "processed"


def test_storage_round_trip(tmp_storage: StorageManager, sample_df: pd.DataFrame):
    """Test saving and loading DataFrame preserves values, dtypes, and DatetimeIndex."""
    saved_path = tmp_storage.save_dataset(
        df=sample_df,
        symbol="^NSEI",
        start_date="2024-01-01",
        end_date="2024-01-03",
        interval="1d",
        tier="processed",
    )
    assert saved_path.exists()

    loaded_df = tmp_storage.load_dataset(
        symbol="^NSEI",
        start_date="2024-01-01",
        end_date="2024-01-03",
        interval="1d",
        tier="processed",
    )

    pd.testing.assert_frame_equal(sample_df, loaded_df, check_freq=False)
    assert isinstance(loaded_df.index, pd.DatetimeIndex)


def test_dataset_exists_check(tmp_storage: StorageManager, sample_df: pd.DataFrame):
    """Test existence check for cached datasets."""
    symbol = "NIFTY50"
    assert not tmp_storage.dataset_exists(symbol, "2024-01-01", "2024-01-03", "1d", "raw")

    tmp_storage.save_dataset(sample_df, symbol, "2024-01-01", "2024-01-03", "1d", "raw")
    assert tmp_storage.dataset_exists(symbol, "2024-01-01", "2024-01-03", "1d", "raw")


def test_save_empty_df_raises(tmp_storage: StorageManager):
    """Test that saving an empty DataFrame raises ValueError."""
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError, match="Cannot save an empty DataFrame"):
        tmp_storage.save_dataset(empty_df, "TEST", "2024-01-01", "2024-01-02")
