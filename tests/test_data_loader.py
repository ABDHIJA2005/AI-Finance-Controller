"""
Unit tests for DataLoader module.
"""

from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from src.data_loader import DataLoader, DataLoaderError
from src.storage import StorageManager


@pytest.fixture
def tmp_storage(tmp_path) -> StorageManager:
    return StorageManager(base_data_dir=tmp_path)


@pytest.fixture
def loader(tmp_storage: StorageManager) -> DataLoader:
    return DataLoader(storage_manager=tmp_storage)


def test_dataframe_normalization(loader: DataLoader):
    """Test column normalization and index handling in DataLoader."""
    raw_df = pd.DataFrame(
        {
            "Open": [100.0, 101.0],
            "High": [105.0, 106.0],
            "Low": [99.0, 100.0],
            "Close": [104.0, 105.0],
            "Adj Close": [104.0, 105.0],
            "Volume": [1000, 1100],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-01"]),  # Out of order
    )

    normalized = loader._normalize_dataframe(raw_df, "^NSEI")

    assert normalized.index.is_monotonic_increasing is True
    assert "open" in normalized.columns
    assert "high" in normalized.columns
    assert "low" in normalized.columns
    assert "close" in normalized.columns
    assert "adj_close" in normalized.columns
    assert "volume" in normalized.columns


def test_fetch_data_failure_raises_exception(loader: DataLoader):
    """Test that DataLoader raises DataLoaderError when download fails."""
    with patch("yfinance.Ticker") as mock_ticker:
        mock_instance = MagicMock()
        mock_instance.history.side_effect = Exception("API Connection Error")
        mock_ticker.return_value = mock_instance

        with pytest.raises(DataLoaderError, match="Failed to fetch market data"):
            loader.fetch_data("^NSEI", start_date="2024-01-01", end_date="2024-01-05", use_cache=False)


def test_fetch_data_empty_raises_exception(loader: DataLoader):
    """Test that DataLoader raises DataLoaderError when empty data returned."""
    with patch("yfinance.Ticker") as mock_ticker:
        mock_instance = MagicMock()
        mock_instance.history.return_value = pd.DataFrame()
        mock_ticker.return_value = mock_instance

        with pytest.raises(DataLoaderError, match="No data returned for symbol"):
            loader.fetch_data("INVALID_TICKER", start_date="2024-01-01", end_date="2024-01-05", use_cache=False)
