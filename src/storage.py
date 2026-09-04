"""
Storage Manager Module.

Provides deterministic local storage and retrieval for market OHLCV datasets using
Apache Parquet format, enforcing clear separation between raw downloaded data and
cleaned/processed data.
"""

import logging
import re
from pathlib import Path
from typing import Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)


class StorageManager:
    """Manager for persisting and loading market OHLCV data to/from Parquet files.

    Enforces deterministic file naming and directory structure separating raw
    and processed datasets.
    """

    def __init__(self, base_data_dir: Union[str, Path] = "data") -> None:
        """Initialize StorageManager.

        Parameters
        ----------
        base_data_dir : Union[str, Path], default "data"
            Root directory path for market data storage.
        """
        self.base_dir = Path(base_data_dir)
        self.raw_dir = self.base_dir / "raw"
        self.processed_dir = self.base_dir / "processed"

        # Ensure storage directories exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def sanitize_symbol(symbol: str) -> str:
        """Sanitize market ticker symbol for safe filesystem path usage.

        Parameters
        ----------
        symbol : str
            Raw ticker symbol (e.g., '^NSEI', 'AAPL', 'EURUSD=X').

        Returns
        -------
        str
            Sanitized symbol string (e.g., 'NSEI', 'AAPL', 'EURUSD_X').
        """
        # Replace caret '^' or other non-alphanumeric characters with safe characters
        sanitized = re.sub(r"[^\w\-]", "_", symbol.strip())
        return sanitized.strip("_")

    def _normalize_date_str(self, date_val: Optional[str]) -> str:
        """Format date input to YYYYMMDD string representation for filenames.

        Parameters
        ----------
        date_val : Optional[str]
            Date string or None.

        Returns
        -------
        str
            Formatted date string or 'latest' if None.
        """
        if not date_val:
            return "latest"
        try:
            return pd.to_datetime(date_val).strftime("%Y%m%d")
        except Exception:
            return str(date_val).replace("-", "")

    def get_filename(
        self,
        symbol: str,
        start_date: Optional[str],
        end_date: Optional[str],
        interval: str = "1d",
    ) -> str:
        """Generate a deterministic file name for a dataset.

        Parameters
        ----------
        symbol : str
            Market instrument ticker symbol.
        start_date : Optional[str]
            Configured start date.
        end_date : Optional[str]
            Configured end date.
        interval : str, default "1d"
            Data sampling interval.

        Returns
        -------
        str
            Deterministic Parquet filename (e.g. 'NSEI_20150101_20260825_1d.parquet').
        """
        clean_sym = self.sanitize_symbol(symbol)
        start_str = self._normalize_date_str(start_date)
        end_str = self._normalize_date_str(end_date)
        clean_interval = self.sanitize_symbol(interval)
        return f"{clean_sym}_{start_str}_{end_str}_{clean_interval}.parquet"

    def get_filepath(
        self,
        symbol: str,
        start_date: Optional[str],
        end_date: Optional[str],
        interval: str = "1d",
        tier: str = "processed",
    ) -> Path:
        """Get the absolute/relative Path for a dataset.

        Parameters
        ----------
        symbol : str
            Market ticker symbol.
        start_date : Optional[str]
            Start date string.
        end_date : Optional[str]
            End date string.
        interval : str, default "1d"
            Sampling interval.
        tier : str, default "processed"
            Data tier ('raw' or 'processed').

        Returns
        -------
        Path
            Path object pointing to the file.
        """
        filename = self.get_filename(symbol, start_date, end_date, interval)
        if tier.lower() == "raw":
            return self.raw_dir / filename
        elif tier.lower() == "processed":
            return self.processed_dir / filename
        else:
            raise ValueError(f"Invalid data tier '{tier}'. Must be 'raw' or 'processed'.")

    def dataset_exists(
        self,
        symbol: str,
        start_date: Optional[str],
        end_date: Optional[str],
        interval: str = "1d",
        tier: str = "processed",
    ) -> bool:
        """Check if a dataset file exists in local storage.

        Parameters
        ----------
        symbol : str
            Market ticker symbol.
        start_date : Optional[str]
            Start date.
        end_date : Optional[str]
            End date.
        interval : str, default "1d"
            Interval.
        tier : str, default "processed"
            Tier ('raw' or 'processed').

        Returns
        -------
        bool
            True if the file exists and is non-empty, False otherwise.
        """
        path = self.get_filepath(symbol, start_date, end_date, interval, tier)
        return path.exists() and path.stat().st_size > 0

    def save_dataset(
        self,
        df: pd.DataFrame,
        symbol: str,
        start_date: Optional[str],
        end_date: Optional[str],
        interval: str = "1d",
        tier: str = "processed",
    ) -> Path:
        """Save a DataFrame to Parquet storage.

        Parameters
        ----------
        df : pd.DataFrame
            Market OHLCV DataFrame with DatetimeIndex.
        symbol : str
            Market ticker symbol.
        start_date : Optional[str]
            Start date.
        end_date : Optional[str]
            End date.
        interval : str, default "1d"
            Interval.
        tier : str, default "processed"
            Data storage tier ('raw' or 'processed').

        Returns
        -------
        Path
            Path where the Parquet file was saved.
        """
        if df.empty:
            raise ValueError("Cannot save an empty DataFrame to Parquet storage.")

        filepath = self.get_filepath(symbol, start_date, end_date, interval, tier)
        
        # Write to parquet preserving index
        df.to_parquet(filepath, engine="pyarrow", index=True)
        logger.info(f"Successfully saved {len(df)} rows to {tier} storage: {filepath}")
        return filepath

    def load_dataset(
        self,
        symbol: str,
        start_date: Optional[str],
        end_date: Optional[str],
        interval: str = "1d",
        tier: str = "processed",
    ) -> pd.DataFrame:
        """Load a DataFrame from Parquet storage.

        Parameters
        ----------
        symbol : str
            Market ticker symbol.
        start_date : Optional[str]
            Start date.
        end_date : Optional[str]
            End date.
        interval : str, default "1d"
            Interval.
        tier : str, default "processed"
            Data storage tier ('raw' or 'processed').

        Returns
        -------
        pd.DataFrame
            Loaded market OHLCV DataFrame with DatetimeIndex preserved.
        """
        filepath = self.get_filepath(symbol, start_date, end_date, interval, tier)
        if not filepath.exists():
            raise FileNotFoundError(f"No dataset file found at {filepath}")

        df = pd.read_parquet(filepath, engine="pyarrow")
        
        # Ensure DatetimeIndex type
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        logger.info(f"Successfully loaded {len(df)} rows from {tier} storage: {filepath}")
        return df
