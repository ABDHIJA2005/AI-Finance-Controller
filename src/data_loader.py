"""
Data Loader Module.

Provides configurable, robust historical market data downloading via public APIs
(e.g., Yahoo Finance) with column normalization, index standardization, duplicate
timestamp detection, and chronological ordering.
"""

import datetime
import logging
from typing import Optional, Union

import pandas as pd
import yfinance as yf

from src.storage import StorageManager

logger = logging.getLogger(__name__)


class DataLoaderError(Exception):
    """Custom exception raised when historical market data downloading fails."""

    pass


class DataLoader:
    """Configurable downloader for historical market OHLCV data.

    Fetches market time-series data from public data sources (yfinance) and standardizes
    schema, datatypes, column names, and DatetimeIndex.
    """

    COLUMN_MAPPING = {
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "adj close": "adj_close",
        "adjclose": "adj_close",
        "volume": "volume",
    }

    def __init__(self, storage_manager: Optional[StorageManager] = None) -> None:
        """Initialize DataLoader.

        Parameters
        ----------
        storage_manager : Optional[StorageManager], default None
            Optional StorageManager instance for dataset caching.
        """
        self.storage_manager = storage_manager or StorageManager()

    def fetch_data(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = "1d",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Download or retrieve cached historical OHLCV market data for a symbol.

        Parameters
        ----------
        symbol : str
            Market ticker symbol (e.g. '^NSEI' for NIFTY 50 index).
        start_date : Optional[str], default None
            Configured start date in 'YYYY-MM-DD' format. If None, defaults to max history available.
        end_date : Optional[str], default None
            Configured end date in 'YYYY-MM-DD' format. If None, defaults to current date.
        interval : str, default '1d'
            Data sampling interval (e.g. '1d', '1wk', '1mo').
        use_cache : bool, default True
            If True, checks for existing cached raw dataset in storage before fetching.

        Returns
        -------
        pd.DataFrame
            Normalized, chronologically sorted OHLCV DataFrame with DatetimeIndex.

        Raises
        ------
        DataLoaderError
            If download fails, returns empty dataset, or API encounters an unrecoverable error.
        """
        # Check cache if enabled
        if use_cache and self.storage_manager.dataset_exists(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            tier="raw",
        ):
            logger.info(
                f"Loading cached raw dataset for {symbol} ({start_date} to {end_date})"
            )
            return self.storage_manager.load_dataset(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                interval=interval,
                tier="raw",
            )

        logger.info(
            f"Downloading market data for symbol '{symbol}' from yfinance (start={start_date}, end={end_date}, interval={interval})"
        )

        try:
            # Download using yfinance Ticker API
            ticker_obj = yf.Ticker(symbol)
            
            # Formulate yfinance parameters
            kwargs = {"interval": interval, "auto_adjust": False}
            if start_date:
                kwargs["start"] = start_date
            if end_date:
                kwargs["end"] = end_date
            if not start_date and not end_date:
                kwargs["period"] = "max"

            raw_df = ticker_obj.history(**kwargs)

        except Exception as exc:
            logger.error(f"yfinance download failed for symbol '{symbol}': {exc}")
            raise DataLoaderError(
                f"Failed to fetch market data for symbol '{symbol}': {exc}"
            ) from exc

        if raw_df is None or raw_df.empty:
            logger.error(f"yfinance returned empty DataFrame for symbol '{symbol}'.")
            raise DataLoaderError(
                f"No data returned for symbol '{symbol}' for period {start_date} to {end_date}."
            )

        # Process and normalize raw DataFrame
        df = self._normalize_dataframe(raw_df, symbol)

        # Save raw dataset to storage cache
        try:
            self.storage_manager.save_dataset(
                df=df,
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                interval=interval,
                tier="raw",
            )
        except Exception as exc:
            logger.warning(f"Could not save raw dataset cache: {exc}")

        return df

    def _normalize_dataframe(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Standardize column names, index, dtypes, and ordering of downloaded DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Raw DataFrame returned from yfinance.
        symbol : str
            Symbol name for logging context.

        Returns
        -------
        pd.DataFrame
            Normalized DataFrame.
        """
        normalized_df = df.copy()

        # Flatten multi-index columns if present (yfinance 0.2.x edge case)
        if isinstance(normalized_df.columns, pd.MultiIndex):
            logger.info("Flattening MultiIndex columns returned by yfinance.")
            normalized_df.columns = [
                col[0] if isinstance(col, tuple) else col
                for col in normalized_df.columns
            ]

        # Standardize column names to lowercase snake_case
        normalized_df.columns = [
            str(col).strip().lower().replace(" ", "_")
            for col in normalized_df.columns
        ]

        # Map mapped column names if needed
        renamed_cols = {}
        for col in normalized_df.columns:
            cleaned_col = col.replace("_", " ")
            if cleaned_col in self.COLUMN_MAPPING:
                renamed_cols[col] = self.COLUMN_MAPPING[cleaned_col]
        normalized_df.rename(columns=renamed_cols, inplace=True)

        # Standardize DatetimeIndex
        if not isinstance(normalized_df.index, pd.DatetimeIndex):
            normalized_df.index = pd.to_datetime(normalized_df.index)

        # Standardize index name
        normalized_df.index.name = "timestamp"

        # Sort chronologically
        if not normalized_df.index.is_monotonic_increasing:
            logger.info("Sorting index chronologically.")
            normalized_df = normalized_df.sort_index(ascending=True)

        # Detect duplicate timestamps
        duplicates = int(normalized_df.index.duplicated().sum())
        if duplicates > 0:
            logger.warning(
                f"Symbol '{symbol}' downloaded dataset contains {duplicates} duplicate timestamp(s)."
            )

        logger.info(
            f"Successfully normalized dataset for '{symbol}': {len(normalized_df)} rows, "
            f"date range [{normalized_df.index.min()} to {normalized_df.index.max()}]."
        )
        return normalized_df
