from __future__ import annotations
from datetime import timedelta, datetime
from pathlib import Path
import time
import logging

import ccxt
import pandas as pd
import pyarrow.parquet as pq
from appdirs import user_cache_dir
from rich.logging import RichHandler
from rich.console import Console
from rich import print as rprint
from alive_progress import alive_bar

# Configure rich logging
FORMAT = "%(message)s"
logging.basicConfig(
    level="INFO",
    format=FORMAT,
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)

logger = logging.getLogger("ccxt_easy_dl")
console = Console()

# Default cache directory
_CACHE_DIR = user_cache_dir("ccxt_easy_dl", version="v1")
TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]

def set_cache_dir(path: str | Path) -> None:
    """
    Set the cache directory for CCXT Easy DL.

    Parameters
    ----------
    path : str | Path
        The path to use as the cache directory
    """
    global _CACHE_DIR
    _CACHE_DIR = str(path)

def get_cache_dir() -> str:
    """
    Get the current cache directory.

    Returns
    -------
    str
        The current cache directory path
    """
    return _CACHE_DIR


def get_and_validate_exchange(exchange_name: str) -> ccxt.Exchange:
    """
    Ensure that the exchange exists in ccxt.exchanges and return the instance

    Parameters
    ----------
    exchange_name : str
        Name of the exchange to validate and initialize

    Returns
    -------
    ccxt.Exchange
            Configured exchange instance

    Raises
    ------
    ValueError
        If exchange_name is not found in ccxt.exchanges
    """
    if exchange_name not in ccxt.exchanges:
        raise ValueError(f"Exchange '{exchange_name}' not found in ccxt.exchanges")

    exchange = getattr(ccxt, exchange_name)(
        {"enableRateLimit": True}  # required by the exchange
    )
    return exchange


def pandas_to_parquet_cache(
    symbol: str, timeframe: str, data: pd.DataFrame, exchange_name: str
):
    """
    Save pandas DataFrame to parquet file in cache directory.

    Parameters
    ----------
    symbol : str
        Trading pair symbol (e.g., 'BTC/USD')
    timeframe : str
        Timeframe of the data (e.g., '1d')
    data : pd.DataFrame
        OHLCV data to save
    exchange_name : str
        Name of the exchange (e.g., 'bitstamp')

    Returns
    -------
    str
        Path to the saved parquet file
    """
    filepath = get_cache_filepath(symbol, timeframe, exchange_name)
    # Create cache directory if it doesn't exist
    filepath.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(filepath)

    return str(filepath)


def round_down_start_date(start_date: datetime, timeframe: str) -> datetime:
    """
    Round down a datetime to the nearest timeframe interval.
    For example:
    - For 1m: rounds to the start of the minute
    - For 1h: rounds to the start of the hour
    - For 1d: rounds to the start of the day
    - For 1w: rounds to the start of the week (Monday)

    Parameters
    ----------
    start_date : datetime
        The datetime to round down
    timeframe : str
        Timeframe interval (must be one of TIMEFRAMES)

    Returns
    -------
    datetime
        The rounded down datetime

    Raises
    ------
    ValueError
        If timeframe is not in TIMEFRAMES
    """
    if timeframe not in TIMEFRAMES:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    if timeframe == "1m":
        return start_date.replace(second=0, microsecond=0)
    elif timeframe == "5m":
        minutes = start_date.minute - (start_date.minute % 5)
        return start_date.replace(minute=minutes, second=0, microsecond=0)
    elif timeframe == "15m":
        minutes = start_date.minute - (start_date.minute % 15)
        return start_date.replace(minute=minutes, second=0, microsecond=0)
    elif timeframe == "30m":
        minutes = start_date.minute - (start_date.minute % 30)
        return start_date.replace(minute=minutes, second=0, microsecond=0)
    elif timeframe == "1h":
        return start_date.replace(minute=0, second=0, microsecond=0)
    elif timeframe == "4h":
        hours = start_date.hour - (start_date.hour % 4)
        return start_date.replace(hour=hours, minute=0, second=0, microsecond=0)
    elif timeframe == "1d":
        return start_date.replace(hour=0, minute=0, second=0, microsecond=0)
    elif timeframe == "1w":
        # For weekly, round down to Monday of the week
        days_since_monday = start_date.weekday()  # Monday is 0
        return (start_date - timedelta(days=days_since_monday)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )


def date_range_to_list(
    start_date: datetime, end_date: datetime, timeframe: str
) -> list[datetime]:
    """
    Generate a list of datetime objects at intervals based on the timeframe.
    The start_date will be rounded down to the nearest timeframe interval.
    For example:
    - For 1m: rounds to the start of the minute
    - For 1h: rounds to the start of the hour
    - For 1d: rounds to the start of the day
    - For 1w: rounds to the start of the week (Monday)

    Parameters
    ----------
    start_date : datetime
        Start of the date range (will be converted to UTC if not already)
    end_date : datetime
        End of the date range (will be converted to UTC if not already)
    timeframe : str
        Timeframe interval (must be one of TIMEFRAMES)

    Returns
    -------
    list[datetime]
        List of datetime objects at the specified intervals in UTC

    Raises
    ------
    ValueError
        If timeframe is not in TIMEFRAMES
    """
    if timeframe not in TIMEFRAMES:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    # Convert input dates to UTC if they have timezone info
    if start_date.tzinfo is not None:
        start_date = start_date.astimezone(tz=None).replace(tzinfo=None)
    if end_date.tzinfo is not None:
        end_date = end_date.astimezone(tz=None).replace(tzinfo=None)

    # Round down start_date based on timeframe
    logger.debug("🔍 Rounded start_date: %s", start_date)

    # Convert timeframe to timedelta
    timeframe_map = {
        "1m": timedelta(minutes=1),
        "5m": timedelta(minutes=5),
        "15m": timedelta(minutes=15),
        "30m": timedelta(minutes=30),
        "1h": timedelta(hours=1),
        "4h": timedelta(hours=4),
        "1d": timedelta(days=1),
        "1w": timedelta(weeks=1),
    }

    delta = timeframe_map[timeframe]
    date_list = []
    current_date = start_date

    while current_date <= end_date:
        date_list.append(current_date)
        current_date += delta

    return date_list


def get_cache_filepath(symbol: str, timeframe: str, exchange_name: str) -> Path:
    """
    Get the file path for cached OHLCV data.

    Parameters
    ----------
    symbol : str
        Trading pair symbol (e.g., 'BTC/USD')
    timeframe : str
        Timeframe of the data (e.g., '1d')
    exchange_name : str
        Name of the exchange (e.g., 'bitstamp')

    Returns
    -------
    Path
        Path object pointing to the cached file
    """
    cache_path = Path(get_cache_dir()) / exchange_name
    filename = f"{symbol.replace('/', '')}.{timeframe}.parquet"
    return cache_path / filename


def parquet_cache_to_pandas(
    symbol: str, timeframe: str, exchange_name: str
) -> pd.DataFrame:
    """
    Load cached OHLCV data from parquet file into pandas DataFrame.

    Parameters
    ----------
    symbol : str
        Trading pair symbol (e.g., 'BTC/USD')
    timeframe : str
        Timeframe of the data (e.g., '1d')
    exchange_name : str
        Name of the exchange (e.g., 'bitstamp')

    Returns
    -------
    pd.DataFrame
        OHLCV data loaded from cache

    Raises
    ------
    FileNotFoundError
        If the cached file does not exist
    """
    filepath = get_cache_filepath(symbol, timeframe, exchange_name)

    if not filepath.exists():
        return pd.DataFrame()

    # Read and return the parquet file
    df = pd.read_parquet(filepath)
    return df


def get_daterange_and_df_diff(
    date_range: list[datetime], df: pd.DataFrame
) -> list[datetime]:
    """
    Get a list of dates that are in date_range but not in the DataFrame.

    Parameters
    ----------
    date_range : list[datetime]
        List of datetime objects to check
    df : pd.DataFrame
        DataFrame containing a 'timestamp' column to compare against

    Returns
    -------
    list[datetime]
        List of datetime objects that are in date_range but not in df
    """
    if df.empty:
        return date_range
    logger.debug("📊 DataFrame index: %s", df.index)
    # Ensure the DataFrame has a datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Convert both to sets of dates (without time) for comparison
    date_range_set = {dt.date() for dt in date_range}
    date_range_list = sorted(list(date_range_set))
    logger.debug("📅 Date range set size: %d (from %s to %s)", 
                len(date_range_set),
                date_range_list[0] if date_range_list else "N/A",
                date_range_list[-1] if date_range_list else "N/A")

    df_dates_set = {dt.date() for dt in df.index}
    logger.debug("📅 DataFrame dates set size: %d", len(df_dates_set))

    # Find dates in range that aren't in the DataFrame
    missing_dates = date_range_set - df_dates_set
    missing_dates_list = sorted(list(missing_dates))
    logger.debug("❌ Missing dates count: %d (from %s to %s)", 
                len(missing_dates),
                missing_dates_list[0] if missing_dates_list else "N/A",
                missing_dates_list[-1] if missing_dates_list else "N/A")

    # Convert back to datetime objects matching the original date_range
    return [dt for dt in date_range if dt.date() in missing_dates]


def fetch_data_from_exchange(symbol: str, exchange: ccxt.Exchange, timeframe: str, since: int, until: int) -> pd.DataFrame:
    """Fetch OHLCV data from exchange and return as DataFrame.

    Parameters
    ----------
    symbol : str
        Trading pair symbol
    exchange : ccxt.Exchange
        Exchange instance to fetch from
    timeframe : str
        Timeframe to fetch
    since : int
        Start timestamp in milliseconds
    until : int
        End timestamp in milliseconds

    Returns
    -------
    pd.DataFrame
        OHLCV data with datetime index and columns [open, high, low, close, volume]
    """
    all_ohlcv = []
    
    # Calculate total number of iterations needed
    timeframe_ms = {
        "1m": 60000, "5m": 300000, "15m": 900000, "30m": 1800000,
        "1h": 3600000, "4h": 14400000, "1d": 86400000, "1w": 604800000
    }
    # Adjust since by subtracting one timeframe interval to make it inclusive
    current_since = since - timeframe_ms[timeframe]
    
    ms_per_iteration = timeframe_ms[timeframe] * 1000  # 1000 candles per request
    total_iterations = max(1, (until - current_since) // ms_per_iteration + 1)
    
    with alive_bar(total_iterations, title=f"Fetching {symbol} {timeframe} data", spinner="waves") as bar:
        while current_since < until:
            try:
                # Fetch OHLCV data
                ohlcv = exchange.fetch_ohlcv(
                    symbol,
                    timeframe=timeframe,
                    since=current_since,
                    limit=1000,  # Maximum number of candles per request
                )

                if not ohlcv:
                    break

                all_ohlcv.extend(ohlcv)

                # Update the current_since for next iteration
                current_since = ohlcv[-1][0] + 1

                # Rate limiting
                time.sleep(exchange.rateLimit / 1000)  # Convert to seconds
                bar()

            except Exception as e:
                print(f"Error downloading {timeframe} data: {str(e)}")
                break

    if not all_ohlcv:
        return pd.DataFrame()
        
    # Define all possible columns
    columns = ["timestamp", "open", "high", "low", "close", "volume"]
    
    # Create DataFrame with all columns
    df = pd.DataFrame(all_ohlcv, columns=columns[:len(all_ohlcv[0])])
    
    # Set timestamp as index
    if "timestamp" in df.columns:
        df.set_index("timestamp", inplace=True)
        df.index = pd.to_datetime(df.index, unit="ms")
    
    return df


def _prepare_dates(start_date: datetime | None, end_date: datetime | None) -> tuple[datetime, datetime]:
    """Prepare start and end dates, applying defaults if needed."""
    if not end_date:
        end_date = datetime.now()
    if not start_date:
        start_date = end_date - timedelta(days=30)
    return start_date, end_date

def _handle_existing_data(
    existing_df: pd.DataFrame,
    date_range_list: list[datetime],
    working_start_date: datetime,
    end_date: datetime
) -> tuple[pd.DataFrame | None, tuple[int, int] | None]:
    """Handle existing data and determine if new data needs to be fetched.
    
    Returns:
    --------
    tuple[pd.DataFrame | None, tuple[int, int] | None]
        - DataFrame if cached data can be used directly, None if new data needed
        - Tuple of (since, until) timestamps if new data needed, None otherwise
    """
    if existing_df.empty:
        return None, None
        
    date_diff = get_daterange_and_df_diff(date_range_list, existing_df)
    # Sanity check - make sure date_diff is not empty before accessing elements
    if date_diff:
        logger.debug("📅 Length of date differences: %s. First date: %s. Last date: %s.", 
                    len(date_diff), date_diff[0], date_diff[-1])
    else:
        logger.debug("📅 No date differences found")
    if not date_diff:
        # Use existing data
        mask = (existing_df.index >= working_start_date) & (existing_df.index <= end_date)
        return existing_df[mask], None
        
    # Need to fetch new data
    min_date = min(date_diff)
    max_date = max(date_diff)
    since = int(min_date.timestamp() * 1000)
    until = int(max_date.timestamp() * 1000)
    return None, (since, until)

def _merge_and_save_data(
    new_df: pd.DataFrame,
    existing_df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    exchange_name: str,
    working_start_date: datetime,
    end_date: datetime,
    export: bool = False
) -> pd.DataFrame:
    """Merge new and existing data, save to cache, and return final DataFrame."""
    if not existing_df.empty:
        logger.debug("🔄 Merging data - existing: %s, new: %s", existing_df.shape, new_df.shape)
        df = pd.concat([existing_df, new_df])
        df = df.drop_duplicates()
        df = df.sort_index()  # Sort by index after merging
        logger.debug("📊 After merge - first 5 rows:\n%s", df.head())
        logger.debug("📊 After merge - last 5 rows:\n%s", df.tail())
    else:
        df = new_df

    cache_path = pandas_to_parquet_cache(symbol, timeframe, df, exchange_name)
    logger.debug("💾 Saved to cache: %s", cache_path)

    if export:
        filename = f"{exchange_name}_{symbol.replace('/', '')}_{timeframe}.csv"
        df.to_csv(filename)
        logger.debug("📁 Exported to %s", filename)

    logger.debug("📊 Final DataFrame index type: %s", type(df.index))
    logger.debug("📊 Final DataFrame index range: %s to %s", df.index.min(), df.index.max())
    logger.debug("📊 Working date range: %s to %s", working_start_date, end_date)
    logger.debug("📊 Index dtype: %s", df.index.dtype)
    
    # Ensure working_start_date and end_date are timezone-naive if the index is
    if df.index.tz is not None:
        if working_start_date.tzinfo is None:
            working_start_date = working_start_date.replace(tzinfo=df.index.tz)
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=df.index.tz)
    elif df.index.tz is None and (working_start_date.tzinfo is not None or end_date.tzinfo is not None):
        working_start_date = working_start_date.replace(tzinfo=None)
        end_date = end_date.replace(tzinfo=None)

    result = df.loc[working_start_date:end_date]
    logger.debug("📊 Result shape: %s", result.shape)
    return result

def download_ohlcv(
    symbol: str = "BTC/USD",
    exchange_name: str = "bitstamp",
    timeframes: list[str] = ["1d"],
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    export: bool = False,
) -> dict[str, pd.DataFrame]:
    """Download OHLCV data from specified exchange for given timeframes."""
    assert all([tf in TIMEFRAMES for tf in timeframes])
    exchange = get_and_validate_exchange(exchange_name)
    results = {}
    
    start_date, end_date = _prepare_dates(start_date, end_date)

    for timeframe in timeframes:
        working_start_date = round_down_start_date(start_date, timeframe)
        date_range_list = date_range_to_list(working_start_date, end_date, timeframe)
        
        existing_df = parquet_cache_to_pandas(symbol, timeframe, exchange_name)
        cached_df, time_range = _handle_existing_data(
            existing_df, date_range_list, working_start_date, end_date
        )
        
        if cached_df is not None:
            results[timeframe] = cached_df
            continue
            
        if time_range:
            since, until = time_range
        else:
            since = int(working_start_date.timestamp() * 1000)
            until = int(end_date.timestamp() * 1000)
            
        logger.debug("⬇️  Downloading %s %s from %s to %s...", 
                    symbol, timeframe, 
                    datetime.fromtimestamp(since/1000),
                    datetime.fromtimestamp(until/1000))
                    
        new_df = fetch_data_from_exchange(symbol, exchange, timeframe, since, until)
        
        results[timeframe] = _merge_and_save_data(
            new_df, existing_df, symbol, timeframe, exchange_name,
            working_start_date, end_date, export
        )

    return results

def download_multiple_ohlcv(
    symbols: list[str],
    timeframes: list[str],
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    exchange_name: str = "bitstamp",
    export: bool = False,
) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Download OHLCV data for multiple symbols and timeframes.

    Parameters
    ----------
    symbols : list[str]
        List of trading pairs to download (e.g., ['BTC/USD', 'ETH/USD'])
    timeframes : list[str]
        List of timeframes to download
    start_date : datetime, optional
        Start date for data download. If None, defaults to 30 days before end_date
    end_date : datetime, optional
        End date for data download. If None, defaults to current time
    exchange_name : str, default='bitstamp'
        Name of the exchange to download from
    export : bool, default=False
        If True, saves each timeframe's data to a CSV file

    Returns
    -------
    dict[str, dict[str, pd.DataFrame]]
        Nested dictionary mapping each symbol to its timeframes and corresponding OHLCV DataFrames
    """
    results = {}
    
    for symbol in symbols:
        logger.debug(f"📥 Downloading data for {symbol}")
        symbol_data = download_ohlcv(
            symbol=symbol,
            timeframes=timeframes,
            start_date=start_date,
            end_date=end_date,
            exchange_name=exchange_name,
            export=export
        )
        results[symbol] = symbol_data
        
    return results
