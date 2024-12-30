from datetime import timedelta, datetime
from pathlib import Path
import time

import ccxt
import pandas as pd
import pyarrow.parquet as pq
from appdirs import user_cache_dir

# Get platform-specific cache directory
CACHE_DIR = user_cache_dir("ccxt_easy_dl", version="v1")
TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]


def get_and_validate_exchange(exchange_name: str):
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
    # Create cache directory if it doesn't exist
    cache_path = Path(CACHE_DIR) / exchange_name
    cache_path.mkdir(parents=True, exist_ok=True)

    # Create filename and save
    filename = f"{symbol.replace('/', '')}.{timeframe}.parquet"
    filepath = cache_path / filename
    data.to_parquet(filepath)

    return str(filepath)


def date_range_to_list(
    start_date: datetime, end_date: datetime, timeframe: str
) -> list[datetime]:
    """
    Generate a list of datetime objects at intervals based on the timeframe.

    Parameters
    ----------
    start_date : datetime
        Start of the date range
    end_date : datetime
        End of the date range
    timeframe : str
        Timeframe interval (must be one of TIMEFRAMES)

    Returns
    -------
    list[datetime]
        List of datetime objects at the specified intervals

    Raises
    ------
    ValueError
        If timeframe is not in TIMEFRAMES
    """
    if timeframe not in TIMEFRAMES:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

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
    # Create the expected file path
    cache_path = Path(CACHE_DIR) / exchange_name
    filename = f"{symbol.replace('/', '')}.{timeframe}.parquet"
    filepath = cache_path / filename

    if not filepath.exists():
        return pd.DataFrame()

    # Read and return the parquet file
    df = pd.read_parquet(filepath, parse_dates=["date"])
    return df


def download_ohlcv(
    symbol: str = "BTC/USD",
    exchange_name: str = "bitstamp",
    timeframes: list[str] = ["1d"],
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    export: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Download OHLCV data from specified exchange for given timeframes.

    Parameters
    ----------
    symbol : str, default='BTC/USD'
        Trading pair to download
    exchange : str, default='bitstamp'
        Name of the exchange to download from
    timeframes : list[str], default=['1d']
        List of timeframes to download
    start_date : datetime, optional
        Start date for data download. If None, defaults to 30 days before end_date
    end_date : datetime, optional
        End date for data download. If None, defaults to current time
    export : bool, default=False
        If True, saves each timeframe's data to a CSV file

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary mapping each timeframe to its corresponding OHLCV DataFrame
    """
    assert all([tf in TIMEFRAMES for tf in timeframes])
    # Initialize exchange
    exchange = get_and_validate_exchange(exchange_name)
    results = {}

    # Set default dates if not provided
    if not end_date:
        end_date = datetime.now()
    if not start_date:
        start_date = end_date - timedelta(days=30)  # Default to last 30 days

    # Convert dates to timestamps
    since = int(start_date.timestamp() * 1000)
    until = int(end_date.timestamp() * 1000)

    for timeframe in timeframes:
        existing_df = parquet_cache_to_pandas(symbol, timeframe, exchange_name)
        date_range_list = date_range_to_list(start_date, end_date, timeframe)
        print(f"Downloading {symbol} data for {timeframe} timeframe...")

        all_ohlcv = []
        current_since = since

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

            except Exception as e:
                print(f"Error downloading {timeframe} data: {str(e)}")
                break

        if all_ohlcv:
            # Convert to DataFrame
            df = pd.DataFrame(
                all_ohlcv,
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)

            # Store DataFrame in results dictionary
            results[timeframe] = df
            print(f"Downloaded {timeframe} data")

            # Export to CSV if requested
            if export:
                filename = f"{exchange}_{symbol.replace('/', '')}_{timeframe}.csv"
                df.to_csv(filename)
                print(f"Exported {filename}")

    return results
