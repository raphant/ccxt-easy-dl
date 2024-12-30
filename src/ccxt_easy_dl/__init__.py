from datetime import timedelta, datetime
from pathlib import Path
import time

import ccxt
import pandas as pd
import pyarrow.parquet as pq
from appdirs import user_cache_dir

# Get platform-specific cache directory
CACHE_DIR = user_cache_dir("ccxt_easy_dl", version="v1")


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


def pandas_to_parquet_cache(symbol: str, timeframe: str, data: pd.DataFrame, exchange_name: str):
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

def parquet_cache_to_pandas(symbol: str, timeframe: str, exchange_name: str) -> pd.DataFrame:
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
        raise FileNotFoundError(f"No cached data found at {filepath}")

    # Read and return the parquet file
    df = pd.read_parquet(filepath)
    return df
def download_ohlcv(
    symbol: str = "BTC/USD",
    exchange: str = "bitstamp",
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
    # Initialize exchange
    exchange = get_and_validate_exchange(exchange)
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
