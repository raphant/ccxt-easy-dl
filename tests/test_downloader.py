from datetime import datetime, timedelta
import tempfile
import shutil
from pathlib import Path
import logging

import pytest
import pandas as pd
from ccxt_easy_dl import (
    date_range_to_list,
    download_ohlcv,
    get_and_validate_exchange,
    get_cache_filepath,
    get_daterange_and_df_diff,
    set_cache_dir,
)

# Set ccxt_easy_dl logger to debug level
logging.getLogger("ccxt_easy_dl").setLevel(logging.DEBUG)

exchange_name = "bitstamp"
symbol = "BTC/USD"

@pytest.fixture(autouse=True)
def temp_cache_dir():
    """Create a temporary directory for cache and clean it up after tests."""
    temp_dir = tempfile.mkdtemp()
    set_cache_dir(temp_dir)
    yield temp_dir
    shutil.rmtree(temp_dir)


def test_test_exchange(temp_cache_dir):
    exchange = get_and_validate_exchange(exchange_name)
    assert hasattr(exchange, "fetch_ohlcv")



def test_caching(temp_cache_dir):
    """Test that data is properly cached and retrieved."""
    timeframe = "1d"
    end_date = datetime.now() - timedelta(days=1)
    start_date = end_date - timedelta(days=7)
    
    # First download - should fetch from exchange
    data1 = download_ohlcv(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        timeframes=[timeframe],
    )
    
    # Verify cache file exists
    cache_file = get_cache_filepath(symbol, timeframe, exchange_name)
    assert cache_file.exists()
    
    # Second download - should use cache
    data2 = download_ohlcv(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        timeframes=[timeframe],
    )
    
    # Data should be identical
    pd.testing.assert_frame_equal(data1[timeframe], data2[timeframe])
    
    # Verify cache directory is temp directory
    assert Path(temp_cache_dir) in cache_file.parents

@pytest.fixture
def sample_ohlcv_df():
    # Create sample OHLCV data for 3 days
    dates = pd.date_range(start='2023-01-01', periods=3, freq='D')
    assert len(dates) == 3
    data = {
        'open': [40000, 41000, 42000],
        'high': [42000, 43000, 44000],
        'low': [39000, 40000, 41000],
        'close': [41000, 42000, 43000],
        'volume': [100, 120, 110]
    }
    df = pd.DataFrame(data, index=dates)
    df.index.name = 'timestamp'
    return df

def test_get_daterange_diff(sample_ohlcv_df, temp_cache_dir):
    # Create a date range that partially overlaps with the sample data
    date_range = [
        datetime(2023, 1, 1),
        datetime(2023, 1, 2),
        datetime(2023, 1, 3),
        datetime(2023, 1, 4)  # This date is not in the sample data
    ]
    df = sample_ohlcv_df
    diff = get_daterange_and_df_diff(date_range, df)

    assert len(diff) == 1
    assert diff[0] == datetime(2023, 1, 4)

@pytest.fixture
def mock_fetch_data(monkeypatch):
    """Mock fetch_data_from_exchange to return predictable data."""
    def mock_fetch(symbol: str, exchange, timeframe: str, since: int, until: int) -> pd.DataFrame:
        # Generate one candle per day
        data = []
        timestamps = []
        current_ts = since
        
        while current_ts <= until:
            # Use timestamp as seed for predictable values
            base_price = current_ts % 10000  # Simple way to get a base price
            timestamps.append(current_ts)
            data.append([
                base_price,  # open
                base_price * 1.1,  # high
                base_price * 0.9,  # low
                base_price * 1.05,  # close
                base_price * 100,  # volume
            ])
            # Move to next day
            current_ts += 86400000  # Add one day in milliseconds
            
        if not data:
            return pd.DataFrame()
            
        # Convert to DataFrame with timestamp index
        df = pd.DataFrame(
            data,
            columns=["open", "high", "low", "close", "volume"],
            index=pd.to_datetime(timestamps, unit="ms")
        )
        df.index.name = "timestamp"
        
        return df
    
    import ccxt_easy_dl
    monkeypatch.setattr(ccxt_easy_dl, 'fetch_data_from_exchange', mock_fetch)
    return mock_fetch

def test_download_with_gap(mock_fetch_data, temp_cache_dir):
    """Test downloading data with a gap in the middle."""
    timeframe = "1d"
    
    # First download - get 5 days of data
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 1, 5)
    download_ohlcv(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        timeframes=[timeframe],
    )
    
    # Second download - get data including a gap
    new_start = datetime(2023, 1, 3)  # Overlaps with previous data
    new_end = datetime(2023, 1, 7)    # Extends beyond previous data
    data2 = download_ohlcv(
        symbol=symbol,
        start_date=new_start,
        end_date=new_end,
        timeframes=[timeframe],
    )
    
    # Verify the results
    df = data2[timeframe]
    assert not df.empty
    assert df.index.min().date() == start_date.date()  # Should include earliest date
    assert df.index.max().date() == new_end.date()     # Should include latest date
    assert len(df) == 7  # Should have all days without duplicates
    assert df.index.is_monotonic_increasing  # Should be sorted
    assert not df.index.has_duplicates      # Should have no duplicates

def test_download(temp_cache_dir):
    timeframe = "1d"
    start_date = datetime.now() - timedelta(days=7)
    end_date = datetime.now() - timedelta(days=1)
    
    # Create expected date range
    expected_dates = date_range_to_list(start_date, end_date, timeframe)
    
    data = download_ohlcv(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        timeframes=[timeframe],
    )
    assert timeframe in data
    assert len(data[timeframe]) == len(expected_dates)
    assert Path(get_cache_filepath(symbol, timeframe, exchange_name)).exists()

def test_download_multiple_coins(temp_cache_dir):
    """Test downloading data for multiple coins simultaneously."""
    timeframe = "1d"
    symbols = ["BTC/USD", "ETH/USD"]
    start_date = datetime.now() - timedelta(days=7)
    end_date = datetime.now() - timedelta(days=1)
    
    # Download data for multiple symbols
    data = download_ohlcv(
        symbol=symbols,
        start_date=start_date,
        end_date=end_date,
        timeframes=[timeframe],
    )
    
    # Verify results for each symbol
    for symbol in symbols:
        assert timeframe in data[symbol]
        expected_dates = date_range_to_list(start_date, end_date, timeframe)
        assert len(data[symbol][timeframe]) == len(expected_dates)
        assert Path(get_cache_filepath(symbol, timeframe, exchange_name)).exists()
