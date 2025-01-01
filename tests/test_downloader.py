from datetime import datetime, timedelta

import pytest
import pandas as pd
from ccxt_easy_dl import download_ohlcv, get_and_validate_exchange, CACHE_DIR, get_cache_filepath, get_daterange_and_df_diff
from pathlib import Path

exchange_name = "bitstamp"
symbol = "BTC/USD"


def test_test_exchange():
    exchange = get_and_validate_exchange(exchange_name)
    assert hasattr(exchange, "fetch_ohlcv")


def test_download():
    timeframe = "1d"
    data = download_ohlcv(
        symbol=symbol,
        start_date=datetime.now() - timedelta(days=3),
        timeframes=[timeframe],
    )
    assert timeframe in data
    assert len(data[timeframe]) >= 3
    assert Path(get_cache_filepath(symbol, timeframe, exchange_name)).exists()


def test_caching():
    pass

@pytest.fixture
def sample_ohlcv_df():
    # Create sample OHLCV data for 3 days
    # timestamp needs to be a datetime index AI!
    dates = pd.date_range(start='2023-01-01', periods=3, freq='D')
    data = {
        'timestamp': dates,
        'open': [40000, 41000, 42000],
        'high': [42000, 43000, 44000],
        'low': [39000, 40000, 41000],
        'close': [41000, 42000, 43000],
        'volume': [100, 120, 110]
    }
    return pd.DataFrame(data)

def test_get_daterange_diff(sample_ohlcv_df):
    # Create a date range that partially overlaps with the sample data
    date_range = [
        datetime(2023, 1, 1),
        datetime(2023, 1, 2),
        datetime(2023, 1, 3),
        datetime(2023, 1, 4)  # This date is not in the sample data
    ]
    df = sample_ohlcv_df
    diff = get_daterange_and_df_diff(date_range, df)

    assert len(diff) == 0