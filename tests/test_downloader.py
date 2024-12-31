from datetime import datetime, timedelta

import pandas as pd
from ccxt_easy_dl import download_ohlcv, get_and_validate_exchange, CACHE_DIR, get_cache_filepath
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

def test_get_daterange_diff():
    date_range = None
    df = pd.DataFrame() # populate this with fake ohlcv data. make a fixture AI!
    