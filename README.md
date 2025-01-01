# ccxt-easy-dl

A Python package that simplifies downloading and caching of cryptocurrency OHLCV (Open, High, Low, Close, Volume) data using the CCXT library.

## Features

- Easy downloading of OHLCV data from any exchange supported by CCXT
- Automatic caching of downloaded data in Parquet format
- Support for multiple timeframes (1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w)
- Smart cache management - only downloads missing data
- Platform-independent cache directory management
- Export capability to CSV format

## Installation

```bash
pip install ccxt-easy-dl
```

## Usage

Here's a simple example to download Bitcoin/USD data from Bitstamp:

```python
from datetime import datetime, timedelta
from ccxt_easy_dl import download_ohlcv

# Download last 30 days of BTC/USD data from Bitstamp
data = download_ohlcv(
    symbol="BTC/USD",
    exchange_name="bitstamp",
    timeframes=["1d"],
    start_date=datetime.now() - timedelta(days=30),
    end_date=datetime.now()
)

# Access the data
daily_data = data["1d"]
```

### Advanced Usage

```python
# Download multiple timeframes
data = download_ohlcv(
    symbol="ETH/USD",
    exchange_name="kraken",
    timeframes=["1h", "4h", "1d"],
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2023, 12, 31),
    export=True  # Also export to CSV files
)
```

## Supported Timeframes

- 1m (1 minute)
- 5m (5 minutes)
- 15m (15 minutes)
- 30m (30 minutes)
- 1h (1 hour)
- 4h (4 hours)
- 1d (1 day)
- 1w (1 week)

## Caching

Data is automatically cached in Parquet format in the platform-specific user cache directory:
- Linux: `~/.cache/ccxt_easy_dl/v1/`
- macOS: `~/Library/Caches/ccxt_easy_dl/v1/`
- Windows: `C:\Users\<username>\AppData\Local\ccxt_easy_dl\v1\`

The package intelligently manages the cache:
- Only downloads data that isn't already in the cache
- Automatically merges new data with existing cached data
- Uses efficient Parquet format for storage

## Dependencies

- ccxt
- pandas
- pyarrow
- appdirs
- loguru

## License

[Add your license here]
