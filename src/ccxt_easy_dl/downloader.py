from datetime import timedelta
import datetime
import time

import ccxt
import pandas as pd


def download_ohlcv(symbol='BTC/USD', exchange='bitstamp', timeframes=['1d'], 
                   start_date=None, end_date=None):
    # type hints AI!
    """
    Download OHLCV data from specified exchange for given timeframes.
    
    Parameters
    ----------
    symbol : str, default='BTC/USD'
        Trading pair to download
    exchange : str, default='bitstamp'
        Name of the exchange to download from
    timeframes : list, default=['1d']
        List of timeframes to download
    start_date : datetime, optional
        Start date for data download. If None, defaults to 30 days before end_date
    end_date : datetime, optional
        End date for data download. If None, defaults to current time
    """
    
    # Initialize Bitstamp exchange
    assert exchange in ccxt.exchanges, "Exchange not found in exchanges"
    exchange = ccxt.bitstamp({
        'enableRateLimit': True,  # required by the exchange
    })
    
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
                    limit=1000  # Maximum number of candles per request
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
            df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Save to CSV
            filename = f"bitstamp_{symbol.replace('/', '')}_{timeframe}.csv"
            df.to_csv(filename)
            print(f"Saved {filename}")
