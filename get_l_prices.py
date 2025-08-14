import ccxt
import pandas as pd
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LIVE_DATA_DIR = os.path.join(BASE_DIR, "live_data")
STATUS_FILE = os.path.join(BASE_DIR, "pipeline_status.txt")

COINS = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "XRP/USDT", "ADA/USDT",
         "DOGE/USDT", "SOL/USDT", "DOT/USDT", "MATIC/USDT", "LTC/USDT"]

def log_status(message):
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    with open(STATUS_FILE, "a") as f:
        f.write(f"{timestamp} - {message}\n")
    print(f"{timestamp} - {message}")

def fetch_ohlcv(symbol, timeframe="30m", limit=2):
    exchange = ccxt.binance()
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        data = []
        for candle in ohlcv:
            ts = datetime.utcfromtimestamp(candle[0] / 1000).isoformat()
            data.append({
                "timestamp": ts,
                "open": candle[1],
                "high": candle[2],
                "low": candle[3],
                "close": candle[4],
                "volume": candle[5],
            })
        return data
    except Exception as e:
        log_status(f"Error fetching {symbol}: {e}")
        return []

def save_live_data(data, filename):
    os.makedirs(LIVE_DATA_DIR, exist_ok=True)
    with open(filename, "a") as f:
        for entry in data:
            f.write(pd.Series(entry).to_json() + "\n")

def main():
    log_status("Starting live data fetch...")

    for coin in COINS:
        data = fetch_ohlcv(coin, "30m", limit=2)
        if data:
            filename = os.path.join(LIVE_DATA_DIR, coin.replace("/", "_") + "_live.jsonl")
            save_live_data(data, filename)
            log_status(f"Saved {len(data)} candles for {coin}")
        else:
            log_status(f"No data saved for {coin} (fetch failed)")

if __name__ == "__main__":
    main()



# stop running
# pkill -f get_l_prices.py


# to run
# python3 get_l_prices.py

#to run in background 
# nohup python3 get_l_prices.py &
