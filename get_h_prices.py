import ccxt
import pandas as pd
import os
import time

def fetch_historical_data(symbol="BTC/USDT", timeframe="1d", limit=1000, max_retries=5):
    exchange = ccxt.binance()
    all_candles = []
    
    # Binance only has data since ~2017, so let's start as far back as possible (or ~5 years ago)
    start_date = int((pd.Timestamp.now() - pd.Timedelta(days=5*365)).timestamp() * 1000)
    since = start_date

    for attempt in range(max_retries):
        try:
            while True:
                candles = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
                if not candles:
                    break
                all_candles.extend(candles)
                since = candles[-1][0] + 1  # move to next timestamp
                time.sleep(exchange.rateLimit / 1000)  # rate limit safety
            break
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            print(f"Error fetching {symbol}: {e}, retrying...")
            time.sleep(5)

    df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df

def fetch_all_top_coins():
    coins = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "XRP/USDT", "ADA/USDT",
             "DOGE/USDT", "SOL/USDT", "DOT/USDT", "MATIC/USDT", "LTC/USDT"]

    os.makedirs("data", exist_ok=True)

    for coin in coins:
        print(f"Fetching {coin} data...")
        df = fetch_historical_data(coin)
        coin_file = coin.replace("/", "_") + ".json"
        df.to_json(os.path.join("data", coin_file), orient="records", date_format="iso")
        print(f"Saved {coin} data to data/{coin_file}")

if __name__ == "__main__":
    fetch_all_top_coins()
