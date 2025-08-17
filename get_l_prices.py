import requests
import pandas as pd
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LIVE_DATA_DIR = os.path.join(BASE_DIR, "live_data")
STATUS_FILE = os.path.join(BASE_DIR, "pipeline_status.txt")

# Mapping: CCXT symbols -> CoinGecko IDs
COINS = {
    "BTC/USDT": "bitcoin",
    "ETH/USDT": "ethereum",
    "BNB/USDT": "binancecoin",
    "XRP/USDT": "ripple",
    "ADA/USDT": "cardano",
    "DOGE/USDT": "dogecoin",
    "SOL/USDT": "solana",
    "DOT/USDT": "polkadot",
    "MATIC/USDT": "matic-network",
    "LTC/USDT": "litecoin",
}

def log_status(message):
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    with open(STATUS_FILE, "a") as f:
        f.write(f"{timestamp} - {message}\n")
    print(f"{timestamp} - {message}")

def fetch_ohlcv(coin_id, days=1):
    """
    Fetch last 1 day hourly candles from CoinGecko.
    Returns list of OHLCV dicts.
    """
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": days, "interval": "hourly"}
    try:
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()

        ohlcv_data = []
        prices = data.get("prices", [])
        for i, (ts, price) in enumerate(prices):
            ts = datetime.utcfromtimestamp(ts / 1000).isoformat()
            ohlcv_data.append({
                "timestamp": ts,
                "open": price,   # CoinGecko only gives "prices" as points
                "high": price,
                "low": price,
                "close": price,
                "volume": data.get("total_volumes", [[0,0]])[i][1]
            })
        return ohlcv_data
    except Exception as e:
        log_status(f"Error fetching {coin_id}: {e}")
        return []

def save_live_data(data, filename):
    os.makedirs(LIVE_DATA_DIR, exist_ok=True)
    with open(filename, "a") as f:
        for entry in data:
            f.write(pd.Series(entry).to_json() + "\n")

def main():
    log_status("Starting live data fetch (CoinGecko)...")

    for symbol, coin_id in COINS.items():
        data = fetch_ohlcv(coin_id, days=1)
        if data:
            filename = os.path.join(LIVE_DATA_DIR, symbol.replace("/", "_") + "_live.jsonl")
            save_live_data(data, filename)
            log_status(f"Saved {len(data)} candles for {symbol}")
        else:
            log_status(f"No data saved for {symbol} (fetch failed)")

if __name__ == "__main__":
    main()