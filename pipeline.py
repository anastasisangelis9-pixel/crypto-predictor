# start pipeline
# python3 pipeline.py > pipeline.log 2>&1 &

# stop pipeline
# touch stop_pipeline.flag

# restart pipeline
# rm stop_pipeline.flag


# pipeline.py (for GitHub Actions)
import subprocess
from datetime import datetime

COINS = [
    "BTC_USDT", "ETH_USDT", "BNB_USDT", "XRP_USDT", "ADA_USDT",
    "DOGE_USDT", "SOL_USDT", "DOT_USDT", "MATIC_USDT", "LTC_USDT",
]

def run_get_live_prices():
    print(f"[{datetime.utcnow()}] Running get_l_prices.py...")
    subprocess.run(["python3", "get_l_prices.py"], check=True)

def run_predictions():
    for coin in COINS:
        print(f"[{datetime.utcnow()}] Running predictions for {coin}...")
        subprocess.run(["python3", "predict_3.py", coin], check=True)

def run_retrain():
    print(f"[{datetime.utcnow()}] Running retrain.py...")
    subprocess.run(["python3", "retrain.py"], check=True)

def main():
    print(f"=== Pipeline run started at {datetime.utcnow()} ===")

    run_get_live_prices()
    run_predictions()

    # Optional: retrain once a week (Sunday 00:00 UTC)
    if datetime.utcnow().weekday() == 6 and datetime.utcnow().hour == 0:
        run_retrain()

    print(f"=== Pipeline run finished at {datetime.utcnow()} ===")

if __name__ == "__main__":
    main()