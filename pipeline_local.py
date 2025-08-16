import os
import subprocess
import time
import threading
from datetime import datetime, timedelta

COINS = [
    "BTC_USDT", "ETH_USDT", "BNB_USDT", "XRP_USDT", "ADA_USDT",
    "DOGE_USDT", "SOL_USDT", "DOT_USDT", "MATIC_USDT", "LTC_USDT",
]

PREDICTION_INTERVAL = 3600  # 1 hour
RETRAIN_INTERVAL = 11 * 24 * 3600  # 11 days

stop_flag = False
STOP_FILE = "stop_pipeline.flag"

def listen_for_stop_file():
    global stop_flag
    while not stop_flag:
        if os.path.exists(STOP_FILE):
            print("Stop file detected. Stopping pipeline gracefully...")
            stop_flag = True
            # Optionally delete the stop file so next runs start fresh
            os.remove(STOP_FILE)
        time.sleep(2)  # Check every 2 seconds

def run_get_live_prices():
    print(f"[{datetime.utcnow()}] Running get_live_prices.py...")
    subprocess.run(["python3", "get_l_prices.py"], check=True)

def run_predictions():
    for coin in COINS:
        print(f"[{datetime.utcnow()}] Running predictions for {coin}...")
        subprocess.run(["python3", "predict_3.py", coin], check=True)

def run_retrain():
    print(f"[{datetime.utcnow()}] Running retrain.py...")
    subprocess.run(["python3", "retrain.py"], check=True)

def main():
    global stop_flag
    last_retrain_time = datetime.utcnow() - timedelta(seconds=RETRAIN_INTERVAL)

    # Start listener thread
    threading.Thread(target=listen_for_stop_file, daemon=True).start()

    while not stop_flag:
        start_time = datetime.utcnow()

        # Step 1: Update live prices
        run_get_live_prices()

        # Step 2: Run predictions for all coins
        run_predictions()

        # Step 3: Check if retraining is due
        if (start_time - last_retrain_time).total_seconds() >= RETRAIN_INTERVAL:
            run_retrain()
            last_retrain_time = start_time

        # Step 4: Sleep until the next hourly cycle (but exit if stop_flag is set)
        elapsed = (datetime.utcnow() - start_time).total_seconds()
        sleep_time = max(0, PREDICTION_INTERVAL - elapsed)
        for _ in range(int(sleep_time)):
            if stop_flag:
                break
            time.sleep(1)

    print("Scheduler stopped.")

if __name__ == "__main__":
    main()