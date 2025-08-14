# benchmark.py
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

SEQ_LEN = 10

def benchmark(symbol):
    data_file = DATA_DIR / f"{symbol}.json"
    if not data_file.exists():
        print(f"No historical data for {symbol}")
        return

    # Load data
    with open(data_file, "r") as f:
        hist_data = json.load(f)
    closes = np.array([x["close"] for x in hist_data], dtype=np.float32)

    if len(closes) < 100:
        print(f"Not enough data for benchmarking {symbol}")
        return

    # Naive predictions (last value from each window)
    preds, actuals = [], []
    for i in range(SEQ_LEN, len(closes)):
        preds.append(closes[i-1])  # Last known price
        actuals.append(closes[i])

    preds = np.array(preds)
    actuals = np.array(actuals)

    mae = mean_absolute_error(actuals, preds)
    rmse = mean_squared_error(actuals, preds, squared=False)
    mape = np.mean(np.abs((actuals - preds) / actuals)) * 100

    print(f"\nBenchmark (Naïve) Results for {symbol}:")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.2f}% error")

if __name__ == "__main__":
    symbols = [f.stem for f in DATA_DIR.glob("*.json")]
    for symbol in symbols:
        benchmark(symbol)
