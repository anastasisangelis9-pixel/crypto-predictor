# backtest.py
import json
import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "accuracy_reports"
REPORTS_DIR.mkdir(exist_ok=True)  # create folder if it doesn't exist

SEQ_LEN = 10  # Must match your training sequence length

# Timestamped report filename
timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
REPORT_FILE = REPORTS_DIR / f"accuracy_report_{timestamp}.json"


class PricePredictionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=1, hidden_size=50, num_layers=2, batch_first=True)
        self.fc = torch.nn.Linear(50, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def load_model(symbol):
    model = PricePredictionModel()
    model_path = MODEL_DIR / f"daily_model_{symbol}.pth"
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def backtest(symbol, forecast_horizon=1):
    data_file = DATA_DIR / f"{symbol}.json"
    if not data_file.exists():
        return {"warning": f"No historical data for {symbol}"}, None

    with open(data_file, "r") as f:
        hist_data = json.load(f)

    filtered_data = [x for x in hist_data if "close" in x]
    missing_count = len(hist_data) - len(filtered_data)
    warning_msg = f"Skipped {missing_count} entries without 'close'" if missing_count > 0 else ""

    closes = np.array([x["close"] for x in filtered_data], dtype=np.float32)
    if len(closes) < 100 + forecast_horizon:
        return {"warning": f"Not enough data for backtesting {symbol}"}, None

    max_price = closes.max()
    norm_closes = closes / max_price

    X, y = [], []
    for i in range(len(norm_closes) - SEQ_LEN - forecast_horizon + 1):
        X.append(norm_closes[i:i+SEQ_LEN])
        y.append(norm_closes[i + SEQ_LEN + forecast_horizon - 1])
    X, y = np.array(X), np.array(y)

    split_idx = int(len(X) * 0.8)
    X_test, y_test = X[split_idx:], y[split_idx:]

    model = load_model(symbol)
    with torch.no_grad():
        preds = model(torch.tensor(X_test).unsqueeze(2)).squeeze().numpy()

    preds *= max_price
    y_test *= max_price

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))  # ✅ FIXED HERE
    mape = np.mean(np.abs((y_test - preds) / y_test)) * 100
    accuracy = max(0, 100 - mape)

    # Determine textual analysis
    if mape > 100:
        analysis = f"Model unusable. Accuracy ~ {accuracy:.1f}%"
    elif mape > 75:
        analysis = f"Very poor accuracy ~ {accuracy:.1f}%"
    elif mape > 50:
        analysis = f"Highly inaccurate ~ {accuracy:.1f}%"
    elif mape > 35:
        analysis = f"Low accuracy ~ {accuracy:.1f}%"
    elif mape > 20:
        analysis = f"Moderately accurate ~ {accuracy:.1f}%"
    elif mape > 10:
        analysis = f"Reasonably accurate ~ {accuracy:.1f}%"
    elif mape > 5:
        analysis = f"Very accurate ~ {accuracy:.1f}%"
    elif mape > 1:
        analysis = f"Extremely accurate ~ {accuracy:.1f}%"
    else:
        analysis = f"Almost perfect ~ {accuracy:.1f}%"

    result = {
        "forecast_horizon": forecast_horizon,
        "warning": warning_msg,
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape),
        "accuracy": float(accuracy),
        "analysis": analysis
    }

    return result, accuracy


if __name__ == "__main__":
    symbols = [f.stem for f in DATA_DIR.glob("*.json")]
    report_data = {
        "timestamp": timestamp,
        "results": {},
        "overall_accuracy": None
    }

    forecast_horizons = [1, 7, 30]
    accuracies = []

    for symbol in symbols:
        report_data["results"][symbol] = {}
        for horizon in forecast_horizons:
            result, accuracy = backtest(symbol, forecast_horizon=horizon)
            report_data["results"][symbol][str(horizon)] = result
            if accuracy is not None:
                accuracies.append(accuracy)

    if accuracies:
        report_data["overall_accuracy"] = float(np.mean(accuracies))

    with open(REPORT_FILE, "w") as f:
        json.dump(report_data, f, indent=4)

    print(f"Accuracy report saved to {REPORT_FILE}")