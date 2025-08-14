# backtest.py
import json
import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
REPORT_FILE = BASE_DIR / "accuracy_report.txt"  # Output file

SEQ_LEN = 10  # Must match the training sequence length

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
        return f"No historical data for {symbol}\n", None

    # Load price history
    with open(data_file, "r") as f:
        hist_data = json.load(f)

    # Filter entries that have the "close" key
    filtered_data = [x for x in hist_data if "close" in x]
    missing_count = len(hist_data) - len(filtered_data)
    warning_msg = ""
    if missing_count > 0:
        warning_msg = f"Warning: Skipped {missing_count} entries without 'close' for {symbol}\n"

    closes = np.array([x["close"] for x in filtered_data], dtype=np.float32)

    if len(closes) < 100 + forecast_horizon:
        return f"{warning_msg}Not enough data for backtesting {symbol} with forecast horizon {forecast_horizon}\n", None

    # Normalize
    max_price = closes.max()
    norm_closes = closes / max_price

    # Prepare sequences with forecast horizon
    X, y = [], []
    for i in range(len(norm_closes) - SEQ_LEN - forecast_horizon + 1):
        X.append(norm_closes[i:i+SEQ_LEN])
        y.append(norm_closes[i + SEQ_LEN + forecast_horizon - 1])
    X, y = np.array(X), np.array(y)

    # Use last 20% as "test set"
    split_idx = int(len(X) * 0.8)
    X_test, y_test = X[split_idx:], y[split_idx:]

    model = load_model(symbol)
    with torch.no_grad():
        preds = model(torch.tensor(X_test).unsqueeze(2)).squeeze().numpy()

    # Denormalize
    preds *= max_price
    y_test *= max_price

    mae = mean_absolute_error(y_test, preds)
    rmse = root_mean_squared_error(y_test, preds)
    mape = np.mean(np.abs((y_test - preds) / y_test)) * 100
    accuracy = max(0, 100 - mape)  # Avoid negative accuracy

    # Build the output text
    result = f"\nBacktest Results for {symbol} (Forecast Horizon: {forecast_horizon} days):\n"
    result += warning_msg
    result += f"MAE: {mae:.4f}\n"
    result += f"RMSE: {rmse:.4f}\n"
    result += f"MAPE: {mape:.2f}% error\n"

    result += "Analysis:\n"
    result += f"- MAE of {mae:.4f} means the predictions differ from actual prices by this amount on average.\n"
    result += f"- RMSE of {rmse:.4f} shows error magnitude (larger errors penalized more).\n"
    result += f"- MAPE of {mape:.2f}% shows the average percentage error relative to price.\n"

    # Append human-readable assessment based on thresholds
    if mape > 100:
        result += f"- The model is unusable. Accuracy ~ {accuracy:.1f}%.\n"
    elif mape > 75:
        result += f"- The model is very poor. Accuracy ~ {accuracy:.1f}%.\n"
    elif mape > 50:
        result += f"- The model is highly inaccurate. Accuracy ~ {accuracy:.1f}%.\n"
    elif mape > 35:
        result += f"- The model is low accuracy. Accuracy ~ {accuracy:.1f}%.\n"
    elif mape > 20:
        result += f"- The model is moderately accurate. Accuracy ~ {accuracy:.1f}%.\n"
    elif mape > 10:
        result += f"- The model is reasonably accurate. Accuracy ~ {accuracy:.1f}%.\n"
    elif mape > 5:
        result += f"- The model is very accurate. Accuracy ~ {accuracy:.1f}%.\n"
    elif mape > 1:
        result += f"- The model is extremely accurate. Accuracy ~ {accuracy:.1f}%.\n"
    else:
        result += f"- The model is almost perfect. Accuracy ~ {accuracy:.1f}%.\n"

    result += "\n" + "="*60 + "\n"
    return result, accuracy


if __name__ == "__main__":
    symbols = [f.stem for f in DATA_DIR.glob("*.json")]
    full_report = "CRYPTO PREDICTION ACCURACY REPORT\n" + "="*60 + "\n"

    forecast_horizons = [1, 7, 30]
    accuracies = []  # To collect accuracy scores

    for symbol in symbols:
        for horizon in forecast_horizons:
            result, accuracy = backtest(symbol, forecast_horizon=horizon)
            full_report += result
            if accuracy is not None:
                accuracies.append(accuracy)

    # Compute overall average accuracy
    if accuracies:
        avg_accuracy = np.mean(accuracies)
        full_report += f"\nOVERALL AVERAGE MODEL ACCURACY: {avg_accuracy:.2f}%\n"
        full_report += "="*60 + "\n"
    else:
        full_report += "\nNo valid results to compute overall accuracy.\n" + "="*60 + "\n"

    # Save to file
    with open(REPORT_FILE, "w") as f:
        f.write(full_report)

    print(f"Accuracy report saved to {REPORT_FILE}")
