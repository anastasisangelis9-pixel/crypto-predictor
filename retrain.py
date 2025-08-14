import os
import json
import torch
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent
LIVE_DATA_DIR = BASE_DIR / "live_data"
HISTORICAL_DATA_DIR = BASE_DIR / "historical_data"
MODEL_DIR = BASE_DIR / "models"

WINDOW = 10           # LSTM sequence length
MIN_DAYS_TO_RETRAIN = 11  # Require at least a week’s worth of data

class PricePredictionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=1, hidden_size=50, num_layers=2, batch_first=True)
        self.fc = torch.nn.Linear(50, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def load_json_lines(file_path):
    if not file_path.exists():
        return []
    with open(file_path, "r") as f:
        lines = f.read().strip().split("\n")
        return [json.loads(line) for line in lines if line.strip()]

def append_to_historical(symbol, live_data):
    HISTORICAL_DATA_DIR.mkdir(exist_ok=True)
    historical_file = HISTORICAL_DATA_DIR / f"{symbol}_historical.jsonl"
    with open(historical_file, "a") as f:
        for entry in live_data:
            f.write(json.dumps(entry) + "\n")

def clear_live_file(live_file):
    # Truncate the file (keep it but make it empty)
    open(live_file, "w").close()

def train_model(symbol, live_data):
    closes = np.array([entry["close"] for entry in live_data], dtype=np.float32)
    if len(closes) < max(WINDOW + 1, MIN_DAYS_TO_RETRAIN):
        print(f"Not enough live data to train model for {symbol} "
              f"(have {len(closes)}, need at least {max(WINDOW + 1, MIN_DAYS_TO_RETRAIN)})")
        return False  # Indicate no training

    max_price = closes.max()
    norm_closes = closes / max_price

    X, y = [], []
    for i in range(len(norm_closes) - WINDOW):
        X.append(norm_closes[i:i+WINDOW])
        y.append(norm_closes[i+WINDOW])
    X = np.array(X)
    y = np.array(y)

    # Ensure shape is (batch, seq_len, 1)
    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(2)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PricePredictionModel().to(device)
    model_path = MODEL_DIR / f"{symbol}_model.pt"
    MODEL_DIR.mkdir(exist_ok=True)

    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded existing model for {symbol}")
    else:
        print(f"Creating new model for {symbol}")

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    X_tensor = X_tensor.to(device)
    y_tensor = y_tensor.to(device)

    epochs = 10
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

    torch.save(model.state_dict(), model_path)
    print(f"Model saved for {symbol} at {model_path}")
    return True  # Indicate successful training

def main():
    symbols = [f.stem.replace("_live", "") for f in LIVE_DATA_DIR.glob("*_live.jsonl")]
    if not symbols:
        print("No live data files found for retraining.")
        return

    for symbol in symbols:
        live_file = LIVE_DATA_DIR / f"{symbol}_live.jsonl"
        live_data = load_json_lines(live_file)
        if not live_data:
            print(f"No live data for {symbol}, skipping.")
            continue

        print(f"Processing live data for {symbol}...")

        # Try to train; if successful, archive the live data
        trained = train_model(symbol, live_data)
        if trained:
            append_to_historical(symbol, live_data)
            clear_live_file(live_file)
            print(f"Archived and cleared live data for {symbol}")
        else:
            print(f"Keeping live data for {symbol} (not enough to train yet).")

if __name__ == "__main__":
    main()
