# fetch_model_and_data.py
import json
import torch
import numpy as np
from pathlib import Path

BASE_DIR = Path("/Users/anastangelis/Desktop/Crypto Predict")
DATA_DIR = BASE_DIR / "data"
LIVE_DATA_DIR = BASE_DIR / "live_data"
MODEL_DIR = BASE_DIR / "models"

WINDOW = 30  # must match your model training window (update if different)

# --- Use the SAME model as in training (PricePredictionModel from training script) ---
class PricePredictionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=1, hidden_size=50, num_layers=2, batch_first=True)
        self.fc = torch.nn.Linear(50, 1)

    def forward(self, x):
        # x should be (batch, seq_len, 1)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # predict from last timestep
        return out

# --- Loaders ---
def load_json_lines(file_path):
    with open(file_path, "r") as f:
        lines = f.read().strip().split("\n")
        return [json.loads(line) for line in lines]

def load_combined_prices(symbol, window=WINDOW):
    hist_file = DATA_DIR / f"{symbol}.json"
    live_file = LIVE_DATA_DIR / f"{symbol}_live.jsonl"

    prices = []
    if hist_file.exists():
        with open(hist_file, "r") as f:
            hist_data = json.load(f)
        hist_prices = [d["close"] for d in hist_data if "close" in d]
        prices.extend(hist_prices)

    if live_file.exists():
        live_data = load_json_lines(live_file)
        live_prices = [d.get("price", d.get("close")) for d in live_data if ("price" in d or "close" in d)]
        prices.extend(live_prices)

    if len(prices) < window:
        raise ValueError(f"Not enough combined data ({len(prices)}) to form {window}-point input sequence")

    return np.array(prices[-window:], dtype=np.float32)

def prepare_input_sequence(prices):
    # Normalize by max to keep values between 0-1
    max_price = prices.max() if prices.max() > 0 else 1.0
    norm_prices = prices / max_price
    # Correct shape: (batch=1, seq_len=WINDOW, features=1)
    tensor_seq = torch.tensor(norm_prices, dtype=torch.float32).unsqueeze(0).unsqueeze(2)
    # Debug: verify shape
    print(f"Prepared sequence shape: {tensor_seq.shape}")  # Should be torch.Size([1, WINDOW, 1])
    return tensor_seq, max_price

def load_model(symbol):
    model_path = MODEL_DIR / f"daily_model_{symbol}.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = PricePredictionModel()
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model

# --- Script entry point ---
if __name__ == "__main__":
    import sys
    symbol = sys.argv[1] if len(sys.argv) > 1 else "BTC_USDT"

    model = load_model(symbol)
    prices = load_combined_prices(symbol)
    seq, max_price = prepare_input_sequence(prices)

    # Save for downstream prediction
    torch.save(model.state_dict(), f"{symbol}_model_state.pth")
    np.save(f"{symbol}_latest_prices.npy", prices)
    torch.save(seq, f"{symbol}_latest_seq.pt")
    print(f"Model state and latest prices for {symbol} saved.")
