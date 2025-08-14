import os
import json
import torch
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

WINDOW = 10  # sequence length for LSTM
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    # Load entire JSON file as a list of dicts
    with open(file_path, "r") as f:
        data = json.load(f)  # <-- changed here
    return data

def normalize_closes(closes, clip_outliers=True):
    if clip_outliers:
        lower = np.percentile(closes, 1)
        upper = np.percentile(closes, 99)
        closes = np.clip(closes, lower, upper)
    min_c = closes.min()
    max_c = closes.max()
    norm = (closes - min_c) / (max_c - min_c + 1e-8)
    return norm, min_c, max_c

def train_model(symbol, data):
    closes = np.array([entry["close"] for entry in data if "close" in entry], dtype=np.float32)
    if len(closes) < WINDOW + 1:
        print(f"Not enough data to train for {symbol}")
        return

    norm_closes, min_c, max_c = normalize_closes(closes)

    X, y = [], []
    for i in range(len(norm_closes) - WINDOW):
        X.append(norm_closes[i:i+WINDOW])
        y.append(norm_closes[i+WINDOW])
    X = np.array(X)
    y = np.array(y)

    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(2).to(device)  # (batch, seq, 1)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(device)  # (batch, 1)

    model = PricePredictionModel().to(device)
    MODEL_DIR.mkdir(exist_ok=True)  # Ensure models directory exists

    model_path = MODEL_DIR / f"daily_model_{symbol}.pth"
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded existing model for {symbol}")
    else:
        print(f"Creating new model for {symbol}")

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)


    epochs = 20
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        print(f"Epoch {epoch+1}/{epochs} Loss: {loss.item():.6f}")

    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")

def main():
    json_files = list(DATA_DIR.glob("*.json"))
    if not json_files:
        print("No data files found for training.")
        return

    for file_path in json_files:
        symbol = file_path.stem
        data = load_json_lines(file_path)
        if not data:
            print(f"No data for {symbol}, skipping.")
            continue

        print(f"Training for {symbol}...")
        train_model(symbol, data)

if __name__ == "__main__":
    main()
