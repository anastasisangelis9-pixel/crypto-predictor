import torch
import numpy as np
from pathlib import Path

WINDOW = 60  # Increased window size from 30 to 60

class LSTMModel(torch.nn.Module):
    def __init__(self, window=WINDOW, input_size=1, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last timestep
        out = self.fc(out)
        return out.squeeze(1)

def ensure_seq_shape(seq):
    """
    Ensure the sequence is shaped (batch, seq_len, 1) for the LSTM.
    Handles cases where seq is (seq_len,), (1, seq_len), or (1, 1, seq_len).
    """
    if isinstance(seq, np.ndarray):
        seq = torch.tensor(seq, dtype=torch.float32)

    if seq.dim() == 1:  # (seq_len,) -> add batch + feature
        seq = seq.unsqueeze(0).unsqueeze(2)
    elif seq.dim() == 2:  # (1, seq_len) -> add feature dim
        seq = seq.unsqueeze(2)
    elif seq.shape[1] == 1 and seq.shape[-1] != 1:  # (1,1,seq_len) -> (1,seq_len,1)
        seq = seq.transpose(1, 2)

    return seq

def predict_next_price(model, seq, min_price, price_range, last_price=None, blend=0.4):
    # Fix the shape before inference
    seq = ensure_seq_shape(seq)

    with torch.no_grad():
        pred_norm = model(seq).item()
    pred_price = pred_norm * price_range + min_price

    # Smooth prediction
    if last_price is not None:
        max_increase = last_price * 1.03
        min_decrease = last_price * 0.97
        pred_price = max(min(pred_price, max_increase), min_decrease)
        pred_price = pred_price * (1 - blend) + last_price * blend

    return float(pred_price)

def predict_multiple_steps(model, prices, steps=7, blend=0.4):
    preds = []
    seq = prices.copy()
    last_price = seq[-1]

    for _ in range(steps):
        current_min = seq[-WINDOW:].min()
        current_max = seq[-WINDOW:].max()
        current_range = max(current_max - current_min, 1.0)

        norm_seq = (seq[-WINDOW:] - current_min) / current_range
        tensor_seq = torch.tensor(norm_seq, dtype=torch.float32).unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1)

        with torch.no_grad():
            pred_norm = model(tensor_seq).item()
        pred_price = pred_norm * current_range + current_min

        # Constrain change and smooth
        max_increase = last_price * 1.05
        min_decrease = last_price * 0.95
        pred_price = max(min(pred_price, max_increase), min_decrease)
        pred_price = pred_price * (1 - blend) + last_price * blend

        preds.append(pred_price)
        seq = np.append(seq, pred_price)
        last_price = pred_price

    return preds

def run_predictions(symbol):
    prices = np.load(f"{symbol}_latest_prices.npy")
    seq = torch.load(f"{symbol}_latest_seq.pt")

    model = LSTMModel(window=WINDOW)
    model_path = f"/Users/anastangelis/Desktop/Crypto Predict/models/daily_model_{symbol}.pth"
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    min_price = prices.min()
    max_price = prices.max()
    price_range = max_price - min_price if max_price > min_price else 1.0

    # Run both short-term and multi-step forecasts
    next_price = predict_next_price(model, seq, min_price, price_range, last_price=prices[-1])
    future_prices = predict_multiple_steps(model, prices, steps=7)

    print(f"\nPredictions for {symbol}:")
    print(f"Current price: {prices[-1]:.2f}")
    print(f"Next predicted price: {next_price:.2f}")
    print(f"7-day forecast: {', '.join([f'{p:.2f}' for p in future_prices])}")
    print(f"Final 7-day price: {future_prices[-1]:.2f}")
