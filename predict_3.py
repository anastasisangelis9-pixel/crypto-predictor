from datetime import datetime
from pathlib import Path
from predict_2 import predict_next_price, predict_multiple_steps
from predict_1 import load_model, load_combined_prices, prepare_input_sequence
import torch

# Coin name mapping
COIN_NAMES = {
    "BTC": "Bitcoin", "ETH": "Ethereum", "BNB": "Binance Coin",
    "XRP": "Ripple", "ADA": "Cardano", "DOGE": "Dogecoin",
    "SOL": "Solana", "DOT": "Polkadot", "MATIC": "Polygon", "LTC": "Litecoin"
}

BASE_DIR = Path("/Users/anastangelis/Desktop/Crypto Predict")
PREDICTIONS_DIR = BASE_DIR / "predictions"


def save_predictions_txt(symbol, short_term_pred, long_term_preds):
    """Save predictions to a human-readable text file."""
    PREDICTIONS_DIR.mkdir(exist_ok=True)
    now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    file_path = PREDICTIONS_DIR / f"{symbol}_predictions.txt"

    coin_name = COIN_NAMES.get(symbol.split("_")[0], symbol)
    last_known_price = long_term_preds[0] if long_term_preds else short_term_pred

    def fmt(val, pct=False):
        sign = "+" if val > 0 else ""
        return f"{sign}{val:.2f}%" if pct else f"{sign}{val:.6f}"

    with open(file_path, "w") as f:
        f.write(f"Price predictions for {coin_name} ({symbol}) at {now_str}\n\n")
        f.write(f"Last known (input) price: {last_known_price:.6f}\n\n")

        # Short-term prediction (1 hour ahead)
        f.write(f"Short-term prediction (next hour): {short_term_pred:.6f}\n")
        rise = short_term_pred - last_known_price
        rise_pct = (rise / last_known_price) * 100 if last_known_price > 0 else 0
        f.write(f"Expected change next hour: {fmt(rise)} ({fmt(rise_pct, pct=True)})\n")

        # Plain-language summary for short-term
        if rise > 0:
            f.write(f"In simple terms: Price likely to RISE by about {abs(rise_pct):.2f}% in the next hour.\n\n")
        elif rise < 0:
            f.write(f"In simple terms: Price likely to FALL by about {abs(rise_pct):.2f}% in the next hour.\n\n")
        else:
            f.write(f"In simple terms: Price expected to stay FLAT in the next hour.\n\n")

        # Long-term (7-day) prediction
        f.write(f"Long-term predictions for the next {len(long_term_preds)} days:\n")
        for i, price in enumerate(long_term_preds, 1):
            change = price - last_known_price
            pct = (change / last_known_price) * 100 if last_known_price > 0 else 0
            f.write(f"  Day {i}: {price:.6f} (Change: {fmt(change)}, {fmt(pct, pct=True)})\n")

        # Overall trend (plain language)
        if long_term_preds:
            first = last_known_price
            last = long_term_preds[-1]
            change = last - first
            change_pct = (change / first) * 100 if first > 0 else 0
            if change > 0:
                f.write(f"\nOverall trend: Price expected to RISE over {len(long_term_preds)} days, "
                        f"from {first:.6f} to {last:.6f} (Gain: {fmt(change)}, {fmt(change_pct, pct=True)}).\n")
                f.write("In plain words: Market looks bullish — expect a gradual climb.\n")
            elif change < 0:
                f.write(f"\nOverall trend: Price expected to FALL over {len(long_term_preds)} days, "
                        f"from {first:.6f} to {last:.6f} (Loss: {fmt(change)}, {fmt(change_pct, pct=True)}).\n")
                f.write("In plain words: Market looks bearish — expect a gradual drop.\n")
            else:
                f.write(f"\nOverall trend: Price expected to stay STABLE at around {first:.6f}.\n")
                f.write("In plain words: Little to no change expected.\n")

    print(f"Predictions saved to {file_path}")


def main(symbol):
    print(f"Loading model for {symbol}")
    model = load_model(symbol)

    print(f"Loading price data for {symbol}")
    prices = load_combined_prices(symbol)

    # Normalize input using min/max scaling
    min_price = float(prices.min())
    max_price = float(prices.max())
    price_range = max_price - min_price if max_price > min_price else 1.0
    norm_prices = (prices - min_price) / price_range
    seq = torch.tensor(norm_prices, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # Short-term (1-hour) prediction
    short_term_pred = predict_next_price(model, seq, min_price, price_range, last_price=prices[-1])

    # Long-term (7-day) predictions
    long_term_preds = predict_multiple_steps(model, prices, steps=7)


    # Save to text file
    save_predictions_txt(symbol, short_term_pred, long_term_preds)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python predict_process.py <symbol_like_ADA_USDT>")
        sys.exit(1)
    main(sys.argv[1])
