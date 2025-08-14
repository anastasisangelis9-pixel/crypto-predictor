import json

with open("data/DOGE_USDT.json") as f:
    data = json.load(f)

print(data[0])
