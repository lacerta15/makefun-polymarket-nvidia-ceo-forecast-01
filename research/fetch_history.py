import json, time, urllib.request, urllib.error
from pathlib import Path

HERE = Path(__file__).resolve().parent
with open(HERE / "markets_union.json", encoding="utf-8") as f:
    markets = json.load(f)["markets"]

OUT_DIR = HERE / "history"
OUT_DIR.mkdir(exist_ok=True)

START_TS = 1794463200  # 2026-11-12T06:00:00Z
END_TS = 1794550200    # 2026-11-13T06:10:00Z (buffer)

def fetch(token, start, end, fidelity=1, retries=4):
    url = f"https://clob.polymarket.com/prices-history?market={token}&startTs={start}&endTs={end}&fidelity={fidelity}"
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "research-script/1.0"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise last_err

results = {}
errors = {}
for mid, info in markets.items():
    token = info["yes_token"]
    try:
        data = fetch(token, START_TS, END_TS)
        hist = data.get("history", [])
        results[mid] = hist
        print(mid, info["name"], "npoints=", len(hist))
    except Exception as e:
        errors[mid] = str(e)
        print("ERROR", mid, info["name"], e)
    time.sleep(0.3)

with open(OUT_DIR / "realized.json", "w", encoding="utf-8") as f:
    json.dump(results, f)

if errors:
    with open(OUT_DIR / "errors.json", "w", encoding="utf-8") as f:
        json.dump(errors, f, indent=2)
    print("ERRORS:", errors)
else:
    print("All fetched OK, no errors.")
