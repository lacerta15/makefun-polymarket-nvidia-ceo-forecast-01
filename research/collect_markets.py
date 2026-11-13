import json, sys
from pathlib import Path

FIX = Path(__file__).resolve().parent.parent / "fixtures" / "input"

cases = {}
markets = {}  # id -> {"yes_token":..., "name":..., "slug":..., "start":...}
for cid in ["p00","p01","p02","p03","p04","p05"]:
    with open(FIX / f"{cid}.in", encoding="utf-8") as f:
        d = json.load(f)
    cases[cid] = d
    for m in d["markets"]:
        markets.setdefault(m["id"], {
            "yes_token": m["yes_token"],
            "name": m["name"],
            "slug": m["slug"],
            "start": m["start"],
        })

print("num cases", len(cases))
print("union markets", len(markets))
for cid, d in cases.items():
    print(cid, d["at"], "nmarkets=", len(d["markets"]))

out = {"markets": markets}
with open(Path(__file__).resolve().parent / "markets_union.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
print("wrote markets_union.json")
