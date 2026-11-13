"""Score the literal main.py .out files (out1/) against the fetched realized ground truth."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research.backtest import load_cases, load_history, build_realized, score_forecast, QUANTILES, CASES

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out1"

cases = load_cases()
hist = load_history()
realized = build_realized(cases, hist)

grand_total = 0.0
grand_n = 0
for cid in CASES:
    with open(OUT / f"{cid}.out", encoding="utf-8") as f:
        out = json.load(f)
    by_id = {fc["id"]: fc["q"] for fc in out["forecasts"]}
    case_total = 0.0
    case_n = 0
    for mid, q_rows in by_id.items():
        if mid not in realized[cid]:
            continue
        t, n = score_forecast(realized[cid][mid], q_rows)
        case_total += t
        case_n += n
    print(f"{cid}: loss={case_total/case_n:.6f}  (n={case_n})")
    grand_total += case_total
    grand_n += case_n

print()
print(f"main.py mean pinball loss over all 6 practice cases: {grand_total/grand_n:.6f}")
print(f"house baseline (reference):                          0.363589")
improvement = (0.363589 - grand_total/grand_n) / 0.363589 * 100
print(f"Improvement over house baseline: {improvement:.2f}%")
