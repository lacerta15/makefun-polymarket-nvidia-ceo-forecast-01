"""Validate main.py's .out files against the contest's exact shape requirements."""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIX = ROOT / "fixtures" / "input"
OUT = ROOT / "out1"

CASES = ["p00", "p01", "p02", "p03", "p04", "p05"]

errors = []

for cid in CASES:
    with open(FIX / f"{cid}.in", encoding="utf-8") as f:
        case = json.load(f)
    out_path = OUT / f"{cid}.out"
    size = out_path.stat().st_size
    if size > 1_048_576:
        errors.append(f"{cid}: output too large ({size} bytes)")
    with open(out_path, encoding="utf-8") as f:
        text = f.read()
    try:
        out = json.loads(text)
    except Exception as e:
        errors.append(f"{cid}: not valid JSON: {e}")
        continue

    if set(out.keys()) != {"forecasts"}:
        errors.append(f"{cid}: top-level keys {list(out.keys())}, expected only 'forecasts'")
        continue

    forecasts = out["forecasts"]
    case_market_ids = {m["id"] for m in case["markets"]}
    out_ids = [f["id"] for f in forecasts]
    out_id_set = set(out_ids)

    if len(out_ids) != len(out_id_set):
        errors.append(f"{cid}: duplicate market ids in output")
    if out_id_set != case_market_ids:
        missing = case_market_ids - out_id_set
        extra = out_id_set - case_market_ids
        errors.append(f"{cid}: id mismatch. missing={missing} extra={extra}")

    horizon = case["horizon_minutes"]
    for fc in forecasts:
        if set(fc.keys()) != {"id", "q"}:
            errors.append(f"{cid}/{fc.get('id')}: forecast keys {list(fc.keys())}, expected id,q")
            continue
        q = fc["q"]
        if len(q) != horizon:
            errors.append(f"{cid}/{fc['id']}: {len(q)} rows, expected {horizon}")
            continue
        for h, row in enumerate(q):
            if len(row) != 5:
                errors.append(f"{cid}/{fc['id']} row {h}: {len(row)} values, expected 5")
                break
            for v in row:
                if not isinstance(v, (int, float)):
                    errors.append(f"{cid}/{fc['id']} row {h}: non-numeric value {v}")
                elif not (0.0 <= v <= 1.0):
                    errors.append(f"{cid}/{fc['id']} row {h}: value {v} out of [0,1]")
                elif v != v or v in (float('inf'), float('-inf')):
                    errors.append(f"{cid}/{fc['id']} row {h}: non-finite value {v}")
            for j in range(1, 5):
                if row[j] < row[j - 1]:
                    errors.append(f"{cid}/{fc['id']} row {h}: not non-decreasing: {row}")
                    break

print(f"Checked {len(CASES)} cases.")
if errors:
    print(f"FOUND {len(errors)} ERRORS:")
    for e in errors[:50]:
        print(" -", e)
    sys.exit(1)
else:
    print("ALL CHECKS PASSED: valid JSON, correct keys, correct market-id coverage,")
    print("correct row/column counts, values in [0,1], finite, non-decreasing per row.")
