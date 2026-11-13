"""Backtest harness: loads practice cases + fetched realized history, evaluates model variants."""
import json, math, bisect, datetime as dt
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIX = ROOT / "fixtures" / "input"

CASES = ["p00", "p01", "p02", "p03", "p04", "p05"]
QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.644854, -0.67449, 0.0, 0.67449, 1.644854]

def iso_to_ts(s):
    # "2026-11-12T06:00:00.000Z"
    s2 = s.replace("Z", "+00:00")
    d = dt.datetime.fromisoformat(s2)
    return int(d.timestamp())

def load_cases():
    cases = {}
    for cid in CASES:
        with open(FIX / f"{cid}.in", encoding="utf-8") as f:
            cases[cid] = json.load(f)
    return cases

def load_history():
    with open(HERE / "history" / "realized.json", encoding="utf-8") as f:
        raw = json.load(f)
    hist = {}
    for mid, points in raw.items():
        pts = sorted((p["t"], p["p"]) for p in points)
        hist[mid] = pts
    return hist

def grid_price(points_ts, points_p, minute_ts):
    # last point with t <= minute_ts
    idx = bisect.bisect_right(points_ts, minute_ts) - 1
    if idx < 0:
        return None
    return points_p[idx]

def build_realized(cases, hist):
    """Returns dict cid -> mid -> list of 240 realized prices for minutes at+1..at+240."""
    realized = {}
    missing = []
    for cid, c in cases.items():
        at_ts = iso_to_ts(c["at"])
        realized[cid] = {}
        for m in c["markets"]:
            mid = m["id"]
            if mid not in hist:
                missing.append((cid, mid))
                continue
            pts = hist[mid]
            ts_list = [p[0] for p in pts]
            p_list = [p[1] for p in pts]
            row = []
            last = None
            for h in range(1, 241):
                minute = at_ts + h * 60
                v = grid_price(ts_list, p_list, minute)
                if v is None:
                    v = last if last is not None else m["prices"][-1]
                row.append(v)
                last = v
            realized[cid][mid] = row
    if missing:
        print("MISSING realized data for:", missing)
    return realized

def pinball(y, q, tau):
    d = y - q
    return 100.0 * max(tau * d, (tau - 1) * d)

def score_forecast(realized_row, q_rows):
    """realized_row: list of 240 actuals. q_rows: list of 240 lists of 5 quantiles."""
    total = 0.0
    n = 0
    for h in range(240):
        y = realized_row[h]
        qs = q_rows[h]
        for j, tau in enumerate(QUANTILES):
            total += pinball(y, qs[j], tau)
            n += 1
    return total, n

def evaluate(model_fn, cases, realized, verbose=False):
    """model_fn(market_dict, horizon) -> list of 240 rows of 5 quantiles."""
    grand_total = 0.0
    grand_n = 0
    per_case = {}
    for cid, c in cases.items():
        horizon = c["horizon_minutes"]
        case_total = 0.0
        case_n = 0
        for m in c["markets"]:
            mid = m["id"]
            if mid not in realized[cid]:
                continue
            q_rows = model_fn(m, horizon)
            t, n = score_forecast(realized[cid][mid], q_rows)
            case_total += t
            case_n += n
        per_case[cid] = case_total / case_n
        grand_total += case_total
        grand_n += case_n
    mean_loss = grand_total / grand_n
    if verbose:
        for cid in CASES:
            print(f"  {cid}: {per_case[cid]:.6f}")
    return mean_loss, per_case

if __name__ == "__main__":
    cases = load_cases()
    hist = load_history()
    realized = build_realized(cases, hist)
    print("Loaded", len(cases), "cases,", len(hist), "markets with history.")
    nmarkets_total = sum(len(realized[c]) for c in realized)
    print("Total (case,market) pairs with realized data:", nmarkets_total)
