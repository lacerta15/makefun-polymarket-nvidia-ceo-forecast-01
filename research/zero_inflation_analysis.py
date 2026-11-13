"""
Round-3 investigation #1: is there a real point-mass-at-"no change" structure in
these markets' prices, and if so, does modeling it explicitly (mixture of a point
mass at zero logit-change with the existing continuous logit-diffusion) beat the
current smooth model?

Two independent measurements:

(A) Pre-checkpoint history (the `prices` arrays in each case's *.in file -- these
    are already on a 1-minute grid, forward-filled from raw trade/quote ticks, so
    "zero diff" really means "no new trade moved the price this minute"). For each
    market we have ~3000 minutes of this. We measure, for each lag h, the fraction
    of h-step overlapping diffs that are *exactly* zero (raw price identical), as a
    function of h. This has huge sample size (pooled over ~20 markets x 6 cases x
    thousands of overlapping windows) so it's a reliable estimate of the market's
    own "staleness" structure, independent of the one checkpoint per case we have
    to forecast from.

(B) Post-checkpoint realized ground truth (research/history/realized.json, gridded
    the same way backtest.py does it): for each (case, market), is the realized
    price at h minutes after the checkpoint *exactly equal* to the checkpoint price?
    This is the actual quantity the model needs to predict well (it directly affects
    the 0.05/0.25/0.5 quantiles when "no change yet" is the modal outcome). Sample
    size here is only 6 cases x ~20 markets, so much noisier -- used as a sanity
    check against (A), not the primary evidence.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research.backtest import load_cases, load_history, build_realized, CASES

cases = load_cases()
hist = load_history()
realized = build_realized(cases, hist)

HORIZONS = [1, 2, 3, 5, 10, 20, 30, 60, 120, 240]

print("=== (A) Pre-checkpoint history: P(price unchanged over h minutes) ===")
print("pooled over all markets in all 6 practice cases, overlapping windows\n")

# pool counts per horizon
pooled = {h: [0, 0] for h in HORIZONS}  # [n_zero, n_total]
per_market_zero1 = []  # fraction of exactly-zero 1-step diffs, per market (for thin/active split)

for cid in CASES:
    c = cases[cid]
    for m in c["markets"]:
        prices = m["prices"]
        n = len(prices)
        if n < 2:
            continue
        n_zero1 = sum(1 for i in range(n - 1) if prices[i + 1] == prices[i])
        per_market_zero1.append((cid, m["id"], n_zero1 / (n - 1), n))
        for h in HORIZONS:
            if n <= h:
                continue
            for i in range(n - h):
                pooled[h][1] += 1
                if prices[i + h] == prices[i]:
                    pooled[h][0] += 1

print(f"{'h':>5s}  {'P(unchanged)':>14s}  {'n_pairs':>10s}")
for h in HORIZONS:
    nz, nt = pooled[h]
    frac = nz / nt if nt else float("nan")
    print(f"{h:5d}  {frac:14.4f}  {nt:10d}")

print()
print("--- thin vs active markets (split on fraction of zero 1-min diffs) ---")
per_market_zero1.sort(key=lambda r: r[2])
n_mk = len(per_market_zero1)
print(f"n markets = {n_mk}")
print("quartiles of per-market P(zero 1-min diff):")
for qname, idx in [("min", 0), ("p25", n_mk // 4), ("median", n_mk // 2),
                    ("p75", 3 * n_mk // 4), ("max", n_mk - 1)]:
    print(f"  {qname:6s}: {per_market_zero1[idx][2]:.4f}  ({per_market_zero1[idx][0]}/{per_market_zero1[idx][1]})")

thin = [r for r in per_market_zero1 if r[2] >= 0.85]
active = [r for r in per_market_zero1 if r[2] < 0.85]
print(f"\n'thin' markets (P(zero 1-min)>=0.85): {len(thin)}/{n_mk}")
print(f"'active' markets (P(zero 1-min)<0.85): {len(active)}/{n_mk}")

print()
print("=== (B) Post-checkpoint realized: P(realized price == checkpoint price) ===")
print(f"{'h':>5s}  {'P(unchanged)':>14s}  {'n_pairs':>10s}")
pooled_b = {h: [0, 0] for h in HORIZONS}
for cid in CASES:
    c = cases[cid]
    for m in c["markets"]:
        mid = m["id"]
        if mid not in realized[cid]:
            continue
        x0 = m["prices"][-1]
        row = realized[cid][mid]
        for h in HORIZONS:
            pooled_b[h][1] += 1
            if row[h - 1] == x0:
                pooled_b[h][0] += 1
for h in HORIZONS:
    nz, nt = pooled_b[h]
    frac = nz / nt if nt else float("nan")
    print(f"{h:5d}  {frac:14.4f}  {nt:10d}")

print()
print("=== per-case breakdown of (B) at h=1, h=5, h=30, h=240 ===")
for cid in CASES:
    c = cases[cid]
    row_counts = {h: [0, 0] for h in [1, 5, 30, 240]}
    for m in c["markets"]:
        mid = m["id"]
        if mid not in realized[cid]:
            continue
        x0 = m["prices"][-1]
        row = realized[cid][mid]
        for h in row_counts:
            row_counts[h][1] += 1
            if row[h - 1] == x0:
                row_counts[h][0] += 1
    parts = []
    for h in [1, 5, 30, 240]:
        nz, nt = row_counts[h]
        parts.append(f"h={h}:{nz}/{nt}={nz/nt if nt else float('nan'):.2f}")
    print(f"  {cid}: " + "  ".join(parts))
