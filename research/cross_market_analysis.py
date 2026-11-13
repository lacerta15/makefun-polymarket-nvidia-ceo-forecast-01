"""
Empirical investigation of the cross-market "sum-to-one" structure for round 2.

This event ("Next CEO of NVIDIA?") is a Polymarket negRisk multi-outcome event:
107 total candidate markets exist, but each practice case's `markets` list only
includes a subset (the ones with recorded price history / activity at that
checkpoint) -- 19 to 24 of them. In a frictionless negRisk market the Yes prices
of ALL outcomes sum to 1 (every outcome is mutually exclusive and exactly one
happens). Since our cases only see a *subset* of outcomes, the raw sum of listed
markets' prices need not be exactly 1 even in theory -- but if the subset carries
nearly all the probability mass (unlisted candidates are near-zero longshots),
it should be *close* to 1, with deviations from market inefficiency / thin
liquidity / bid-ask noise, not structural missing mass.

This script answers, empirically, on the 6 practice cases + fetched realized
ground truth:
  1. What is sum(last price) across each case's markets at the checkpoint?
  2. How does sum(realized price) evolve over the 240-minute forecast horizon --
     does it revert toward 1, stay near its checkpoint level, or drift freely?
  3. Does naively rescaling each market's forecast by (1 / checkpoint sum) --
     i.e. assuming the *future* sum will equal 1 exactly -- track what actually
     happens better than assuming the future sum stays at the checkpoint's own
     level S0?
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research.backtest import load_cases, load_history, build_realized, CASES

cases = load_cases()
hist = load_history()
realized = build_realized(cases, hist)

print("=== Checkpoint sums (sum of last observed price across case markets) ===")
checkpoint_sums = {}
for cid in CASES:
    c = cases[cid]
    prices = [m["prices"][-1] for m in c["markets"]]
    s0 = sum(prices)
    checkpoint_sums[cid] = s0
    print(f"  {cid}: n_markets={len(c['markets']):2d}  sum_last_price={s0:.4f}")

print()
print("=== Realized sum trajectory over the 240-minute horizon ===")
print("    (sum of realized Yes price across all of the case's markets, at each")
print("     checkpoint-relative minute -- shows whether it reverts toward 1,")
print("     stays near the checkpoint sum S0, or drifts elsewhere)")
for cid in CASES:
    c = cases[cid]
    mids = [m["id"] for m in c["markets"] if m["id"] in realized[cid]]
    rows = realized[cid]
    horizon = c["horizon_minutes"]
    # sum at h=1, 30, 60, 120, 180, 240
    checkpoints_h = [1, 15, 30, 60, 120, 180, 240]
    sums = []
    for h in checkpoints_h:
        s = sum(rows[mid][h - 1] for mid in mids)
        sums.append(s)
    s0 = checkpoint_sums[cid]
    print(f"  {cid}: S0={s0:.4f}  " + "  ".join(f"h={h}:{s:.4f}" for h, s in zip(checkpoints_h, sums)))

print()
print("=== Mean realized sum (over full horizon) vs checkpoint sum ===")
mean_realized_sum = {}
for cid in CASES:
    c = cases[cid]
    mids = [m["id"] for m in c["markets"] if m["id"] in realized[cid]]
    rows = realized[cid]
    horizon = c["horizon_minutes"]
    all_sums = [sum(rows[mid][h] for mid in mids) for h in range(horizon)]
    mean_s = sum(all_sums) / len(all_sums)
    mean_realized_sum[cid] = mean_s
    s0 = checkpoint_sums[cid]
    # how much of the gap (S0 - 1) persists on average, vs closing to 1?
    gap0 = s0 - 1.0
    gap_mean = mean_s - 1.0
    frac_persisting = gap_mean / gap0 if abs(gap0) > 1e-9 else float("nan")
    print(f"  {cid}: S0={s0:.4f}  mean_future_sum={mean_s:.4f}  "
          f"gap0={gap0:+.4f}  mean_gap={gap_mean:+.4f}  frac_persisting={frac_persisting:.2f}")

print()
print("=== Does the per-market price level correlate with 'excess' mass? ===")
print("    (checking whether sum > 1 cases are driven by a few markets near 0.5")
print("     overlapping, vs sum < 1 cases missing a dominant candidate)")
for cid in CASES:
    c = cases[cid]
    prices = sorted(((m["prices"][-1], m["name"]) for m in c["markets"]), reverse=True)
    top3 = ", ".join(f"{n}={p:.3f}" for p, n in prices[:3])
    print(f"  {cid}: top3 = {top3}")
