"""
Round-3 investigations #2 (skewness) and #3 (short-term reversal / bounce), run on
the pre-checkpoint `prices` histories pooled across all 6 practice cases' markets
(large sample: ~20 markets x 6 cases x thousands of overlapping windows each).

#2 Skewness: is there a real, level-dependent asymmetry in realized logit-price
   changes? Partition 1-minute (and h-minute) logit diffs by the *current price
   level* at the start of the window (low / mid / high) and compare the skewness
   (and simple up-move vs down-move frequency/magnitude) of the diff distribution
   in each bucket. If markets near 0 jump up more easily (in logit terms) than they
   jump down, and vice versa near 1, that's a real, structural, price-level-driven
   skew worth encoding (not just noise from one case).

#3 Reversal/"bounce": after a larger-than-typical recent move (say, over the last
   L minutes), does the price tend to partially revert over the next few minutes,
   beyond what the sub-diffusion exponent (h^0.3) already captures on average?
   Measured via the correlation between a lagged move (z[t] - z[t-L]) and the
   subsequent move (z[t+F] - z[t]), pooled over all markets/cases. A negative
   correlation (controlling for L) is evidence of mean-reversion beyond the
   existing power-law decay (which is already a form of "average" reversion but
   doesn't condition on how big the *most recent* move was).
"""
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research.backtest import load_cases, CASES

cases = load_cases()
EPS = 1e-6


def logit(p, eps=EPS):
    p = min(1 - eps, max(eps, p))
    return math.log(p / (1 - p))


# ---------------------------------------------------------------------------
# #2 Skewness by price level
# ---------------------------------------------------------------------------
print("=== Skewness of 1-minute logit diffs, by price-level bucket ===")
buckets = {"low(<0.15)": [], "mid(0.15-0.5)": [], "midhigh(0.5-0.85)": [], "high(>0.85)": []}


def bucket_for(p):
    if p < 0.15:
        return "low(<0.15)"
    if p < 0.5:
        return "mid(0.15-0.5)"
    if p < 0.85:
        return "midhigh(0.5-0.85)"
    return "high(>0.85)"


for cid in CASES:
    for m in cases[cid]["markets"]:
        prices = m["prices"]
        n = len(prices)
        for i in range(n - 1):
            p0 = prices[i]
            d = logit(prices[i + 1]) - logit(p0)
            if d == 0.0:
                continue  # exclude exact-zero (staleness), focus on the moves that DO happen
            buckets[bucket_for(p0)].append(d)


def skew(xs):
    n = len(xs)
    if n < 3:
        return float("nan")
    mean = sum(xs) / n
    m2 = sum((x - mean) ** 2 for x in xs) / n
    m3 = sum((x - mean) ** 3 for x in xs) / n
    sd = math.sqrt(m2)
    if sd == 0:
        return float("nan")
    return m3 / sd ** 3


print(f"{'bucket':20s} {'n':>8s} {'mean':>9s} {'skew':>9s} {'P(up)':>7s} {'P(down)':>7s}")
for b, xs in buckets.items():
    if not xs:
        continue
    n = len(xs)
    nup = sum(1 for x in xs if x > 0)
    ndown = sum(1 for x in xs if x < 0)
    print(f"{b:20s} {n:8d} {sum(xs)/n:9.5f} {skew(xs):9.4f} {nup/n:7.3f} {ndown/n:7.3f}")

print()
print("--- leave-one-case-out: skew sign of 'low' and 'high' buckets per case ---")
for cid in CASES:
    low, high = [], []
    for m in cases[cid]["markets"]:
        prices = m["prices"]
        n = len(prices)
        for i in range(n - 1):
            p0 = prices[i]
            d = logit(prices[i + 1]) - logit(p0)
            if d == 0.0:
                continue
            if p0 < 0.15:
                low.append(d)
            elif p0 > 0.85:
                high.append(d)
    sl = skew(low) if len(low) >= 3 else float("nan")
    sh = skew(high) if len(high) >= 3 else float("nan")
    print(f"  {cid}: low n={len(low):5d} skew={sl:8.4f}   high n={len(high):5d} skew={sh:8.4f}")


# ---------------------------------------------------------------------------
# #3 Reversal / bounce
# ---------------------------------------------------------------------------
print()
print("=== Reversal: corr(lagged move z[t]-z[t-L], future move z[t+F]-z[t]) ===")
print("pooled over all markets/cases, overlapping windows\n")


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan"), n
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return float("nan"), n
    return sxy / math.sqrt(sxx * syy), n


for L in [5, 15, 30, 60]:
    for F in [5, 15, 30, 60]:
        xs, ys = [], []
        for cid in CASES:
            for m in cases[cid]["markets"]:
                prices = m["prices"]
                zs = [logit(p) for p in prices]
                n = len(zs)
                for t in range(L, n - F):
                    lag = zs[t] - zs[t - L]
                    fut = zs[t + F] - zs[t]
                    xs.append(lag)
                    ys.append(fut)
        r, n = pearson(xs, ys)
        print(f"  L={L:3d} F={F:3d}  r={r:8.4f}  n={n}")

print()
print("--- conditional on |lagged move| being large (top decile) ---")
for L in [15, 30]:
    for F in [15, 30, 60]:
        pairs = []
        for cid in CASES:
            for m in cases[cid]["markets"]:
                prices = m["prices"]
                zs = [logit(p) for p in prices]
                n = len(zs)
                for t in range(L, n - F):
                    lag = zs[t] - zs[t - L]
                    fut = zs[t + F] - zs[t]
                    pairs.append((lag, fut))
        pairs.sort(key=lambda r: abs(r[0]))
        n = len(pairs)
        top = pairs[int(0.9 * n):]
        lag_sign_follow = sum(1 for lag, fut in top if (fut * lag) > 0)
        lag_sign_rev = sum(1 for lag, fut in top if (fut * lag) < 0)
        mean_fut_given_pos_lag = (sum(fut for lag, fut in top if lag > 0) /
                                   max(1, sum(1 for lag, fut in top if lag > 0)))
        mean_fut_given_neg_lag = (sum(fut for lag, fut in top if lag < 0) /
                                   max(1, sum(1 for lag, fut in top if lag < 0)))
        print(f"  L={L:3d} F={F:3d} top-decile |lag|: n={len(top)}  "
              f"same-sign={lag_sign_follow} opp-sign={lag_sign_rev}  "
              f"E[fut|lag>0]={mean_fut_given_pos_lag:+.5f}  E[fut|lag<0]={mean_fut_given_neg_lag:+.5f}")
