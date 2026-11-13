import sys
sys.path.insert(0, ".")
from research.backtest import load_cases, load_history, build_realized, evaluate, CASES
from research import models as M

cases = load_cases()
hist = load_history()
realized = build_realized(cases, hist)

results = []

def run(name, fn):
    loss, per_case = evaluate(fn, cases, realized)
    results.append((name, loss, per_case))
    print(f"{name:45s} {loss:.6f}   " + " ".join(f"{per_case[c]:.4f}" for c in CASES))

print(f"{'model':45s} {'mean_loss':10s}   " + " ".join(CASES))
run("house_baseline (sigma=0.0025)", lambda m, h: M.house_baseline(m, h, sigma=0.0025))

for window in [30, 60, 120, 180, 240, 360]:
    run(f"per_market_sigma_raw window={window}", lambda m, h, w=window: M.per_market_sigma_raw(m, h, window=w))

for window in [30, 60, 120, 180, 240, 360]:
    run(f"logit_rw window={window}", lambda m, h, w=window: M.logit_rw(m, h, window=w))

for mult in [0.8, 1.0, 1.2, 1.5, 2.0]:
    run(f"logit_rw window=180 mult={mult}", lambda m, h, mm=mult: M.logit_rw(m, h, window=180, mult=mm))

for tail_mult in [1.0, 1.15, 1.3, 1.5, 1.75, 2.0]:
    run(f"logit_rw_fattail window=180 tail_mult={tail_mult}", lambda m, h, tm=tail_mult: M.logit_rw_fattail(m, h, window=180, tail_mult=tm))

for dd in [0.0, 0.15, 0.3, 0.5, 0.75, 1.0]:
    run(f"logit_rw_drift window=180 drift_damp={dd}", lambda m, h, d=dd: M.logit_rw_drift(m, h, window=180, drift_damp=d))

print()
print("Sorted by loss:")
for name, loss, pc in sorted(results, key=lambda r: r[1]):
    print(f"  {loss:.6f}  {name}")
