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
    return loss

print("=== raw-space per_market_sigma: window x mult grid ===")
for window in [90, 120, 150, 180, 210, 240]:
    for mult in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        loss = run(f"raw w={window} mult={mult}", lambda m,h,w=window,mm=mult: M.per_market_sigma_raw(m,h,window=w,mult=mm))
        print(f"  w={window:4d} mult={mult:.2f}  loss={loss:.6f}")

print()
print("=== logit-space: window x mult grid ===")
for window in [90, 120, 150, 180, 210, 240]:
    for mult in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        loss = run(f"logit w={window} mult={mult}", lambda m,h,w=window,mm=mult: M.logit_rw(m,h,window=w,mult=mm))
        print(f"  w={window:4d} mult={mult:.2f}  loss={loss:.6f}")

print()
print("Top 15 overall:")
for name, loss, pc in sorted(results, key=lambda r: r[1])[:15]:
    print(f"  {loss:.6f}  {name}   per-case: " + " ".join(f"{pc[c]:.4f}" for c in CASES))
