import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research.backtest import load_cases, load_history, build_realized, evaluate_case_level, CASES
from research import models_xmarket as X

cases = load_cases()
hist = load_history()
realized = build_realized(cases, hist)

results = []

def run(name, fn):
    loss, per_case = evaluate_case_level(fn, cases, realized)
    results.append((name, loss, per_case))
    print(f"{name:55s} {loss:.6f}   " + " ".join(f"{per_case[c]:.4f}" for c in CASES))
    return loss

print(f"{'model':55s} {'mean_loss':10s}   " + " ".join(CASES))
baseline = run("round1_baseline (lam=0, case-wrapped)", X.baseline_case_model)

print()
print("--- multiplicative price shift, no decay, sum_window=None (instant last price) ---")
for lam in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5]:
    run(f"xmarket_shift lam={lam}", lambda c, h, L=lam: X.xmarket_shift(c, h, lam=L))

print()
print("--- multiplicative price shift, sum_window=30 (smoothed S0) ---")
for lam in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]:
    run(f"xmarket_shift w=30 lam={lam}", lambda c, h, L=lam: X.xmarket_shift(c, h, lam=L, sum_window=30))

print()
print("--- multiplicative price shift with decay half-life ---")
for hl in [30, 60, 120, 240]:
    for lam in [0.5, 1.0]:
        run(f"xmarket_shift lam={lam} half_life={hl}",
            lambda c, h, L=lam, HL=hl: X.xmarket_shift(c, h, lam=L, decay_half_life=HL))

print()
print("--- logit-space constant shift ---")
for lam in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]:
    run(f"xmarket_logit_shift lam={lam}", lambda c, h, L=lam: X.xmarket_logit_shift(c, h, lam=L))

print()
print("Sorted by loss (top 20):")
for name, loss, pc in sorted(results, key=lambda r: r[1])[:20]:
    tag = "  <-- BEATS ROUND1" if loss < baseline - 1e-9 else ""
    print(f"  {loss:.6f}  {name}{tag}")

print()
print(f"round1 baseline: {baseline:.6f}")
