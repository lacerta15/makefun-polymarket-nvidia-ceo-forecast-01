import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research.backtest import load_cases, load_history, build_realized, evaluate_case_level, CASES
from research import models_round3 as M3

cases = load_cases()
hist = load_history()
realized = build_realized(cases, hist)

results = []


def run(name, **kwargs):
    fn = lambda c, h: M3.case_model_v3(c, h, **kwargs)
    loss, per_case = evaluate_case_level(fn, cases, realized)
    results.append((name, loss, per_case))
    print(f"{name:60s} {loss:.6f}   " + " ".join(f"{per_case[c]:.4f}" for c in CASES))
    return loss


print(f"{'model':60s} {'mean_loss':10s}   " + " ".join(CASES))
round2 = run("round2 baseline (no new features)")

print()
print("--- Zero-inflation only: vary qmax, b, preserve_var ---")
for qmax in [0.5, 0.7, 0.8, 0.9, 0.95]:
    run(f"ZI qmax={qmax} b=0.27 preserve_var=T", zi_enable=True, zi_qmax=qmax)
for qmax in [0.7, 0.9]:
    run(f"ZI qmax={qmax} preserve_var=F", zi_enable=True, zi_qmax=qmax, zi_preserve_var=False)
for b in [0.0, 0.15, 0.27, 0.4, 0.6]:
    run(f"ZI qmax=0.9 b={b}", zi_enable=True, zi_qmax=0.9, zi_b=b)
for w in [60, 120, 180, 300, 500]:
    run(f"ZI qmax=0.9 b=0.27 window={w}", zi_enable=True, zi_qmax=0.9, zi_window=w)

print()
print("--- Skew only: vary g0, g1, cap ---")
for g0, g1, cap in [(0.1, 0.7, 0.2), (0.2, 1.4, 0.35), (0.2, 1.4, 0.2), (0.3, 2.0, 0.3),
                     (0.15, 1.0, 0.25), (0.4, 2.0, 0.4), (0.1, 1.4, 0.35)]:
    run(f"SKEW g0={g0} g1={g1} cap={cap}", skew_enable=True, skew_g0=g0, skew_g1=g1, skew_cap=cap)

print()
print("--- Reversal only: vary alpha, lookback, asymmetric ---")
for alpha in [0.01, 0.02, 0.05, 0.08, 0.12, 0.2]:
    run(f"REV alpha={alpha} L=30 asym=T", rev_enable=True, rev_alpha=alpha, rev_lookback=30)
for alpha in [0.05, 0.1]:
    run(f"REV alpha={alpha} L=30 asym=F", rev_enable=True, rev_alpha=alpha, rev_lookback=30,
        rev_asymmetric=False)
for L in [10, 15, 30, 60]:
    run(f"REV alpha=0.08 L={L} asym=T", rev_enable=True, rev_alpha=0.08, rev_lookback=L)

print()
print("--- Combinations (best of each family) ---")
BEST_ZI = dict(zi_enable=True, zi_qmax=0.9, zi_b=0.27, zi_window=180)
BEST_SKEW = dict(skew_enable=True, skew_g0=0.1, skew_g1=1.4, skew_cap=0.35)
BEST_REV = dict(rev_enable=True, rev_alpha=0.08, rev_lookback=30)

run("ZI+SKEW", **BEST_ZI, **BEST_SKEW)
run("ZI+REV", **BEST_ZI, **BEST_REV)
run("SKEW+REV", **BEST_SKEW, **BEST_REV)
run("ZI+SKEW+REV", **BEST_ZI, **BEST_SKEW, **BEST_REV)

print()
print("Sorted by loss (top 25):")
for name, loss, pc in sorted(results, key=lambda r: r[1])[:25]:
    tag = "  <-- BEATS ROUND2" if loss < round2 - 1e-9 else ""
    print(f"  {loss:.6f}  {name}{tag}")

print()
print(f"round2 baseline: {round2:.6f}")
