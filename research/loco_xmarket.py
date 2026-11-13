"""Leave-one-case-out robustness check for the cross-market shift lambda.

For each held-out case, pick the best constant lambda on the OTHER 5 cases
(from a coarse grid), then report that lambda's loss on the held-out case vs
round 1's (lambda=0) loss on that same held-out case. If cross-fitting
consistently helps (or at least never meaningfully hurts) the held-out case,
that's evidence the structural pull-toward-sum-1 effect generalizes, rather
than being a fit to exactly these 6 cases.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research.backtest import load_cases, load_history, build_realized, CASES
from research import models_xmarket as X

cases = load_cases()
hist = load_history()
realized = build_realized(cases, hist)

LAM_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def case_loss(cid, lam):
    """loss for a single case under a given constant lambda."""
    from research.backtest import score_forecast
    c = cases[cid]
    horizon = c["horizon_minutes"]
    by_mid = X.xmarket_shift(c, horizon, lam=lam)
    total, n = 0.0, 0
    for m in c["markets"]:
        mid = m["id"]
        if mid not in realized[cid]:
            continue
        t, nn = score_forecast(realized[cid][mid], by_mid[mid])
        total += t
        n += nn
    return total / n


# cache per-case, per-lambda losses
loss_table = {cid: {lam: case_loss(cid, lam) for lam in LAM_GRID} for cid in CASES}

print("per-case loss by lambda:")
print(f"{'cid':5s} " + " ".join(f"{lam:7.1f}" for lam in LAM_GRID))
for cid in CASES:
    print(f"{cid:5s} " + " ".join(f"{loss_table[cid][lam]:7.4f}" for lam in LAM_GRID))

print()
print("=== Leave-one-case-out ===")
held_out_results = []
for held in CASES:
    train = [c for c in CASES if c != held]
    # pick lambda minimizing mean loss over the training cases
    best_lam, best_train_loss = None, None
    for lam in LAM_GRID:
        mean_train = sum(loss_table[c][lam] for c in train) / len(train)
        if best_train_loss is None or mean_train < best_train_loss:
            best_train_loss = mean_train
            best_lam = lam
    held_loss_cross = loss_table[held][best_lam]
    held_loss_round1 = loss_table[held][0.0]
    held_out_results.append((held, best_lam, held_loss_round1, held_loss_cross))
    delta = held_loss_cross - held_loss_round1
    tag = "BETTER" if delta < 0 else ("WORSE" if delta > 1e-9 else "TIE")
    print(f"  held-out={held}: best_lam_on_other5={best_lam:.1f}  "
          f"round1_loss={held_loss_round1:.4f}  cross_fit_loss={held_loss_cross:.4f}  "
          f"delta={delta:+.4f} [{tag}]")

n_better = sum(1 for _, _, r1, cf in held_out_results if cf < r1)
print(f"\n{n_better}/{len(CASES)} held-out cases improved by the cross-fit lambda.")

# also report what a single fixed lambda (chosen once on ALL 6, the real deployment
# choice) looks like, for comparison
print()
print("=== Fixed lambda on all 6 (what we'd actually ship) ===")
for lam in [0.3, 0.4, 0.5, 0.6, 0.7]:
    mean_all = sum(loss_table[c][lam] for c in CASES) / len(CASES)
    print(f"  lam={lam:.1f}  mean_loss_all6={mean_all:.6f}")
