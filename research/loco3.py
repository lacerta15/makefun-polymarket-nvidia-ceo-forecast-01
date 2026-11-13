"""
Leave-one-case-out robustness check for round-3 candidates (zero-inflation,
skew, reversal), in the same spirit as research/loco_xmarket.py: for each held-
out case, pick the best hyperparameters for a given feature family on the OTHER
5 cases, then report that choice's loss on the held-out case vs round 2's loss
on the same held-out case.

IMPORTANT CAVEAT (unlike round 2's cross-market lambda, which is a structural
property of the event's market design): all 6 practice cases are checkpoints on
the SAME continuous price history, spaced exactly 4 hours apart (p00 at 06:00,
p01 at 10:00, ..., p05 at 02:00 the next day -- see each case's "at" field and
shared "start" timestamp). Their pre-checkpoint histories overlap almost
entirely and their forecast windows abut in real time. So "leave-one-case-out"
here is a weaker test than true independence would give: a trend or event that
happened once in this ~24h window can make a hyperparameter look robust across
all 6 "cases" without it being a genuinely generalizable effect. This is most
relevant for REV, which uses a long lookback and is the most sensitive to this.
Treat LOCO "pass" here as necessary, not sufficient, evidence.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research.backtest import load_cases, load_history, build_realized, score_forecast, CASES
from research import models_round3 as M3

cases = load_cases()
hist = load_history()
realized = build_realized(cases, hist)


def case_loss(cid, **kwargs):
    c = cases[cid]
    horizon = c["horizon_minutes"]
    by_mid = M3.case_model_v3(c, horizon, **kwargs)
    total, n = 0.0, 0
    for m in c["markets"]:
        mid = m["id"]
        if mid not in realized[cid]:
            continue
        t, nn = score_forecast(realized[cid][mid], by_mid[mid])
        total += t
        n += nn
    return total / n


def loco(name, grid):
    """grid: list of kwargs dicts (candidate hyperparameter settings for this family)."""
    loss_table = {cid: {i: case_loss(cid, **kw) for i, kw in enumerate(grid)} for cid in CASES}
    baseline = {cid: case_loss(cid) for cid in CASES}  # round2, no kwargs

    print(f"=== LOCO: {name} ({len(grid)} candidate settings) ===")
    n_better = 0
    deltas = []
    for held in CASES:
        train = [c for c in CASES if c != held]
        best_i, best_train_loss = None, None
        for i in range(len(grid)):
            mean_train = sum(loss_table[c][i] for c in train) / len(train)
            if best_train_loss is None or mean_train < best_train_loss:
                best_train_loss = mean_train
                best_i = i
        held_loss = loss_table[held][best_i]
        base_loss = baseline[held]
        delta = held_loss - base_loss
        deltas.append(delta)
        tag = "BETTER" if delta < -1e-9 else ("WORSE" if delta > 1e-9 else "TIE")
        if delta < 0:
            n_better += 1
        print(f"  held={held}: best_setting={grid[best_i]}  round2={base_loss:.4f}  "
              f"cross_fit={held_loss:.4f}  delta={delta:+.4f} [{tag}]")
    print(f"  -> {n_better}/{len(CASES)} held-out cases improved; mean delta = "
          f"{sum(deltas)/len(deltas):+.4f}")
    print()
    return n_better, deltas


# --- Zero-inflation: sweep qmax and b jointly (small grid, cheap) ---
zi_grid = [dict(zi_enable=True, zi_qmax=qmax, zi_b=b)
           for qmax in [0.5, 0.7, 0.9]
           for b in [0.2, 0.27, 0.35]]
loco("Zero-inflation (ZI)", zi_grid)

# --- Skew: sweep g0, g1, cap ---
skew_grid = [dict(skew_enable=True, skew_g0=g0, skew_g1=g1, skew_cap=cap)
             for g0, g1, cap in [(0.1, 0.7, 0.2), (0.1, 1.4, 0.35), (0.15, 1.0, 0.25),
                                  (0.2, 1.4, 0.35), (0.3, 2.0, 0.3)]]
loco("Skew (SKEW)", skew_grid)

# --- Reversal: sweep alpha and lookback ---
rev_grid = [dict(rev_enable=True, rev_alpha=a, rev_lookback=L)
            for L in [30, 45, 60, 75, 90, 120, 150]
            for a in [0.04, 0.06, 0.08, 0.1, 0.12]]
loco("Reversal (REV)", rev_grid)

# --- Combined: ZI + modest REV (conservative alpha/lookback region) ---
combo_grid = [
    dict(zi_enable=True, zi_qmax=0.9, zi_b=0.27,
         rev_enable=True, rev_alpha=a, rev_lookback=L)
    for L in [30, 45, 60]
    for a in [0.04, 0.06, 0.08]
]
loco("ZI + conservative REV", combo_grid)
