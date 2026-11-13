"""
Round-2 candidate models: cross-market sum-to-one normalization on top of round 1's
logit-space sub-diffusive per-market volatility model.

All of these take a *whole case* (case_model_fn(case_dict, horizon) -> {mid: rows}),
not a single market, because they need the cross-sectional sum of all of the case's
current prices before forecasting any one market. Evaluated with
`research.backtest.evaluate_case_level`.

Core idea: this event's markets are (an incomplete subset of) outcomes of one
negRisk categorical event, so sum(price) across markets "should" be ~1. When the
observed sum S0 deviates from 1, that is evidence of a common-mode mispricing
across the whole case (thin liquidity / bid-ask noise on the shared negRisk
basket), not evidence about any one candidate specifically. We test whether
nudging every market's forecast *proportionally* toward a renormalized price
(x_i' = x_i * k, k = 1 + lambda*(1/S0 - 1)) reduces pinball loss, for a range of
shrinkage strengths lambda (0 = round 1 baseline / no adjustment, 1 = full
renormalization to sum exactly 1), and whether decaying that shift back to 0
over the forecast horizon helps (since the 4-hour-ahead common-mode correction
is less certain to hold than the immediate one).
"""
import math

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.644854, -0.67449, 0.0, 0.67449, 1.644854]

EPS = 1e-6
WINDOW = 180
H_EXPONENT = 0.30
FLOOR_SIGMA_LOGIT = 0.03
CAP_SIGMA_LOGIT = 1.2


def logit(p, eps=EPS):
    p = min(1.0 - eps, max(eps, p))
    return math.log(p / (1.0 - p))


def inv_logit(z):
    if z > 40.0:
        return 1.0
    if z < -40.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))


def price6(x):
    x = min(1.0, max(0.0, x))
    return round(x * 1e6) / 1e6


def estimate_sigma1(prices, window=WINDOW, floor=FLOOR_SIGMA_LOGIT, cap=CAP_SIGMA_LOGIT, eps=EPS):
    n = len(prices)
    tail_n = min(window, n - 1)
    sigma = None
    if tail_n >= 2:
        tail = prices[-(tail_n + 1):]
        diffs = [logit(tail[i + 1], eps) - logit(tail[i], eps) for i in range(len(tail) - 1)]
        m = len(diffs)
        mean = sum(diffs) / m
        var = sum((d - mean) ** 2 for d in diffs) / max(1, m - 1)
        sigma = math.sqrt(var)
    if sigma is None or sigma <= 0.0 or math.isnan(sigma) or math.isinf(sigma):
        sigma = floor
    return max(floor, min(cap, sigma))


def round1_forecast(x, prices, horizon, z_levels=Z, h_exp=H_EXPONENT):
    """Round 1's exact per-market model, given an (possibly adjusted) current price x
    and the market's own price history (used only for the sigma1 volatility estimate)."""
    z0 = logit(x)
    sigma1 = estimate_sigma1(prices)
    rows = []
    for h in range(1, horizon + 1):
        spread = sigma1 * (h ** h_exp)
        rows.append([price6(inv_logit(z0 + z * spread)) for z in z_levels])
    return rows


# ---------------------------------------------------------------------------
# Cross-market normalization variants
# ---------------------------------------------------------------------------

def _case_sum(c, use_window=None):
    """sum of current prices across the case's markets. If use_window is set,
    use the mean of the trailing-window sum instead of the single last price
    (reduces snapshot noise from one thin market's bid/ask)."""
    if use_window is None:
        return sum(m["prices"][-1] for m in c["markets"])
    # align all markets' last `use_window` minutes (they're all sampled at the
    # same minute grid up to the checkpoint) and average the per-minute sums
    w = use_window
    cols = []
    for m in c["markets"]:
        p = m["prices"]
        tail = p[-w:] if len(p) >= w else p
        cols.append(tail)
    minlen = min(len(t) for t in cols)
    if minlen == 0:
        return sum(m["prices"][-1] for m in c["markets"])
    sums = [sum(t[-(i + 1)] for t in cols) for i in range(minlen)]
    return sum(sums) / len(sums)


def xmarket_shift(c, horizon, lam=1.0, sum_window=None, decay_half_life=None,
                   z_levels=Z, h_exp=H_EXPONENT):
    """Shift every market's current price multiplicatively toward sum-to-one.

    k0 = 1 + lam*(1/S0 - 1)   (lam=0 -> no shift, lam=1 -> full renorm to S0*k0=1)
    If decay_half_life is set, the shift decays geometrically with minutes ahead:
      k(h) = 1 + (k0 - 1) * 0.5**(h / decay_half_life)
    so at h=0 the full shift applies and it relaxes back to the raw price for
    large h. If None, the same shift k0 is used for the whole horizon (applied
    by adjusting z0 once; sigma/spread growth is unaffected either way).
    """
    S0 = _case_sum(c, use_window=sum_window)
    if S0 <= 1e-9:
        k0 = 1.0
    else:
        k0 = 1.0 + lam * (1.0 / S0 - 1.0)

    out = {}
    for m in c["markets"]:
        mid = m["id"]
        prices = m["prices"]
        x = prices[-1]
        sigma1 = estimate_sigma1(prices)
        if decay_half_life is None:
            x_adj = min(1.0 - EPS, max(EPS, x * k0))
            z0 = logit(x_adj)
            rows = []
            for h in range(1, horizon + 1):
                spread = sigma1 * (h ** h_exp)
                rows.append([price6(inv_logit(z0 + z * spread)) for z in z_levels])
        else:
            z0_raw = logit(x)
            rows = []
            for h in range(1, horizon + 1):
                decay = 0.5 ** (h / decay_half_life)
                k_h = 1.0 + (k0 - 1.0) * decay
                x_adj = min(1.0 - EPS, max(EPS, x * k_h))
                z0_h = logit(x_adj)
                spread = sigma1 * (h ** h_exp)
                rows.append([price6(inv_logit(z0_h + z * spread)) for z in z_levels])
        out[mid] = rows
    return out


def xmarket_logit_shift(c, horizon, lam=1.0, sum_window=None,
                         z_levels=Z, h_exp=H_EXPONENT):
    """Alternative: shift in logit space by a constant lam*log(1/S0) applied
    equally to every market's z0, instead of multiplicative price rescaling."""
    S0 = _case_sum(c, use_window=sum_window)
    shift = lam * math.log(max(1e-9, 1.0 / max(1e-9, S0))) if S0 > 1e-9 else 0.0
    out = {}
    for m in c["markets"]:
        mid = m["id"]
        prices = m["prices"]
        x = prices[-1]
        sigma1 = estimate_sigma1(prices)
        z0 = logit(x) + shift
        rows = []
        for h in range(1, horizon + 1):
            spread = sigma1 * (h ** h_exp)
            rows.append([price6(inv_logit(z0 + z * spread)) for z in z_levels])
        out[mid] = rows
    return out


def baseline_case_model(c, horizon, z_levels=Z, h_exp=H_EXPONENT):
    """Round 1, wrapped as a case-level model (lam=0 equivalent) for apples-to-apples
    comparison inside evaluate_case_level."""
    out = {}
    for m in c["markets"]:
        mid = m["id"]
        prices = m["prices"]
        x = prices[-1]
        out[mid] = round1_forecast(x, prices, horizon, z_levels, h_exp)
    return out
