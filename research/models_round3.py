"""
Round-3 candidate models, layered on top of round 2 (logit-space sub-diffusive
per-market volatility + cross-market sum-to-one shrinkage, XMARKET_LAMBDA=0.30).

Three independent, separately-toggleable extensions, each motivated by a
measured empirical pattern in research/zero_inflation_analysis.py and
research/skew_reversal_analysis.py:

1. Zero-inflation mixture (ZI). Measured: pooled over the 6 practice cases'
   pre-checkpoint history, P(price unchanged over h minutes) falls from ~0.88 at
   h=1 to ~0.12 at h=240 (research/zero_inflation_analysis.py, part A), closely
   matching round 1's variance-ratio finding and confirmed independently in the
   post-checkpoint realized ground truth (part B). Instead of only encoding this
   via a sub-diffusive *smooth* exponent (round 1/2), model the h-step logit
   change explicitly as a mixture: a point mass at "no change at all" with
   probability q(h), plus the existing continuous logit-normal with probability
   1-q(h), with q(h) anchored per-market by that market's OWN measured 1-minute
   "stuck" fraction and decayed with h via a globally-fit power law (exponent
   ~0.27, fit on the pooled ratio q(h)/q(1) in log-log space -- notably close to
   the existing H_EXPONENT=0.30, consistent with both being symptoms of the same
   underlying bursty-activity structure).

2. Level-dependent skew (SKEW). Measured: partitioning 1-minute logit diffs by
   the starting price level, markets below ~0.15 show strong, highly robust
   (same sign in all 6 practice cases, large pooled n) *positive* skew (rare big
   up-jumps), while markets from ~0.15 to ~0.5 show equally robust *negative*
   skew (rare big down-moves), growing in magnitude with price level across that
   validated range. There is essentially no data above ~0.6 in the practice
   cases (these are long-shot-heavy candidate markets), so the effect is encoded
   only as a function of the *current* price, capped, and NOT extrapolated with
   unbounded confidence toward 1 -- a conservative clamp limits how negative the
   skew can get, exactly as round 2 clamped k0 against cases we have not seen.

3. Short-term reversal / momentum drift (REV). Measured: the correlation between
   a market's lagged L-minute logit move and its subsequent F-minute logit move
   is consistently negative (~-0.15 to -0.30) across many (L, F) combinations
   pooled over all practice markets, including F out to 240 minutes -- i.e. this
   is not a transient few-minute bounce but a persistent reversion signal that
   does not decay away over the 4-hour horizon. It is also clearly *asymmetric*:
   conditioning on the top decile of |lag|, upward recent moves predict strong
   reversion (E[future move] very negative), while downward recent moves predict
   almost no reversion (E[future move] near zero). Modeled as an asymmetric drift
   added to the forecast center, proportional to max(0, recent move) and growing
   with h at roughly the same sub-diffusive rate as the existing spread term
   (consistent with the measured E[future|lag] magnitude growing slowly and
   sub-linearly with the forecast horizon F).

All three are optional (each has an `enable` flag) so research/run_sweep3.py can
ablate them individually and in combination, and research/loco3.py can check
leave-one-case-out robustness before anything is adopted into main.py.
"""
import math

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.644854, -0.67449, 0.0, 0.67449, 1.644854]

EPS = 1e-6
WINDOW = 180
H_EXPONENT = 0.30
FLOOR_SIGMA_LOGIT = 0.03
CAP_SIGMA_LOGIT = 1.2

XMARKET_LAMBDA = 0.30
K0_MIN = 0.5
K0_MAX = 2.0
S0_FLOOR = 1e-3


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


def norm_ppf(p):
    """Inverse CDF of the standard normal (Acklam's algorithm)."""
    if p <= 0.0:
        return -8.0
    if p >= 1.0:
        return 8.0
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    p_low = 0.02425
    p_high = 1.0 - p_low
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0)
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
            ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)


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


def estimate_q1(prices, window=WINDOW):
    """Per-market fraction of exactly-zero 1-minute (raw) price diffs over the
    trailing `window` minutes (or everything available). This is the market's
    own measured "staleness" level, anchoring the zero-inflation mixture at h=1."""
    n = len(prices)
    tail_n = min(window, n - 1)
    if tail_n < 1:
        return 0.0
    tail = prices[-(tail_n + 1):]
    nz = sum(1 for i in range(len(tail) - 1) if tail[i + 1] == tail[i])
    return nz / (len(tail) - 1)


def current_price(prices):
    return prices[-1] if prices else 0.5


def cross_market_k0(markets, lam=XMARKET_LAMBDA):
    s0 = sum(current_price(m["prices"]) for m in markets)
    if s0 <= S0_FLOOR:
        return 1.0
    k0 = 1.0 + lam * (1.0 / s0 - 1.0)
    return max(K0_MIN, min(K0_MAX, k0))


# ---------------------------------------------------------------------------
# 1. Zero-inflation mixture
# ---------------------------------------------------------------------------

ZI_B = 0.27          # global decay exponent for q(h)/q(1), fit on pooled practice data
ZI_QMAX = 0.90        # cap on q(h) (keeps the continuous part's variance finite/sane)
ZI_WINDOW = WINDOW


def zero_inflated_quantile(tau, q, spread_cont):
    """Quantile of the mixture: point mass at 0 w.p. q, N(0, spread_cont^2) w.p. 1-q."""
    if q <= 0.0:
        return spread_cont * norm_ppf(tau)
    one_m_q = 1.0 - q
    half = one_m_q / 2.0
    if tau < half:
        return spread_cont * norm_ppf(tau / one_m_q)
    if tau > 1.0 - half:
        return spread_cont * norm_ppf((tau - q) / one_m_q)
    return 0.0


# ---------------------------------------------------------------------------
# 2. Level-dependent skew
# ---------------------------------------------------------------------------

SKEW_G0 = 0.10        # gamma(x) = clamp(G0 - G1*x, -CAP, CAP)
SKEW_G1 = 1.40        # defaults here are the values actually shipped in main.py
SKEW_CAP = 0.35       # (chosen conservatively -- see run_sweep3.py / loco3.py)


def skew_gamma(x, g0=SKEW_G0, g1=SKEW_G1, cap=SKEW_CAP):
    g = g0 - g1 * x
    return max(-cap, min(cap, g))


# ---------------------------------------------------------------------------
# 3. Short-term reversal / momentum drift
# ---------------------------------------------------------------------------

REV_LOOKBACK = 45     # shipped value in main.py; alpha defaults to 0 (disabled) here
REV_ALPHA = 0.0       # so callers/sweeps opt in explicitly -- shipped value is 0.03
REV_H_EXP = H_EXPONENT


def recent_move(prices, lookback=REV_LOOKBACK):
    n = len(prices)
    L = min(lookback, n - 1)
    if L < 1:
        return 0.0
    return logit(prices[-1]) - logit(prices[-1 - L])


# ---------------------------------------------------------------------------
# Unified per-market forecaster
# ---------------------------------------------------------------------------

def forecast_market_v3(prices, horizon, k0=1.0, z_levels=Z, quantiles=QUANTILES,
                        h_exp=H_EXPONENT,
                        zi_enable=False, zi_b=ZI_B, zi_qmax=ZI_QMAX, zi_window=ZI_WINDOW,
                        zi_preserve_var=True,
                        skew_enable=False, skew_g0=SKEW_G0, skew_g1=SKEW_G1, skew_cap=SKEW_CAP,
                        rev_enable=False, rev_alpha=REV_ALPHA, rev_lookback=REV_LOOKBACK,
                        rev_h_exp=REV_H_EXP, rev_asymmetric=True):
    x = current_price(prices)
    x_adj = min(1.0 - EPS, max(EPS, x * k0))
    z0 = logit(x_adj)
    sigma1 = estimate_sigma1(prices)

    q1 = estimate_q1(prices, window=zi_window) if zi_enable else 0.0
    gamma = skew_gamma(x_adj, skew_g0, skew_g1, skew_cap) if skew_enable else 0.0

    lag = recent_move(prices, rev_lookback) if rev_enable else 0.0
    signed_lag = max(0.0, lag) if rev_asymmetric else lag

    rows = []
    for h in range(1, horizon + 1):
        spread = sigma1 * (h ** h_exp)
        center = z0
        if rev_enable and rev_alpha != 0.0:
            center -= rev_alpha * signed_lag * (h ** rev_h_exp)

        if zi_enable:
            qh = q1 * (h ** (-zi_b))
            qh = max(0.0, min(zi_qmax, qh))
        else:
            qh = 0.0

        if zi_enable and zi_preserve_var and qh > 0.0:
            spread_cont = spread / math.sqrt(max(1e-6, 1.0 - qh))
        else:
            spread_cont = spread

        row = []
        for tau, z in zip(quantiles, z_levels):
            if zi_enable and qh > 0.0:
                d = zero_inflated_quantile(tau, qh, spread_cont)
            else:
                d = spread_cont * z
            if skew_enable and gamma != 0.0:
                if d >= 0.0:
                    d = d * (1.0 + gamma)
                else:
                    d = d * (1.0 - gamma)
            row.append(price6(inv_logit(center + d)))
        rows.append(row)
    return rows


def case_model_v3(c, horizon, **kwargs):
    """Case-level wrapper for evaluate_case_level: applies round-2's cross-market
    k0 shift, then forecast_market_v3 per market with the given toggles."""
    k0 = cross_market_k0(c["markets"])
    out = {}
    for m in c["markets"]:
        out[m["id"]] = forecast_market_v3(m["prices"], horizon, k0=k0, **kwargs)
    return out


def round2_baseline_case_model(c, horizon):
    """Round 2 exactly, wrapped the same way, for apples-to-apples comparison."""
    return case_model_v3(c, horizon)
