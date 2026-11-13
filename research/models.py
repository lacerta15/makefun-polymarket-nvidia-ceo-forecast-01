"""Candidate forecasting models, evaluated via backtest.py against real realized outcomes."""
import math

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.644854, -0.67449, 0.0, 0.67449, 1.644854]

def price6(x):
    x = min(1.0, max(0.0, x))
    return round(x * 1e6) / 1e6

# ---------- Model 1: house baseline (constant sigma random walk) ----------
def house_baseline(m, horizon, sigma=0.0025):
    x = m["prices"][-1]
    rows = []
    for h in range(1, horizon + 1):
        spread = sigma * math.sqrt(h)
        rows.append([price6(x + z * spread) for z in Z])
    return rows

# ---------- helpers ----------
def trailing_returns(prices, window):
    """Minute-to-minute diffs over the trailing `window` minutes (or all available)."""
    n = len(prices)
    w = min(window, n - 1)
    if w <= 0:
        return []
    tail = prices[-(w + 1):]
    return [tail[i + 1] - tail[i] for i in range(len(tail) - 1)]

def realized_sigma(prices, window):
    diffs = trailing_returns(prices, window)
    if len(diffs) < 2:
        return None
    mean = sum(diffs) / len(diffs)
    var = sum((d - mean) ** 2 for d in diffs) / (len(diffs) - 1)
    return math.sqrt(var)

def logit(p, eps=1e-4):
    p = min(1 - eps, max(eps, p))
    return math.log(p / (1 - p))

def inv_logit(z):
    if z > 40:
        return 1.0
    if z < -40:
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))

# normal quantile via Acklam's algorithm (stdlib-safe, no scipy)
def norm_ppf(p):
    if p <= 0:
        return -8.0
    if p >= 1:
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
    p_high = 1 - p_low
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    elif p <= p_high:
        q = p - 0.5
        r = q*q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    else:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)

# ---------- Model 2: per-market sigma in raw price space ----------
def per_market_sigma_raw(m, horizon, window=180, floor_sigma=0.0008, cap_sigma=0.05, mult=1.0):
    prices = m["prices"]
    x = prices[-1]
    sig = realized_sigma(prices, window)
    if sig is None or sig <= 0:
        sig = floor_sigma
    sig = max(floor_sigma, min(cap_sigma, sig * mult))
    rows = []
    for h in range(1, horizon + 1):
        spread = sig * math.sqrt(h)
        rows.append([price6(x + z * spread) for z in Z])
    return rows

# ---------- Model 3: logit-space random walk with per-market sigma ----------
def logit_rw(m, horizon, window=180, floor_sigma_logit=0.02, cap_sigma_logit=1.2, mult=1.0,
             eps=1e-4):
    prices = m["prices"]
    x = prices[-1]
    z0 = logit(x, eps)
    # estimate sigma in logit space from trailing logit diffs
    tail_n = min(window, len(prices) - 1)
    if tail_n >= 2:
        tail = prices[-(tail_n + 1):]
        zdiffs = [logit(tail[i + 1], eps) - logit(tail[i], eps) for i in range(len(tail) - 1)]
        mean = sum(zdiffs) / len(zdiffs)
        var = sum((d - mean) ** 2 for d in zdiffs) / max(1, len(zdiffs) - 1)
        sig = math.sqrt(var)
    else:
        sig = floor_sigma_logit
    if sig <= 0 or math.isnan(sig):
        sig = floor_sigma_logit
    sig = max(floor_sigma_logit, min(cap_sigma_logit, sig * mult))
    rows = []
    for h in range(1, horizon + 1):
        spread = sig * math.sqrt(h)
        rows.append([price6(inv_logit(z0 + z * spread)) for z in Z])
    return rows

# ---------- Model 4: Student-t-like fat tail via inflated extreme quantiles ----------
def logit_rw_fattail(m, horizon, window=180, floor_sigma_logit=0.02, cap_sigma_logit=1.2,
                      mult=1.0, tail_mult=1.35, eps=1e-4):
    """Like logit_rw but widen the 0.05/0.95 quantiles more than 0.25/0.75 to mimic fat tails."""
    prices = m["prices"]
    x = prices[-1]
    z0 = logit(x, eps)
    tail_n = min(window, len(prices) - 1)
    if tail_n >= 2:
        tail = prices[-(tail_n + 1):]
        zdiffs = [logit(tail[i + 1], eps) - logit(tail[i], eps) for i in range(len(tail) - 1)]
        mean = sum(zdiffs) / len(zdiffs)
        var = sum((d - mean) ** 2 for d in zdiffs) / max(1, len(zdiffs) - 1)
        sig = math.sqrt(var)
    else:
        sig = floor_sigma_logit
    if sig <= 0 or math.isnan(sig):
        sig = floor_sigma_logit
    sig = max(floor_sigma_logit, min(cap_sigma_logit, sig * mult))
    zmults = [Z[0] * tail_mult, Z[1], Z[2], Z[3], Z[4] * tail_mult]
    rows = []
    for h in range(1, horizon + 1):
        spread = sig * math.sqrt(h)
        rows.append([price6(inv_logit(z0 + zz * spread)) for zz in zmults])
    return rows

# ---------- Model 6: calibrated-horizon diffusion (direct empirical h-step variance) ----------
def logit_z_series(prices, eps=1e-4):
    return [logit(p, eps) for p in prices]

def horizon_var(zs, h):
    """Variance of overlapping h-step differences in a logit series."""
    n = len(zs)
    if n <= h:
        return None
    diffs = [zs[i + h] - zs[i] for i in range(n - h)]
    if len(diffs) < 2:
        return None
    mean = sum(diffs) / len(diffs)
    var = sum((d - mean) ** 2 for d in diffs) / (len(diffs) - 1)
    return var, len(diffs)

def calibrated_diffusion(m, horizon, target_h=240, min_samples=40, fallback_window=120,
                          fallback_mult=0.35, floor_sigma_logit=0.02, cap_sigma_logit=1.2,
                          eps=1e-4, min_target_h_data=None):
    """
    Estimate the spread at horizon `target_h` directly from the market's own trailing
    history (overlapping target_h-step logit differences), when enough data exists.
    This sidesteps the sqrt(h) random-walk assumption and calibrates to the market's
    *actual* multi-step dispersion (captures mean-reversion / stickiness that a naive
    1-minute realized-vol * sqrt(h) extrapolation overstates).
    Falls back to a shrunk 1-minute-diff estimate, then to a fixed floor, when history
    is too short for a reliable direct estimate.
    """
    prices = m["prices"]
    x = prices[-1]
    z0 = logit(x, eps)
    zs = logit_z_series(prices, eps)
    n = len(zs)
    th = target_h if min_target_h_data is None else min_target_h_data

    sig_th = None
    hv = horizon_var(zs, min(th, n - 1)) if n > 1 else None
    if hv is not None:
        var, nsamp = hv
        used_h = min(th, n - 1)
        if nsamp >= min_samples and used_h >= 30:
            sig_th = math.sqrt(var / used_h)  # per-sqrt-minute sigma implied by this horizon's own dispersion

    if sig_th is None:
        # fallback: shrunk 1-minute realized sigma
        sig1 = realized_sigma(prices, fallback_window)
        if sig1 is not None and sig1 > 0:
            # convert raw-price sigma to logit-space approx via local derivative d(logit)/dp = 1/(p(1-p))
            deriv = 1.0 / max(1e-6, x * (1 - x))
            sig_th = sig1 * deriv * fallback_mult
        else:
            sig_th = floor_sigma_logit

    sig_th = max(floor_sigma_logit, min(cap_sigma_logit, sig_th))
    rows = []
    for h in range(1, horizon + 1):
        spread = sig_th * math.sqrt(h)
        rows.append([price6(inv_logit(z0 + z * spread)) for z in Z])
    return rows

# ---------- Model 7: sub-diffusive power-law scaling in logit space ----------
def logit_powerlaw(m, horizon, window=180, H=0.37, floor_sigma_logit=0.02, cap_sigma_logit=1.2,
                    eps=1e-4):
    """
    sigma(h) = sig1 * h^H instead of sig1 * sqrt(h) (H=0.5 would be pure random walk).
    H < 0.5 encodes the empirically observed sub-diffusion / mean-reversion of these
    thin prediction-market prices (variance-ratio test on the practice data shows
    Var(h-step logit diff) grows roughly like h^(2H) with H ~ 0.35-0.40, not h^1).
    """
    prices = m["prices"]
    x = prices[-1]
    z0 = logit(x, eps)
    sig1 = None
    tail_n = min(window, len(prices) - 1)
    if tail_n >= 2:
        tail = prices[-(tail_n + 1):]
        zdiffs = [logit(tail[i + 1], eps) - logit(tail[i], eps) for i in range(len(tail) - 1)]
        mean = sum(zdiffs) / len(zdiffs)
        var = sum((d - mean) ** 2 for d in zdiffs) / max(1, len(zdiffs) - 1)
        sig1 = math.sqrt(var)
    if sig1 is None or sig1 <= 0 or math.isnan(sig1):
        sig1 = floor_sigma_logit
    sig1 = max(floor_sigma_logit, min(cap_sigma_logit, sig1))
    rows = []
    for h in range(1, horizon + 1):
        spread = sig1 * (h ** H)
        rows.append([price6(inv_logit(z0 + z * spread)) for z in Z])
    return rows

# ---------- Model 5: logit RW with drift ----------
def logit_rw_drift(m, horizon, window=180, floor_sigma_logit=0.02, cap_sigma_logit=1.2,
                    mult=1.0, drift_window=60, drift_damp=0.3, eps=1e-4):
    prices = m["prices"]
    x = prices[-1]
    z0 = logit(x, eps)
    tail_n = min(window, len(prices) - 1)
    if tail_n >= 2:
        tail = prices[-(tail_n + 1):]
        zdiffs = [logit(tail[i + 1], eps) - logit(tail[i], eps) for i in range(len(tail) - 1)]
        mean = sum(zdiffs) / len(zdiffs)
        var = sum((d - mean) ** 2 for d in zdiffs) / max(1, len(zdiffs) - 1)
        sig = math.sqrt(var)
    else:
        sig = floor_sigma_logit
    if sig <= 0 or math.isnan(sig):
        sig = floor_sigma_logit
    sig = max(floor_sigma_logit, min(cap_sigma_logit, sig * mult))
    dn = min(drift_window, len(prices) - 1)
    drift_per_min = 0.0
    if dn >= 2:
        z_now = logit(prices[-1], eps)
        z_then = logit(prices[-1 - dn], eps)
        drift_per_min = (z_now - z_then) / dn * drift_damp
    rows = []
    for h in range(1, horizon + 1):
        spread = sig * math.sqrt(h)
        center = z0 + drift_per_min * h
        rows.append([price6(inv_logit(center + z * spread)) for z in Z])
    return rows
