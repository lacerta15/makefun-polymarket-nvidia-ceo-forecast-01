"""
polymarket-nvidia-ceo-forecast-01 entry (round 2).

Forecasts the Yes price of every market of Polymarket's "Next CEO of NVIDIA?" event,
minute by minute, for horizon_minutes (240) minutes after each case's checkpoint.

Model: logit-space sub-diffusive random walk with a per-market realized-volatility
estimate (round 1), plus a cross-market sum-to-one shrinkage adjustment (round 2)
applied to each market's starting price before the per-market diffusion.

  S0       = sum of every market's last price in the case (the checkpoint)
  k0       = 1 + LAMBDA * (1 / S0 - 1), clamped to [K0_MIN, K0_MAX]
  x        = the market's last price in the case, nudged toward the renormalized
             value: x_adj = clamp(x * k0, eps, 1 - eps)
  z0       = logit(x_adj) = ln(x_adj / (1 - x_adj))
  sigma1   = the sample standard deviation of 1-minute logit-price differences,
             over the market's own trailing WINDOW minutes of *unadjusted* history
             (or all of it, when the market has less), floored and capped
  spread(h)= sigma1 * h ** H           (h = minutes ahead, h = 1..horizon)
  q_tau(h) = clamp(inv_logit(z0 + Phi^-1(tau) * spread(h)), 0, 1), rounded to 6dp

Why the cross-market term, briefly (see research/cross_market_analysis.py and
research/run_sweep_xmarket.py + research/loco_xmarket.py for the full writeup):

- This event's markets are an (incomplete) subset of outcomes of one Polymarket
  negRisk categorical event ("who becomes the next CEO"), so in a frictionless
  market sum(price) across all outcomes would be exactly 1. The listed subset
  (19-24 of the event's 107 candidate markets) carries nearly all of the
  probability mass, so its sum should be *close* to 1, with deviations coming
  from thin-liquidity / bid-ask noise on the shared negRisk basket rather than
  missing mass. Empirically, the 6 practice cases' checkpoint sums range from
  0.81 to 1.09 -- a real, case-wide mispricing signal that round 1 ignored
  entirely (it forecasts every market from its own history alone).

- Checking what actually happens next (not just assuming reversion): the
  realized sum over each case's following 4 hours does, on average, move back
  toward 1 (the gap shrinks to roughly a third of its checkpoint size, pooled
  over the 6 cases) but the effect is noisy and case-dependent -- one practice
  case overshoots *past* 1 in the other direction. A full forced renormalization
  (shrinkage weight 1.0, i.e. rescale every price so the case sums to exactly 1)
  is NOT what backtests best; a partial, damped nudge is. LAMBDA = 0.30 was
  chosen deliberately below the raw backtested optimum (~0.5-0.6) because that
  higher optimum is driven almost entirely by one outlier practice case with an
  unusually large, idiosyncratic repricing swing -- excluding that single case,
  the other 5 cases' own optimum sits at LAMBDA ~ 0.2-0.3, which is the value
  used here. This is deliberately conservative: it captures the structural
  sum-to-one pull (which should hold in any future case, since it comes from
  the event's own market design) without curve-fitting to one case's noise.

- Working in logit space (rather than raw price) makes the diffusion symmetric and
  naturally respects the [0,1] boundary: a market pinned near 0 or 1 gets properly
  skewed, tightly-clustered quantiles instead of a symmetric band that a raw-price
  model has to clamp (which silently drops probability mass on the clamped side).

- The random-walk baseline's sqrt(h) scaling (H = 0.5) assumes prices diffuse like
  an unpredictable walk at every horizon. A variance-ratio test on this event's own
  practice history -- Var(logit price change over h minutes) versus h times the
  1-minute variance -- shows that ratio falls from ~0.86 at h=5 to ~0.22 at h=240
  (pooled over 120 market/case pairs with 300+ minutes of history). That is a
  strong, monotonic sub-diffusion signature: these thin, low-information markets
  often drift and then partially revert, rather than compounding 1-minute noise
  all the way out to 4 hours. Fitting a power law to that ratio gives an exponent
  H around 0.3-0.4; H = 0.30 is used here (backtested loss is flat across
  H in roughly [0.26, 0.34], so this is not a knife-edge fit, and remains the
  optimum after adding the cross-market term -- re-checked, not re-assumed).

- A small drift term and fat-tailed (inflated 5%/95%) quantiles were both tried and
  both made the backtested loss *worse* on average, so neither is used: pinball
  loss is a proper scoring rule, and widening/skewing the forecast when the data
  does not support it is a net loss, not a hedge.

No network access, no randomness, no third-party packages: standard library only.
"""

import json
import math
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Scoring-contract constants
# ---------------------------------------------------------------------------

EXPECTED_QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]

# ---------------------------------------------------------------------------
# Model parameters (see module docstring + research/ for how these were chosen)
# ---------------------------------------------------------------------------

WINDOW = 180              # minutes of trailing history used to estimate 1-min sigma
H_EXPONENT = 0.30         # sub-diffusion exponent: spread(h) = sigma1 * h**H_EXPONENT
FLOOR_SIGMA_LOGIT = 0.03  # minimum 1-minute logit-sigma (guards dead-flat markets)
CAP_SIGMA_LOGIT = 1.2     # maximum 1-minute logit-sigma (guards pathological inputs)
EPS = 1e-6                # logit clipping epsilon, keeps prices at 0 or 1 finite

XMARKET_LAMBDA = 0.30     # cross-market sum-to-one shrinkage weight (0 = round 1,
                          # 1 = full renormalization to sum exactly 1); see docstring
K0_MIN = 0.5              # safety clamp on the per-case price-rescale factor k0,
K0_MAX = 2.0              # guards against pathological S0 in future cases we never saw
S0_FLOOR = 1e-3           # floor on the cross-sectional sum before taking 1/S0


# ---------------------------------------------------------------------------
# Small math helpers (stdlib only)
# ---------------------------------------------------------------------------

def logit(p: float, eps: float = EPS) -> float:
    p = min(1.0 - eps, max(eps, p))
    return math.log(p / (1.0 - p))


def inv_logit(z: float) -> float:
    if z > 40.0:
        return 1.0
    if z < -40.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))


def norm_ppf(p: float) -> float:
    """Inverse CDF of the standard normal (Acklam's algorithm, double precision)."""
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


def price6(x: float) -> float:
    """x clamped to [0,1] and rounded to 6 decimals (0, never -0)."""
    x = min(1.0, max(0.0, x))
    return round(x * 1e6) / 1e6 + 0.0


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

def estimate_sigma1(prices, window=WINDOW, floor=FLOOR_SIGMA_LOGIT, cap=CAP_SIGMA_LOGIT, eps=EPS):
    """Sample std of 1-minute logit-price differences over the trailing `window`
    minutes of this market's own history (or everything it has, if shorter)."""
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


def current_price(prices):
    """The market's current (checkpoint) price, or 0.5 if it has no history at all."""
    return prices[-1] if prices else 0.5


def cross_market_k0(markets, lam=XMARKET_LAMBDA):
    """Case-wide price-rescale factor nudging every market's current price toward
    the renormalized value that would make the case's prices sum to 1. lam=0.30
    is a damped (not full) shrinkage -- see the module docstring for why."""
    s0 = sum(current_price(m["prices"]) for m in markets)
    if s0 <= S0_FLOOR:
        return 1.0
    k0 = 1.0 + lam * (1.0 / s0 - 1.0)
    return max(K0_MIN, min(K0_MAX, k0))


def forecast_market(prices, horizon, z_levels, k0=1.0):
    """The forecast for one market: `horizon` rows of len(z_levels) quantiles.

    `k0` is the case-wide cross-market rescale factor (see cross_market_k0): it
    nudges only the starting price used for z0, not the volatility estimate,
    which is still computed from this market's own unadjusted price history.
    """
    x = current_price(prices)
    x_adj = min(1.0 - EPS, max(EPS, x * k0))
    z0 = logit(x_adj)
    sigma1 = estimate_sigma1(prices)
    rows = []
    for h in range(1, horizon + 1):
        spread = sigma1 * (h ** H_EXPONENT)
        rows.append([price6(inv_logit(z0 + z * spread)) for z in z_levels])
    return rows


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def main():
    input_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    output_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    case_files = sorted(p for p in input_dir.iterdir() if p.name.endswith(".in"))
    for case_path in case_files:
        case_id = case_path.name[: -len(".in")]
        with open(case_path, "r", encoding="utf-8") as f:
            case = json.load(f)

        quantiles = case["quantiles"]
        z_levels = [norm_ppf(tau) for tau in quantiles]
        horizon = case["horizon_minutes"]
        k0 = cross_market_k0(case["markets"])

        forecasts = []
        for market in case["markets"]:
            rows = forecast_market(market["prices"], horizon, z_levels, k0=k0)
            forecasts.append({"id": market["id"], "q": rows})

        out_path = output_dir / f"{case_id}.out"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"forecasts": forecasts}))
            f.write("\n")


if __name__ == "__main__":
    main()
