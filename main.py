"""
polymarket-nvidia-ceo-forecast-01 entry (round 3).

Forecasts the Yes price of every market of Polymarket's "Next CEO of NVIDIA?" event,
minute by minute, for horizon_minutes (240) minutes after each case's checkpoint.

Model: logit-space sub-diffusive random walk with a per-market realized-volatility
estimate (round 1), a cross-market sum-to-one shrinkage adjustment (round 2) applied
to each market's starting price before the per-market diffusion, and two further
empirically-grounded, deliberately-damped corrections (round 3): a level-dependent
skew on the quantile shape, and an asymmetric short-term momentum/reversal drift.

  S0       = sum of every market's last price in the case (the checkpoint)
  k0       = 1 + LAMBDA * (1 / S0 - 1), clamped to [K0_MIN, K0_MAX]
  x        = the market's last price in the case, nudged toward the renormalized
             value: x_adj = clamp(x * k0, eps, 1 - eps)
  z0       = logit(x_adj) = ln(x_adj / (1 - x_adj))
  sigma1   = the sample standard deviation of 1-minute logit-price differences,
             over the market's own trailing WINDOW minutes of *unadjusted* history
             (or all of it, when the market has less), floored and capped
  lag      = z[now] - z[now - REV_LOOKBACK]   (unadjusted logit-space move)
  center(h)= z0 - REV_ALPHA * max(0, lag) * h ** H     (reversal drift; see below)
  gamma    = clamp(SKEW_G0 - SKEW_G1 * x_adj, -SKEW_CAP, SKEW_CAP)   (level skew)
  spread(h)= sigma1 * h ** H           (h = minutes ahead, h = 1..horizon)
  d_tau(h) = Phi^-1(tau) * spread(h), then widened by (1+gamma) if >=0 else (1-gamma)
  q_tau(h) = clamp(inv_logit(center(h) + d_tau(h)), 0, 1), rounded to 6dp

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

- A small drift term and fat-tailed (inflated 5%/95%) quantiles were both tried in
  round 1/2 and made the backtested loss *worse* on average there -- those were a
  constant, level-independent drift and a flat tail multiplier, with no empirical
  basis. Round 3 revisits both, grounded this time in direct measurement:

  * Level-dependent skew: partitioning 1-minute logit-price diffs across all 6
    practice cases' markets by the *starting price level* shows markets below
    about 0.15 have strong, robust (same sign in every one of the 6 practice
    cases individually) *positive* skew -- rare large up-jumps are more extreme
    than rare large down-moves -- while markets from about 0.15 to 0.5 show
    equally robust *negative* skew, growing with price level. There is almost no
    practice data above ~0.6 (these are long-shot-heavy candidate markets), so
    SKEW_CAP bounds how far this is extrapolated toward 1, deliberately
    conservative about a region we have not observed (see
    research/skew_reversal_analysis.py).

  * Short-term reversal: the correlation between a market's recent L-minute
    logit move and its subsequent move is consistently negative (~-0.15 to
    -0.30) across many lookback/horizon pairs pooled over all practice
    markets, and -- unlike a transient bounce -- does not decay away over the
    full 4-hour horizon. It is also clearly asymmetric: large recent upward
    moves predict strong reversion; large recent downward moves predict almost
    none. REV_ALPHA and REV_LOOKBACK are both set well below their raw
    backtest-optimal values: the 6 practice cases are checkpoints 4 hours apart
    on ONE continuous price history (same event, same start), so their
    pre-checkpoint windows overlap almost entirely -- a leave-one-case-out
    check here is weaker evidence of generalization than round 2's cross-market
    lambda was, and the full-sample optimum (lookback ~120-150, alpha ~0.06-0.08)
    showed much less stable, more case-dependent behavior (hurting 2 of 6 cases
    meaningfully) than the damped values used here (lookback=45, alpha=0.03),
    which improve 5 of 6 practice cases and only barely touch the 6th. See
    research/zero_inflation_analysis.py, research/skew_reversal_analysis.py,
    research/models_round3.py, research/run_sweep3.py and research/loco3.py for
    the full investigation, including a third candidate -- an explicit
    zero-inflation mixture (point mass at "no change" blended with the
    continuous diffusion) -- that was tested and dropped: real, large,
    independently-measured structure (P(unchanged) falls from ~0.88 at h=1 to
    ~0.12 at h=240), but explicit mixture modeling of it was only neutral
    (3 of 6 held-out cases) in leave-one-case-out testing, i.e. it did not add
    anything beyond what the existing sub-diffusion exponent already captures.

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

# -- round 3 additions (see docstring + research/ for how these were chosen) --
SKEW_G0 = 0.10            # gamma(x) = clamp(SKEW_G0 - SKEW_G1 * x, -SKEW_CAP, SKEW_CAP)
SKEW_G1 = 1.40            # level-dependent quantile skew: widens the upside tail
SKEW_CAP = 0.35           # and narrows the downside when gamma > 0 (low x), and
                          # vice versa for gamma < 0 (x above ~0.07); capped to avoid
                          # extrapolating past the ~0-0.6 price range we have data for

REV_ALPHA = 0.03          # short-term reversal drift strength (damped well below the
REV_LOOKBACK = 45         # raw backtest optimum -- see docstring); minutes of lookback
                          # used to measure the market's most recent logit move


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


def skew_gamma(x, g0=SKEW_G0, g1=SKEW_G1, cap=SKEW_CAP):
    """Level-dependent skew factor: positive (widen upside, narrow downside) for
    low x, negative (widen downside, narrow upside) for higher x; see docstring."""
    g = g0 - g1 * x
    return max(-cap, min(cap, g))


def recent_move(prices, lookback=REV_LOOKBACK):
    """Market's own realized logit-space move over its trailing `lookback` minutes
    (or everything it has, if shorter), from its *unadjusted* price history."""
    n = len(prices)
    L = min(lookback, n - 1)
    if L < 1:
        return 0.0
    return logit(prices[-1]) - logit(prices[-1 - L])


def forecast_market(prices, horizon, z_levels, k0=1.0):
    """The forecast for one market: `horizon` rows of len(z_levels) quantiles.

    `k0` is the case-wide cross-market rescale factor (see cross_market_k0): it
    nudges only the starting price used for z0, not the volatility estimate,
    which is still computed from this market's own unadjusted price history.

    Round 3 adds two further corrections on top of round 2's center/spread:
    a level-dependent skew of the quantile shape (gamma, from x_adj) and an
    asymmetric reversal drift applied to the center, proportional to the size
    of the market's own most recent upward move (not applied to downward
    moves -- see docstring for why only one direction showed a robust effect).
    """
    x = current_price(prices)
    x_adj = min(1.0 - EPS, max(EPS, x * k0))
    z0 = logit(x_adj)
    sigma1 = estimate_sigma1(prices)
    gamma = skew_gamma(x_adj)
    lag = recent_move(prices)
    signed_lag = max(0.0, lag)
    rows = []
    for h in range(1, horizon + 1):
        spread = sigma1 * (h ** H_EXPONENT)
        center = z0
        if REV_ALPHA != 0.0 and signed_lag != 0.0:
            center -= REV_ALPHA * signed_lag * (h ** H_EXPONENT)
        row = []
        for z in z_levels:
            d = z * spread
            if gamma != 0.0:
                d = d * (1.0 + gamma) if d >= 0.0 else d * (1.0 - gamma)
            row.append(price6(inv_logit(center + d)))
        rows.append(row)
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
