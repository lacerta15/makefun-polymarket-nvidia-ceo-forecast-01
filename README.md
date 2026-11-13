# polymarket-nvidia-ceo-forecast-01 entry

Forecasts the Yes price of every market of Polymarket's "Next CEO of NVIDIA?" event,
minute by minute, for 240 minutes after each case's checkpoint, as required by the
`polymarket-nvidia-ceo-forecast-01` competition on make.fun.

## What it does

`main.py` reads every `*.in` case file in `$MAKEFUN_INPUT_DIR` (default `/input`)
and writes a matching `<case>.out` file to `$MAKEFUN_OUTPUT_DIR` (default `/output`),
each holding five quantiles (0.05 / 0.25 / 0.5 / 0.75 / 0.95) of every market's Yes
price for every one of the next 240 minutes. It is pure, deterministic, offline
arithmetic: standard library only (`math`, `json`, `os`, `pathlib`), no network, no
randomness, no third-party packages.

## The model

For each market, work in logit space `z = ln(p / (1-p))` rather than raw price so the
forecast naturally respects the `[0,1]` boundary and skews correctly near 0 or 1,
instead of symmetric bands that a raw-price model has to clamp (silently discarding
probability mass on the clamped side). On top of that per-market diffusion (round 1),
round 2 adds a cross-market sum-to-one shrinkage term: every market's starting price
is nudged toward the value that would make the whole case's prices sum to 1, before
diffusing it forward. Round 3 adds two further, deliberately-damped corrections: a
level-dependent skew of the quantile shape, and an asymmetric short-term
reversal/momentum drift.

```
S0       = sum of every market's last price in the case (the checkpoint)
k0       = 1 + 0.30 * (1 / S0 - 1), clamped to [0.5, 2.0]
x        = the market's last price in the case (price at the checkpoint)
x_adj    = clamp(x * k0, eps, 1 - eps)
z0       = logit(x_adj)
sigma1   = std. dev. of 1-minute logit-price differences over the market's own
           trailing 180 minutes of *unadjusted* history (or everything it has, if
           shorter), floored at 0.03 and capped at 1.2
lag      = z[now] - z[now - 45]         (own unadjusted 45-minute logit move)
center(h)= z0 - 0.03 * max(0, lag) * h ** 0.30     (asymmetric reversal drift)
gamma    = clamp(0.10 - 1.40 * x_adj, -0.35, 0.35)       (level-dependent skew)
spread(h)= sigma1 * h ** 0.30          (h = minutes ahead, 1..240)
d_tau(h) = Phi^-1(tau) * spread(h), widened by (1+gamma) if >=0 else (1-gamma)
q_tau(h) = clamp(inv_logit(center(h) + d_tau(h)), 0, 1), rounded to 6dp
```

**Cross-market term (round 2).** This event's markets are an (incomplete) subset
of outcomes of one Polymarket negRisk categorical event ("who becomes the next
CEO") — 19 to 24 of the event's 107 candidate markets per case. In a frictionless
market the Yes prices of all outcomes of a negRisk event sum to 1; since the
listed subset carries nearly all of the probability mass, its sum should be
*close* to 1, with deviations coming from thin-liquidity / bid-ask noise on the
shared basket rather than missing mass. On the 6 practice cases, checkpoint sums
actually range from 0.81 to 1.09 — a real, case-wide mispricing signal that round
1 ignored completely (it forecasts every market from its own history alone,
independent of every other market in the same case).

Checking what actually happens next (not just assuming reversion):
`research/cross_market_analysis.py` shows the realized sum over each case's
following 4 hours moves back toward 1 on average (the gap shrinks to roughly a
third of its checkpoint size, pooled over the 6 cases), but noisily and
case-dependently — one practice case's sum overshoots *past* 1 the other way.
Backtesting a family of shrinkage strengths (`research/run_sweep_xmarket.py`)
shows a full forced renormalization (weight 1.0, rescale every case to sum
exactly 1) is *not* optimal — a damped, partial nudge is, and the naive
backtest-wide optimum (~0.5-0.6) turns out to be driven almost entirely by one
outlier practice case with an unusually large, idiosyncratic repricing swing. A
leave-one-case-out check (`research/loco_xmarket.py`) shows that excluding that
one case, the other 5 cases' own optimum sits at a shrinkage weight of about
0.2-0.3. **0.30** is used in `main.py` — deliberately on the conservative edge of
that range, because it is a structural property of the event's own market design
(and should hold in future cases) rather than a fit to one case's noise. A safety
clamp (`k0` in `[0.5, 2.0]`) guards against pathological cross-sectional sums in
cases we have not seen.

The `h ** 0.30` term (instead of the pure-random-walk `h ** 0.5`) encodes a
sub-diffusion effect measured directly on the event's own practice history: a
variance-ratio test (`Var(price change over h minutes)` vs. `h` times the 1-minute
variance, in logit space, pooled over 120 market/case pairs with 300+ minutes of
history) falls from ~0.86 at h=5 to ~0.22 at h=240 — these thin, low-information
candidate markets drift and then partially revert rather than compounding 1-minute
noise all the way out to 4 hours. Fitting a power law to that decay gives an exponent
around 0.3-0.4; the backtested loss is flat across H in roughly [0.26, 0.34], so 0.30
is not a knife-edge fit (re-checked after adding the cross-market term: still the
optimum). The 0.03 floor on `sigma1` lands almost exactly on the cross-sectional
median of markets' own (unfloored) realized sigma, which is itself evidence it's a
sensible shrinkage prior for markets whose short window just happened to be quiet,
not an arbitrary constant.

A level-independent drift term and a flat fat-tail multiplier were both tried in
round 1/2 and made the backtested loss worse on average there — pinball loss is a
proper scoring rule, and widening or skewing a forecast the data doesn't support is
a net loss, not a free hedge. Round 3 revisits both ideas, but grounded in direct
measurement instead of guessing a shape:

**Level-dependent skew.** `research/skew_reversal_analysis.py` partitions 1-minute
logit-price diffs (pooled over all 6 practice cases' markets) by the *starting price
level*: markets below about 0.15 show strong, robust (same sign in every one of the
6 practice cases checked individually) *positive* skew — rare large up-jumps are more
extreme than rare large down-moves — while markets from about 0.15 to 0.5 show
equally robust *negative* skew, growing with price level. There is almost no practice
data above ~0.6 (these are long-shot-heavy candidate markets), so the skew factor
`gamma` is capped at 0.35 to avoid extrapolating with false confidence into a price
region we have never observed — the same clamping discipline round 2 used for `k0`.

**Short-term reversal.** The same file measures the correlation between a market's
recent L-minute logit move and its subsequent move: consistently negative (~-0.15 to
-0.30) across many lookback/horizon combinations, and — unlike a transient bounce —
it does not decay away over the full 4-hour forecast horizon. It is also clearly
asymmetric: large recent *upward* moves predict strong reversion; large recent
*downward* moves predict almost none, so the drift term only fires on `max(0, lag)`.
The 6 practice cases are checkpoints exactly 4 hours apart on *one* continuous price
history (same event, same start timestamp), so their pre-checkpoint windows overlap
almost entirely — a leave-one-case-out check here (`research/loco3.py`) is weaker
evidence of generalization than round 2's cross-market lambda had, and the raw
backtest-optimal region (lookback ~120-150 minutes, alpha ~0.06-0.08) turned out to
be noticeably less stable across held-out cases (it meaningfully hurt 2 of the 6)
than the damped values actually shipped (lookback=45, alpha=0.03), which improve 5 of
6 practice cases and only barely touch the 6th.

A third candidate — modeling the point-mass-at-"no change" structure explicitly as a
zero-inflation mixture on top of the continuous logit-diffusion — was investigated in
`research/zero_inflation_analysis.py` and is real: pooled across the practice
markets' own history, P(price unchanged) falls from ~0.88 at h=1 minute to ~0.12 at
h=240 minutes, and the post-checkpoint realized ground truth confirms the same shape.
But explicitly mixing a point mass into the quantile formula, calibrated per-market
and decayed with a globally-fit power law, was only *neutral* under leave-one-case-out
testing (3 of 6 held-out cases improved, 3 worsened, near-zero mean effect) — the
existing sub-diffusion exponent already captures this structure about as well on its
own, so the mixture was not shipped (see `research/models_round3.py` /
`research/run_sweep3.py` / `research/loco3.py` for the full comparison).

Backtested against real Polymarket price history fetched for the six practice cases
(p00-p05), this model scores a mean pinball loss of **0.258945** — **28.78%** better
than the house's constant-sigma random-walk baseline (0.363589), **2.53%** better than
round 2's 0.265667 (confirmed live on make.fun's grader twice, exactly matching the
offline backtest both times), and **5.0%** better than round 1's 0.272584 — on the
same ground truth.

## Run it

```
mkdir -p out && MAKEFUN_INPUT_DIR=<input dir> MAKEFUN_OUTPUT_DIR=out python main.py
```

For example, from this directory, against the kit's practice cases:

```
mkdir -p out && MAKEFUN_INPUT_DIR=fixtures/input MAKEFUN_OUTPUT_DIR=out python main.py
```
