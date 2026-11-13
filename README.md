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
probability mass on the clamped side).

```
x        = the market's last price in the case (price at the checkpoint)
z0       = logit(x)
sigma1   = std. dev. of 1-minute logit-price differences over the market's own
           trailing 180 minutes of history (or everything it has, if shorter),
           floored at 0.03 and capped at 1.2
spread(h)= sigma1 * h ** 0.30          (h = minutes ahead, 1..240)
q_tau(h) = clamp(inv_logit(z0 + Phi^-1(tau) * spread(h)), 0, 1), rounded to 6dp
```

The `h ** 0.30` term (instead of the pure-random-walk `h ** 0.5`) encodes a
sub-diffusion effect measured directly on the event's own practice history: a
variance-ratio test (`Var(price change over h minutes)` vs. `h` times the 1-minute
variance, in logit space, pooled over 120 market/case pairs with 300+ minutes of
history) falls from ~0.86 at h=5 to ~0.22 at h=240 — these thin, low-information
candidate markets drift and then partially revert rather than compounding 1-minute
noise all the way out to 4 hours. Fitting a power law to that decay gives an exponent
around 0.3-0.4; the backtested loss is flat across H in roughly [0.26, 0.34], so 0.30
is not a knife-edge fit. The 0.03 floor on `sigma1` lands almost exactly on the
cross-sectional median of markets' own (unfloored) realized sigma, which is itself
evidence it's a sensible shrinkage prior for markets whose short window just happened
to be quiet, not an arbitrary constant.

A small drift term and inflated ("fat-tailed") extreme quantiles were both tried and
both made the backtested loss worse on average, so neither is used — pinball loss is
a proper scoring rule, and widening or skewing a forecast the data doesn't support is
a net loss, not a free hedge.

Backtested against real Polymarket price history fetched for the six practice cases
(p00-p05), this model scores a mean pinball loss of **0.272584**, about **25%** better
than the house's constant-sigma random-walk baseline (0.363589) on the same ground
truth.

## Run it

```
mkdir -p out && MAKEFUN_INPUT_DIR=<input dir> MAKEFUN_OUTPUT_DIR=out python main.py
```

For example, from this directory, against the kit's practice cases:

```
mkdir -p out && MAKEFUN_INPUT_DIR=fixtures/input MAKEFUN_OUTPUT_DIR=out python main.py
```
