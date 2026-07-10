# RESULTS

What the system actually did on the sealed test, what that does and does not tell us,
and the honest caveats. The reasoning behind the design is in DECISIONS; the pipeline
is in MECHANICS. Seasons are named by starting year. British English.

## The sealed 2025 result

2025 (the 2025-26 season) was the single one-shot out-of-sample test for MVP, DPOY and
ROTY: never trained on, never tuned against, run once. On a 3,000 pooled bankroll with
the shipped configuration (shrunk book weights, no-trade region on, asymmetric trim on):

- Pooled profit +$460, a 15.3% return on the static bankroll (11.4% on the
  mark-to-market equity curve).
- Sharpe 0.71, Sortino 1.23, maximum drawdown -11.7%, over 216 trading days.
- By book: MVP +$152 over 24 transactions, DPOY +$116 over 11, ROTY +$192 over 13.

All three books were profitable on the held-out season. The trade count is low by
design: this is a low-frequency, seasonal strategy that holds through the no-trade band
and acts only when the edge clears the round-trip cost.

## Why the shipped allocation was chosen

The bankroll can be split across the three books three ways. All three were run on 2025
at the same 3,000 bankroll with the region on, and this is the comparison that decided
the shipped configuration:

| Split | Pooled PnL | Sharpe | Max drawdown |
|-------|-----------:|-------:|-------------:|
| Equal weight | +$478 | 0.61 | -11.8% |
| Shrunk BSS (shipped) | +$460 | 0.71 | -11.7% |
| Raw BSS | +$433 | 0.48 | -16.1% |

Raw skill-weighting was dominated on every axis: it concentrated too hard on the
highest-skill book and gave up both return and stability. Equal weight earned about $18
more than the shrunk split but at a materially worse Sharpe (0.61 against 0.71). The
shipped choice is the shrunk skill-weighting, which trades that small amount of headline
return for the better risk-adjusted return and the slightly better drawdown, by
down-weighting the least-calibrated book and up-weighting the best without staking the
whole split on ten noisy seasons. Optimising for risk-adjusted return over raw dollars
is the deliberate call.

## Per-book skill and what it means

The weights come from each book's out-of-sample skill (trailing ten seasons to 2025):

- ROTY 0.619, the most predictable. Rookie-of-the-year races usually have clear
  statistical separation, and the model prices them well.
- MVP 0.346, moderately predictable. The race is legible but narrative-heavy, and the
  occasional stickiness below hurts.
- DPOY 0.185, the least predictable, and this is structural.

## The DPOY ceiling

DPOY is a hard problem, not a broken model. The award rewards defensive impact that the
box score measures poorly (positioning, rim deterrence, switchability, matchup
difficulty), so the features are genuinely less informative than for MVP or ROTY. The
low skill score reflects that ceiling honestly rather than a fixable defect. The
important finding is that the model's honest uncertainty on contested DPOY races is
still tradeable: DPOY returned +$116 on the sealed season precisely because pricing
uncertainty correctly, and fading a market that is overconfident on a narrative
favourite, is itself an edge. This is why DPOY was never gated out of live capital.

## Residual drags

The main identified drag on the sealed season was a value-share-versus-win stickiness on
a faded favourite: the model's vote-share view held a fading MVP frontrunner (Jokic) too
long as the market moved on, costing roughly -$85 on the MVP book. This is a known
limitation of pricing vote share rather than win probability directly, and is recorded
as such in DECISIONS. It did not turn the book negative, but it is the clearest place
where the fair-value view lagged.

## What this result does and does not establish

The direction and the relative ranking are the trustworthy parts. One sealed season is a
single draw, so the honest reading is:

- The point estimate (+$460) is the least trustworthy number here. Treat it as
  indicative, not as an expected annual return. A single season can flatter or punish a
  sound process by luck.
- The relative rankings are more robust than the absolute figure: shrunk over equal over
  raw on risk-adjusted terms, all three books profitable, and the region and trim
  earning their place, are conclusions drawn from the same single season but are
  structural rather than a single lucky trade.
- The durable finding is the principle, not the figure: pricing the vote as a
  distribution and fading retail overreaction in an eventually-objective market produced
  a positive, reasonably stable return out of sample across three independent books. The
  dollar figure is one realisation of that principle.

Two standing constraints bound any forward expectation. Capacity is small: Polymarket
award books are thin, so the strategy cannot absorb large size without moving the price
it is trading against. And the frequency is low: a season offers a limited number of
genuine edges (216 days, roughly 48 pooled transactions), so the sample of independent
bets is small and will stay small until more awards are added. The correct posture is to
paper-trade a live season before staging real capital, and to judge the system over
several seasons rather than on this one.

## A note on 2024

2024 (2024-25) was the development and debugging season, used repeatedly while building
the sizer, the region, the risk adjustment and the DPOY handling. Its numbers are
in-sample by construction and are not reported here as evidence; only 2025 is a clean
test.
