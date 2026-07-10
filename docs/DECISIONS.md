# DECISIONS

Why the NBA Awards Trader is built the way it is. This document is the reasoning
trail: the thesis, the choices that define the system, and, just as importantly,
the approaches that were tried and rejected. It is meant to be readable without
the code open, but it does not shy away from the technical substance. For how the
pipeline runs, see MECHANICS; for what it earned, see RESULTS.

## The thesis

Polymarket's voter-decided NBA award markets are dominated by retail traders, and
retail prices move more on news than the true change in outcome probability
warrants. A strong defensive game, a viral highlight, a media narrative: each
shifts the price further than it shifts the actual chance of winning the award.
Because these awards resolve on an objective event months later (the vote is
counted), a patient trader who can price the eventual vote more accurately than the
crowd can fade the overreaction and hold to resolution.

The critical framing, which everything else follows from, is that the model
predicts voter behaviour, not merit. It does not try to decide who deserves MVP.
It estimates the probability that each candidate wins the vote, given the stats and
the state of the race at a point in time, trained against decades of historical
vote shares. Narrative matters to voters, so narrative is part of the thing being
predicted, not a separate signal bolted on.

Two linked sources of edge are posited. The first is narrative overreaction as
above. The second is longshot bias: retail utility curves overpay for convex
payouts, so favourites are systematically underpriced and longshots overpriced.
The two reinforce each other on underdog storylines. The book trades both as one
strategy: fade the mispricing where fair value and market price diverge most.

## Model architecture

The core is a LightGBM gradient-boosted model with a custom grouped Plackett-Luce
objective. The grouping is what matters: candidates within a single race are not
independent, their vote shares sum to one by construction, so the model computes a
softmax across all candidates in a race and compares it to the observed vote shares
through a multinomial cross-entropy loss. Each candidate's gradient depends on
every other candidate in the race through the softmax denominator, so the model
learns the joint structure of a race rather than scoring names in isolation.

Uncertainty is captured with a bootstrap ensemble: many models trained on
resampled data, resampling whole races rather than individual rows so the grouped
structure survives each resample. The spread across the ensemble is the model's
epistemic uncertainty, and it feeds the risk layer rather than the point estimate.

Training rows are multi-snapshot: each candidate contributes one row per snapshot
date through a season, with features computed as of that date, all predicting the
same end-of-season vote share. This teaches the model how a race looks at different
points in its evolution, which is exactly what live trading needs.

Validation is walk-forward by season: train on seasons up to year T, predict T+1,
roll forward, never splitting a season across train and test. The eventual sealed
test on 2025 was held apart from all of this.

## What was rejected, and why

The rejections are as load-bearing as the inclusions, because each one was a
plausible idea that the evidence killed.

Narrative as a fair-value input is dead. This was the biggest surprise. An entire
natural-language pipeline was built (news and social ingestion, embeddings,
clustering into narrative archetypes, sentiment, entailment scoring) on the thesis
that narrative would sharpen the vote prediction. It did not. Four independent
lines of evidence converged: the narrative residual was near-null, no narrative
feature survived selection into the top gain, per-frame narrative correlations
simply re-encoded the underlying stats, and a placebo test confirmed no genuine
signal. The stats already carry the narrative, because voters' narratives are
mostly downstream of the box score. Narrative was removed from the scoring path
entirely. The one weak surviving signal (a partial correlation between the
MVP narrative-minus-stats residual and outcomes) belongs in a prospective trading
overlay, not in fair value, and is not currently used.

Bankroll-Kelly leverage was rejected. Sizing each book off the pooled bankroll,
rather than a fixed per-book budget, was tested and found to be leverage, not
alpha: it roughly doubled volatility and deepened the drawdown by over three times
while barely lifting return. The capacity constraint on this strategy is market
depth, not exhausted edge, so the way to grow is more markets (breadth), not bigger
bets (leverage). The book is deliberately run at roughly one-third of full Kelly.

Blanket smoothing of the fair-value estimate was rejected. Week-to-week churn in
the target looked like noise worth smoothing, but decomposition showed the churn is
an execution-layer phenomenon (the field re-ranks) rather than genuine fair-value
spikes, so smoothing belongs at the execution layer as hysteresis, not on the
model output.

A late-season sharpening temperature helps ROTY only. Sharpening the distribution
as the season progresses was validated for Rookie of the Year and killed for MVP
and DPOY on walk-forward evidence. The mechanism is simple: sharpening helps only
when the model's top pick is reliably the eventual winner, which is true late in a
rookie race (around ninety per cent) but not for MVP (around two-thirds).

Concentration control stays, tail-distrust goes. The sizing layer had two
option-selling risk controls, a per-name concentration roll-off and a deep-tail
epistemic shrink. The concentration roll-off is load-bearing: removing it pushed
the book toward full Kelly and reproduced exactly the leverage pathology that
bankroll-Kelly was rejected for (drawdown from -11.7% to -30%). It stays. The
tail-distrust term was inert (its calibration knob defaulted to off) and was
removed as dead weight. The lesson from this episode is recorded in the conventions:
whether a component is inert is decided only by neutralising it in code and
re-running the sealed backtest, never by inspecting the trade log, which can lie.

## Known limitations, accepted deliberately

Defensive Player of the Year has a hard ceiling. The award increasingly rewards
perimeter defenders whose value the box score cannot see, so the model, which reads
the box score, is structurally wrong on perimeter winners a large fraction of the
time. The only identified path to closing this is a premium-outlet narrative
corpus, which is future work. DPOY still trades and still deploys capital; it is
just the weakest of the three books and sized accordingly.

The fair-value estimate is sticky on faded former favourites. When a former
favourite's case collapses, the model keeps them at a moderate vote share longer
than the market does. This is a genuine model weakness, understood and left
unfixed for v1.

## Scope boundaries for v1

The system trades only the three voter-decided season awards. It does not trade
championship, conference or division markets (outcome-determined, sharply priced,
smaller edge), statistical-leader markets (a simulation problem, minimal narrative,
scoped as a separate future extension), single-game lines (sharp-book dominated),
or Finals MVP (conditional on the Finals, tiny sample). It does not run a formal
market-making book. These are boundaries of focus, not permanent exclusions.
