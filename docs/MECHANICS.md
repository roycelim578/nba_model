# MECHANICS

How the NBA Awards Trader runs, end to end, every stage and every gate, from the data
layer to order execution. This is the operational companion to DECISIONS (which
explains why the design is shaped this way). Seasons are named by starting year (2025
= 2025-26). British English.

The pipeline is a chain: raw data becomes as-of features, a grouped model turns
features into a distribution over candidates, that distribution is reweighted for
fatigue and eligibility and reconciled against a first-place floor, a risk layer turns
it into an edge, a sizer turns the edge into a target dollar position, a no-trade
region decides whether the target is worth the transaction cost, and a ledger executes
and marks it. Capital is split across the three books by a pinned weight table. Each
link is described below in order, with its inputs, outputs and the gate it applies.

## 1. Data layer

Source of truth is a SQLite database (`data/awards.db`, WAL mode). It is populated by
the pullers in `scripts/data_pull`: `nba` (game logs, advanced box stats, team records
via nba_api), `bref` (Basketball-Reference award-voting history with per-candidate vote
shares, the training labels), `pm` (Polymarket Gamma market discovery and CLOB daily
price history), and `identity` (resolving player names across sources into one id, with
a fuzzy-match alias table).

The one discipline that matters here is as-of correctness. Every feature the model sees
at a snapshot date must be computable from information available on that date, never
later. The data layer stores game-by-game rows with dates so that any feature can be
recomputed as of any point in a season, which is what makes the multi-snapshot training
rows (below) honest and the backtest non-leaky.

## 2. Feature engineering

`scripts/features` turns the raw rows into a feature matrix. For each candidate at each
snapshot date it computes box-score stats, advanced impact metrics, trajectory
(season-to-date, rolling and exponentially-weighted windows), team context (record, net
rating, seeding), career context (prior award shares, age, experience), and eligibility
(games played, projected games, the 65-game-rule flag, injury state). `feature_loader`
assembles these, and `eligibility` (under `features/eligibility`) computes the
availability factors used both as features and later as a reweight.

Two things are load-bearing here. First, relative encodings: for each per-candidate
stat, `relative_feature_cols` also produces its rank, percentile, delta-to-leader and
z-score within the race, because the model needs relativity, not absolute levels. Two
columns that live in the exclude bucket by default but are selected features,
`carry_prior_vote_share` and `on_65_game_pace_flag`, are hoisted explicitly into the
relative pass so their within-race encodings exist. Second, the narrative path is gone:
`feature_loader` no longer merges any `nli_` narrative columns and no shipped model
selects one (see DECISIONS for why narrative was killed).

## 3. Model: grouped Plackett-Luce

The model is LightGBM with a custom grouped Plackett-Luce objective (`scripts/modelling`,
`pl_objective` and `pl_trainer`). Within each race the model computes a softmax across
all candidates and compares it to the observed vote shares through a multinomial
cross-entropy loss, so each candidate's gradient depends on every other candidate in the
race through the softmax denominator. This is what makes it price a race jointly rather
than scoring names in isolation. The Hessian is floored (at 1e-3) to keep the boosting
stable under the softmax coupling.

Training rows are multi-snapshot: each candidate contributes one row per snapshot date
through a season, features as of that date, all predicting the same end-of-season vote
share. Uncertainty is a bootstrap ensemble of K=200 boosters, resampling whole races so
the grouped structure survives each resample. `persist_fold` trains and persists an
ensemble for a given train-cutoff; it carries a hard guard that refuses to train on a
held-out season.

## 4. Model selection and pinning

Which artefacts the backtest loads is pinned in `scripts/common/config.py`, not globbed,
so a later retrain cannot silently change a sealed result. `FINAL_MODEL` names the
deployed final ensemble per award (under `models/{award}/final`), `OOF_BUNDLE` names the
out-of-sample score bundle per award (under `models/{award}/oof`), and `OOF_SEASON_CAP`
(2023) fixes which seasons the forward-noise calibration is fit on, strictly before the
test seasons. Feature-selection ids (which columns a model uses) are distinct from the
model pickle hashes (which file); the config pins the files.

## 5. Scoring path

At backtest or live time, `A_build_samples` (in `backtest_orchestrator`, the shared
helper library) loads the pinned final ensemble and OOF bundle, scores every candidate
at every snapshot, and reduces the K-booster ensemble into a `JointSamples` object per
snapshot. It is wrapped by a content-addressed samples cache (`samples_cache`) keyed on
the artefact paths and mtimes and a fingerprint of the scoring code, so an edit that
does not touch scoring reuses the cached scores.

The reduction produces four distinct objects, and keeping them distinct is a deliberate
decision (see DECISIONS):

- The ensemble mean score per candidate, softmaxed within the race, is
  `vote_share_pred`, the calibrated central estimate (softmax_of_mean). This is the fair
  value and the edge object.
- A forward-noise term `eta` (Student-t, with a dispersion that scales with season
  fraction, fit on OOF residuals up to `OOF_SEASON_CAP`) is added to the per-booster
  scores to build a simulated future-score cloud; the argmax over that cloud gives
  `p_win` (Monte Carlo win probability), and the mean of its softmax gives
  `sizing_weights` (mean_of_softmax). The ensemble dispersion is thus carried into the
  risk layer, not folded into the point estimate.

## 6. Reweight pipeline (per day, in `_award_core`)

For each trading day the carried snapshot's samples are reset to a pristine base (so
reweights do not compound across days) and passed through an ordered chain, each stage
with its own gate:

1. Fatigue reweight (`fatigue_reweight`, gated by `FATIGUE_REWEIGHT` per award). A
   voter-fatigue adjustment on repeat winners, applied only when a composite recency
   signal and a count gate are met. Live for DPOY; the MVP enablement is behind that
   gate.
2. Eligibility reweight (`_elig_factors`, `_elig_reweight`). The availability factors
   (injury, games played, the 65-game rule) multiply the vote-share prediction, so an
   ineligible or heavily-missed candidate is downweighted. The same factors are applied
   as a log-shift to the simulation cloud so the sizing distribution sees them too.
3. Rank floor (`rank_floor_mask`). The tradeable set is the top candidates by base
   P(win): 20 for DPOY, 15 for MVP, 10 for ROTY. This is a set definition, not a
   fair-value correction; the full field's mass is kept (it sums to one) and out-of-set
   candidates are simply pinned to zero position.
4. Renormalisation and first-place floor (`renorm_set`, `fp_point_loader`, gated by
   `RENORM_MODE` and `JSFLOOR_AWARDS`). The contender set's mass is reconciled against
   market tradeability and, for the floored awards, lifted toward a first-place-share
   floor from a separate feed. A key fix here (see DECISIONS): the softmax denominator
   is the contender set ignoring tradeability, while the allocation mask is the
   contender set intersected with tradeability, so an untradeable favourite parks its
   mass as cash rather than dumping it onto the largest tradeable survivor.

## 7. Forward estimates: edge and price variance

`scripts/strategy/forward_estimates` turns the reweighted distribution into a tradeable
edge. `forward_edge` (`_composite`) combines the candidate's central P(win), the leg
price, the size-dependent entry cost, the forward-vol model and the season fraction into
a risk-adjusted edge `radj`, an expected edge, a CVaR (tail risk of the position), and a
per-position sigma. `forward_vol` supplies the price dispersion (`sigma_p`) used both
here and in the sizing scaler; it is calibrated from Polymarket price-history data
(`pm_corpus`).

## 8. Sizing

`scripts/strategy/sizing/sizer.py` (`solve_award_v2`) is a portfolio-Kelly solver: one
signed dollar variable per candidate (positive is a YES, negative a NO), maximising
expected log-growth over the joint winner states weighted by the central estimate, with
size-dependent leg costs, solved by SLSQP with restarts (restart spread flags
non-convexity). Three mechanisms sit inside it: winner-state weights are the calibrated
`vote_share_pred` (not the flatter mean_of_softmax); a turnover toll in the objective
plus a warm start from the current position stabilise the target week to week; and a
fade-your-own-top-k guard forbids a NO leg on the model's own favourites. Kelly fraction
is pinned at 1.0; conservatism is delegated downstream.

The raw signed Kelly allocation then passes through `size_scaling.scale_allocation`,
which applies two smooth multiplicative shrinks: `f_vol` (rolls size off when forward
price dispersion is high; defaults to 1 when the caller has already applied the vol fill
fraction) and `f_conc` (a per-name concentration roll-off, load-bearing, it holds the
book near one-third Kelly and its removal reproduces the rejected full-Kelly drawdown
pathology). The former tail-distrust scaler `f_tail` has been removed as inert.

## 9. Execution: the no-trade region

`scripts/strategy/trade_regions` decides whether a target is worth acting on, given that
every trade pays transaction cost. `notrade_region` (via `region_adapter`, active when
region mode is on) holds the current position inside a cost-sensitive band unless the
target moves enough to earn the round trip: it requires an edge to clear an open hurdle,
fills only a fraction of the gap (the fill form and fill constant govern how much),
enforces a minimum fill, and confirms a signal persists for a number of snapshots before
acting (the confirm and hysteresis settings). Two fixes are baked in (see DECISIONS): a
reversal is detected from the intended target's sign (budget-invariant), not the
band-projected position, and on a confirmed reversal the book trades fully to target
rather than stopping at the band edge; and an asymmetric trim widens the band on
converging winners. The cost itself comes from `scripts/strategy/cost` (`cost_model`,
`pm_fees`), a sqrt-impact curve exposing effective price and cost fraction at a given
size.

## 10. Ledger and settlement

`scripts/backtest/settle/trade_ledger` holds each book's positions, cash and trade log.
`_rebalance_to` moves a position toward its target (calling `_close_leg` when a leg is
being exited), `self_mark` marks the book to the day's mids, and `settle` resolves every
position at the known winner at season end. Positions are valued at cost for the sizing
frame (not marked to market), so gains do not pyramid size through a noisy mid.

## 11. Book weighting: splitting the bankroll

`scripts/strategy/allocation/book_weighting.py` computes how the pooled bankroll splits
across the three books, by shrunk contender-conditioned Brier skill score: per book, the
out-of-sample skill (1 minus Brier over base-rate Brier, restricted to vote-getters) is
averaged over the trailing ten seasons, floored at zero, normalised, shrunk halfway
toward an equal split, and multiplied by the bankroll. It is walk-forward (a season's
weights use only earlier seasons). Because this is a once-per-season calculation, the
module is the offline calculator and the actual numbers are pinned per season in
`config.BOOK_WEIGHTS` (2025: MVP 951, DPOY 741, ROTY 1308 on a 3,000 bankroll); the
table is regenerated and repinned when the book set changes. Each book Kelly-sizes
within its own budget and unused budget stays cash, with no spillover between books.

## 12. Backtest engine

`scripts/backtest/engine/backtest_singlepass.py` runs the whole chain. In the static and
compound cases the three books are independent during stepping, so it fans them across
three worker processes, each running its own book's daily loop, then pools the three
ledgers. `report_pooled` (`scripts/common/risk_metrics`) computes the pooled
mark-to-market curve and its Sharpe, Sortino and maximum drawdown. The default budget
source is the pinned `config.BOOK_WEIGHTS` table (an `ALLOC_BUDGETS_JSON` environment
variable can still override it).

## 13. The gate

The sealed 2025 backtest is the validation harness. `gate.sh` runs the single-pass
engine for 2025 and diffs the output CSVs against a frozen golden, reporting a
bit-identical pass or a failure. Every change to the trading logic is checked against it;
it must reproduce the pinned result (pooled +$460) exactly unless the change is a
deliberate, understood one. The stronger per-change check compares `trade_log`,
`position_log` and `equity_curve` byte-for-byte and diffs the `model_eval` columns.

## 14. Guards and discipline

Two guards protect the held-out discipline in code. `persist_fold` refuses to train on a
held-out season. `config.assert_not_sealed`, called at the top of the backtest entry
(`prepare_award_daily`), refuses to score or backtest a season listed as sealed for that
award in `config.SEAL_REGISTRY` (empty for the three shipped awards, since 2025 is
already spent; defaulting to both 2024 and 2025 for any unlisted new award). Together
they make it hard to accidentally spend a one-shot test season.
