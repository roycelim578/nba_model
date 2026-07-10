# STRUCTURE

A file-by-file map of the repository, grouped by pipeline stage, so the codebase can be
audited for redundancy and its dependencies traced. For each file: what it does, and
whether it runs during the sealed backtest (hot path) or produces inputs and runs
offline (auxiliary). An audit section at the end lists suspected redundancies and stale
names to verify. This complements MECHANICS (which traces the runtime flow) by giving
the static inventory. British English.

Hot path means the module executes when `gate.sh` runs the 2025 backtest. Auxiliary
means it populates the database, trains a model, or computes a pinned constant offline,
and its outputs (not the module) are what the hot path consumes.

## scripts/common — shared infrastructure

- `config.py` (55L, hot). Single source of truth for pinned artefacts and sealed-run
  constants: `FINAL_MODEL`, `OOF_BUNDLE`, `OOF_SEASON_CAP`, `BOOK_WEIGHTS`,
  `SEAL_REGISTRY` and `assert_not_sealed`. Everything downstream reads paths from here.
- `db.py` (94L, hot/aux). SQLite connection and upsert helpers, used by both the pullers
  and the feature loader.
- `injury_categories.py` (129L, hot). Classifies free-text injury-report reasons into
  categories; feeds the eligibility reweight.
- `risk_metrics.py` (99L, hot). Pooled portfolio risk metrics (Sharpe, Sortino, max
  drawdown) for the cross-award report.
- `samples_cache.py` (61L, hot). Content-addressed cache wrapping `build_samples`, keyed
  on pinned artefact paths and mtimes plus a scoring-code fingerprint.

## scripts/data_pull — data acquisition (all auxiliary)

These populate `data/awards.db` and run periodically, never during a backtest.

- `nba/nba_api_pull.py` (754L). Pulls game logs, advanced box stats, team records from
  nba_api.
- `bref/bref_common.py` (141L). Shared Basketball-Reference fetch layer.
- `bref/bref_voting.py` (325L). Scrapes award-voting history with per-candidate vote
  shares: the model's training labels.
- `bref/bref_players.py` (199L). Player reference and bio scraper.
- `pm/pm_gamma.py` (385L). Polymarket Gamma market discovery and candidate population.
- `pm/pm_clob.py` (265L). Polymarket CLOB daily price puller (writes `pm_prices`).
- `pm/pm_clob_targeted.py` (239L). Targeted CLOB backfill for stat-leader and secondary
  markets; belongs to the future extensions, not the three shipped awards.
- `pm/pm_classify.py` (81L). Award classification for PM markets.

## scripts/features — feature engineering

- `feature_loader.py` (673L, hot). Materialises the `feature_stats_asof` join and builds
  the in-memory design matrix, including the relative encodings. Narrative merges have
  been removed.
- `nba_candidate_filter.py` (523L, hot). The as-of candidate-admission filter: who is in
  a given race at a given snapshot.
- `eligibility/eligibility.py` (240L, hot). As-of eligibility reweight (v2), the
  injury-conditioned availability mixture, used both as a feature and as a reweight.
- `positional_z.py` (191L, hot). Within-position standardised box and advanced stats,
  feeding the relative-feature encodings (audit #5, docstring added).

## scripts/modelling — training and scoring (all auxiliary)

These produce the artefacts the backtest loads; they do not run during a backtest.

- `train/pl_objective.py` (208L). The custom LightGBM grouped Plackett-Luce
  multinomial cross-entropy objective.
- `train/pl_trainer.py` (746L). The shared PL training core: objective wiring, K-booster
  ensemble fit and feature handling, imported by `persist_fold`, `retrain`, `score_fold`
  and `oof_stage1`. Load-bearing, not scaffold (audit #3 resolved).
- `train/persist_fold.py` (243L). Trains and persists a single walk-forward fold's
  K-booster ensemble; carries the held-out training guard.
- `train/retrain.py` (322L). The monotone-constraint A/B retrain for MVP and DPOY plus
  the jspeak-floor component for ROTY: this is what produced the deployed finals.
- `score/score_fold.py` (218L). Scores a target season with a persisted ensemble, writes
  `model_predictions`; a validation and fold-scoring tool.
- `score/oof_stage1.py` (257L). Produces the walk-forward OOF score bundles (the
  `OOF_BUNDLE` artefacts) for eta calibration and book-weighting. Has no importers; the
  two-stage narrative model it was named for is dead (audit #4, docstring corrected).

## scripts/strategy — the trading logic

### pricing (reweights and cost inputs)

- `pricing/fatigue_reweight.py` (217L, hot). Performance-conditional voter-fatigue
  reweight (MVP, DPOY).
- `pricing/eligibility` — see features/eligibility above; applied here as a reweight.
- `pricing/injury_miss_model.py` (380L, hot). Distribution over how many further games
  an injured player will miss; feeds eligibility.
- `pricing/renorm_set.py` (218L, hot). Contender-set renormalisation, with the separate
  softmax-denominator-versus-allocation-mask handling.
- `pricing/jspeak_reshape.py` (110L, hot). The first-place-share floor lift, composed
  into `renorm_set`'s masked-cloud reshape.
- `pricing/fp_point_loader.py` (94L, hot). Loads the first-place point estimates that
  the jspeak floor binds against.

### forward_estimates (edge and price variance)

- `forward_estimates/forward_edge.py` (165L, hot). The composed risk-adjusted edge
  object: `radj`, expected edge, CVaR, per-position sigma.
- `forward_estimates/forward_vol.py` (389L, hot). The packaged forward price-volatility
  callable used by the edge and the size scaler.
- `forward_estimates/pm_corpus.py` (521L, aux). Assembles and validates the unified
  forward-vol model offline; its output pickle is what `forward_vol` loads.

### sizing

- `sizing/sizer.py` (197L, hot). The portfolio-Kelly solver (`solve_award_v2`,
  `rank_floor_mask`), v2, successor to the old `backtest_sizer.solve_award`.
- `sizing/size_scaling.py` (80L, hot). The smooth `f_vol` and `f_conc` shrinks
  (`f_tail` removed).
- `sizing/sizer_fill.py` (72L, hot). Fill-fraction glue wiring the vol fill into the
  sizer.
- `sizing/soft_outcome.py` (114L, hot). The score-cloud reductions `edge_weights`
  (softmax_of_mean, used for edge) and `sizing_weights` (mean_of_softmax, the log-Kelly
  outcome weights). Moved here from `backtest/settle` (audit #6).

### allocation

- `allocation/book_weighting.py` (105L, aux). The offline shrunk-BSS calculator; its
  numbers are pinned into `config.BOOK_WEIGHTS`, which is what the hot path reads.

### trade_regions (execution)

- `trade_regions/notrade_region.py` (325L, hot). The transaction-cost no-trade region:
  band, hurdle, fill form, minimum fill, confirmation and hysteresis, reversal logic.
- `trade_regions/region_adapter.py` (172L, hot). Adapts the orchestrator's per-snapshot
  data to `notrade_region` and handles churn.

### cost

- `cost/cost_model.py` (124L, hot). The cost function the backtest trades against; loads
  frozen params.
- `cost/pm_fees.py` (64L, hot). The exact Polymarket taker-fee formula.

## scripts/backtest — the engine and settlement

### engine

- `engine/backtest_singlepass.py` (165L, hot). The entry point: runs the three books as
  independent daily processes and pools them; reads `config.BOOK_WEIGHTS`.
- `engine/backtest_orchestrator.py` (370L, hot). The shared-helper library:
  `A_build_samples`, the sample cloud, `_rebalance_to`, `_close_leg`, `_dump_csv` and the
  execution constants. No run path of its own (audit #1, docstring corrected).
- `engine/backtest_orchestrator_daily.py` (460L, hot). The daily run path:
  `prepare_award_daily` (where the seal guard sits), `_award_core`, the per-day reweight
  chain and rebalance.
- `engine/backtest_samples.py` (159L). Joint-sample producer (audit: daily-vs-non-daily).
- `engine/backtest_pricejoin.py` (150L). Price join (audit: daily-vs-non-daily).
- `engine/backtest_samples`/`pricejoin` `_daily.py` (142L pricejoin_daily). The daily
  variants used by the daily path.

### settle

- `settle/trade_ledger.py` (312L, hot). Positions, cash, trade log, per-player PnL
  attribution, and the model-versus-strategy verdict layer.

## Audit flags: status after the cleanup pass

All six flags from the first inventory were checked by import-grep and, where changed,
re-gated. None turned out to be dead code. Resolutions:

1. `backtest_orchestrator.py` stale docstring: FIXED. Rewritten to describe the
   shared-helper library it now is (no run path of its own).
2. Daily versus non-daily `samples`/`pricejoin`: NOT redundant. The non-daily
   `backtest_samples` and `backtest_pricejoin` are imported by the shared orchestrator
   through `A_build_samples`, so both are on the hot path. `backtest_samples` has no
   daily twin; only `pricejoin` does. One sub-check remains open: confirm
   `backtest_pricejoin_daily` has importers before ever retiring it.
3. Three training entry points: NOT redundant. `pl_trainer` is the shared training core
   imported by `persist_fold`, `retrain`, `score_fold` and `oof_stage1`, not superseded
   scaffold. Docstring clarified.
4. `oof_stage1` narrative naming: FIXED. Docstring corrected. It has no importers and
   produces the OOF bundles; the file was not renamed (nothing imports it, low value).
5. `positional_z` missing docstring: FIXED. Docstring added.
6. `soft_outcome` placement: FIXED. Moved from `backtest/settle` to `strategy/sizing`,
   its real home; the two package-path importers (`forward_edge`, `renorm_set`) were
   repointed and the gate stayed bit-identical.

The only open item is the sub-check under #2 (`backtest_pricejoin_daily` usage). Until
that is confirmed, nothing in the samples/pricejoin group is retired.
