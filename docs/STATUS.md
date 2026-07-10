# STATUS

Living tracker for the NBA Awards Trader. This is the one document that changes
often: what is built, what is in flight, what is next, and the conventions that keep
the work safe. The companion documents (DECISIONS, MECHANICS, RESULTS) change rarely
and describe the system itself.

Last updated: 2026-07-10.

## What this project is

A quantitative fair-value pricing and trading system for the three voter-decided NBA
season awards: Most Valuable Player, Defensive Player of the Year, and Rookie of the
Year. It prices each award as a probability distribution over candidates, compares
that fair value against Polymarket prices, and trades the gap. The edge is retail
overreaction in a market with an eventually objective resolution: prices move more on
narrative than the change in voting probability warrants, and the model prices the
voting outcome rather than the narrative. It predicts how voters will vote, not who
deserves to win, because narrative is part of what voters weigh. The reasoning is in
DECISIONS, the pipeline in MECHANICS, the performance in RESULTS.

## Current state

The three-award system is complete and validated, and the codebase has been migrated
into a clean, self-contained repository with the sealed backtest reproduced
bit-identically at every step.

- The sealed 2025 out-of-sample test has been run and post-mortemed: pooled +$460 on a
  3,000 bankroll, Sharpe 0.71, maximum drawdown -11.7%, split MVP +$152, DPOY +$116,
  ROTY +$192 over 216 trading days. Full reading in RESULTS.
- The repository is self-contained (no symlinks), artefacts live under `models/`, the
  per-season book weights are pinned in `config.BOOK_WEIGHTS`, and a hard seal guard
  refuses to score a held-out season.
- What remains is documentation (this set) and the initial git commit.

## Held-out season discipline

The most important rule in the project, never relaxed. Seasons are named by starting
year, so 2024 means 2024-25.

- 2024 is burned: used extensively for development and backtest debugging, so no longer
  a clean test.
- 2025 has been spent. It was the single one-shot sealed test for MVP, DPOY and ROTY,
  and that test has now been run. It is no longer a clean out-of-sample season for these
  three awards; treat any further 2025 work on them as in-sample.
- Any new award (Sixth Man, stat leaders) holds out BOTH 2024 and 2025, since neither
  has been spent on that award. This is enforced in code: `config.SEAL_REGISTRY` lists
  held-out seasons per award (empty for MVP/DPOY/ROTY, defaulting to 2024 and 2025 for
  any unlisted award), and `assert_not_sealed` raises if a backtest touches one.

## Done

- Data foundation: game logs, advanced stats, team records, award voting history,
  Polymarket price history, injury and eligibility data.
- Stats model: LightGBM with a grouped Plackett-Luce objective, bootstrap ensemble,
  walk-forward cross-validation by season.
- Narrative investigation: concluded narrative adds nothing to fair value and removed
  it from the scoring path entirely (see DECISIONS).
- Pricing, sizing, allocation and execution layers, including the transaction-cost
  no-trade region and the shrunk book-weighting split.
- Sealed 2025 validation, run and post-mortemed.
- Repository migration and cleanup: parallel backtest engine, dead-code removal,
  narrative purge, sizer merge, artefact relocation, symlink resolution. Bit-identical
  throughout.
- Book-weighting productionised: an offline calculator plus a pinned per-season table
  the backtest reads.
- Per-award seal registry and guard.
- Repository hygiene files (dependency spec, gitignore, README).
- Documentation set: STATUS, DECISIONS, MECHANICS, RESULTS, and a file-by-file STRUCTURE
  inventory.
- Post-baseline audit pass: the six structure and redundancy flags checked (none dead
  code), stale docstrings corrected, and `soft_outcome` relocated from `backtest/settle`
  to `strategy/sizing` with the gate bit-identical.

## In flight

- Nothing substantive. One minor audit sub-check remains (whether
  `backtest_pricejoin_daily` has any importers), noted in STRUCTURE for the next time the
  engine is touched.

## Next

- Commit the post-baseline cleanup (the docstring corrections, the `soft_outcome` move,
  and this documentation set) on top of the initial baseline commit.

## On the horizon

Not started, scoped only.

- Sixth Man of the Year: a near-drop-in extension of the existing model, needing a
  bench-role eligibility filter and new ballot labels.
- Statistical-leader markets: a different problem (a season-long stat-total simulation)
  needing a distributional stage-one model and a Monte Carlo layer.
- Live deployment: see below.

## Live deployment status

The blocking constraint is jurisdiction, not code: new Polymarket positions cannot be
opened from Singapore or the UK. The resolution in progress is an Oracle Cloud
always-free VPS provisioned in Stockholm under Royce's name; Stockholm egress is the
geolocation Polymarket sees, and Sweden is unrestricted. Nothing beyond the account
itself is set up yet. The remaining work is to set up the bot within that VPS and test
that the trading path works end to end from there. Read-only data pulling is
unrestricted everywhere, so only order placement is gated on this. Live trading also
waits on the documentation and a period of paper trading before any capital is staged.

## Conventions

- Sealed discipline as above. Never touch 2025 on MVP/DPOY/ROTY as if it were clean; the
  guard enforces the new-award holdouts.
- Every change to the trading logic is validated against the sealed backtest, which must
  reproduce the pinned result exactly unless the change is a deliberate, understood one.
- Whether a component is inert is decided only by neutralising it in code and re-running
  the backtest, never by reading the trade log. Code deletions get a backtest run even
  when a search looks conclusive.
- Terminology: the portfolio is the whole capital base; a book (award) is one of
  MVP/DPOY/ROTY; a position is one candidate leg within a book.
- British English throughout; no em dashes. Long unattended runs use `caffeinate -i`.

## Change log

- 2026-07-10 (later): initial baseline committed to git; post-baseline audit pass applied
  (docstring corrections on `backtest_orchestrator`, `positional_z`, `oof_stage1` and
  `pl_trainer`; `soft_outcome` moved to `strategy/sizing`, gate bit-identical); full
  documentation set finalised.
- 2026-07-10: repository migration and cleanup completed; book-weighting pinned; seal
  registry and guard added; hygiene files added. Sealed result held bit-identical
  throughout.
- Earlier: sealed 2025 validation completed and post-mortemed.
