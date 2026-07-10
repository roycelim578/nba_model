# NBA Awards Trader

A quantitative fair-value pricing and trading system for the three voter-decided NBA
season awards (MVP, DPOY, ROTY) on Polymarket. It prices each award as a probability
distribution over candidates, compares that fair value against market prices, and trades
the gap. This README summarises the whole project; the five documents in `docs/`
(STATUS, DECISIONS, MECHANICS, RESULTS, STRUCTURE) carry the full detail.

## The idea

Polymarket's award markets are retail-dominated, and retail prices move more on news
than the true change in voting probability warrants. Because these awards resolve on an
objective event months later (the vote is counted), a patient trader who can price the
eventual vote more accurately than the crowd can fade the overreaction and hold to
resolution. The defining framing is that the model predicts voter behaviour, not merit:
it estimates the probability each candidate wins the vote given the stats and the state
of the race, trained on decades of historical vote shares. Narrative matters to voters,
so it is part of the thing being predicted rather than a separate signal. A second,
reinforcing edge is longshot bias: retail overpays for convex payouts, so favourites are
underpriced and longshots overpriced. Full reasoning in `docs/DECISIONS.md`.

## How it works

The pipeline is a chain, traced in full in `docs/MECHANICS.md`:

- Raw data (game logs, advanced stats, team records, award-voting history, Polymarket
  prices, injuries) lands in a SQLite database, stored game-by-game so every feature can
  be computed strictly as of a snapshot date (no leakage).
- Features are built per candidate per snapshot, including within-race relative
  encodings, and fed to a LightGBM model with a custom grouped Plackett-Luce objective
  that prices a whole race jointly rather than scoring names in isolation. Uncertainty
  comes from a bootstrap ensemble.
- At trade time the pinned model produces two weightings: a sharp point estimate
  (softmax_of_mean) that is differenced against price to quote edge, and a flatter
  outcome-frequency weighting (mean_of_softmax) that a hold-to-resolution log-Kelly
  objective must size by. The distribution is then reweighted for voter fatigue and
  eligibility and reconciled against a first-place floor.
- A risk layer turns edge into a risk-adjusted signal; a portfolio-Kelly solver turns it
  into a target dollar position; a transaction-cost no-trade region decides whether the
  target is worth acting on; and a ledger executes and settles. Capital is split across
  the three books by a pinned skill-weighting table.

Narrative was investigated at length (a full NLP pipeline was built) and found to add
nothing to fair value, so it was removed entirely; the stats already carry it.

## Results

On the sealed 2025 out-of-sample test (never trained or tuned on, run once), on a 3,000
pooled bankroll: pooled +$460 (15.3% static return), Sharpe 0.71, maximum drawdown
-11.7% over 216 trading days, split MVP +$152, DPOY +$116, ROTY +$192, all three books
profitable. The shipped capital split (shrunk skill-weighting) was chosen over equal or
raw weighting for its better risk-adjusted return. This is one sealed season, so the
direction and the relative rankings are the trustworthy findings and the dollar figure
is indicative, not an expected annual return. Detail, the allocation comparison, the
per-book skill scores, and the honest caveats are in `docs/RESULTS.md`.

## Layout

Run modules from the repo root as `uv run python -m scripts.<stage>.<module>`.

- `scripts/` the code, staged: `common`, `data_pull`, `features`, `modelling`,
  `strategy` (pricing, forward estimates, sizing, allocation, trade regions, cost),
  `backtest` (engine, settle).
- `models/` deployed model finals, out-of-sample bundles, per-fold boosters and the
  forward-vol artefact. NOT in git (large, load-bearing); a backed-up artefact store to
  restore on a fresh clone or regenerate via the training pipeline.
- `data/` the SQLite database and caches, and `out/` run outputs and backtest goldens.
  Neither is in git; both are regenerable or restored from backup.
- `schema/` DDL, `tests/` the suite, `docs/` the five documents. A file-by-file
  inventory of every module is in `docs/STRUCTURE.md`.

## Reproducing the sealed backtest

From the repo root, with `models/` and `data/` restored:

```
caffeinate -i ./gate.sh
```

`gate.sh` runs the 2025 single-pass backtest and diffs the output against a frozen
golden, reporting a bit-identical pass or failure. Book budgets come from the pinned
`config.BOOK_WEIGHTS` table (2025: MVP 951, DPOY 741, ROTY 1308); no environment
override is needed. Every change to the trading logic is validated against this gate.

## Held-out discipline

The central rule. Seasons are named by starting year (2025 = 2025-26). 2024 is a burned
development season. 2025 has been spent on the one-shot sealed test for MVP, DPOY and
ROTY, so it is no longer clean for them. Any new award (Sixth Man, stat leaders) holds
out both 2024 and 2025. This is enforced in code: `config.SEAL_REGISTRY` plus
`assert_not_sealed`, which refuses to score a sealed season.

## Status

The three-award system is complete, sealed and documented, and the repository has been
migrated into this clean, self-contained tree with the sealed result reproduced
bit-identically throughout. What remains is not modelling but deployment: live trading is
blocked only by jurisdiction (Polymarket cannot be opened from Singapore or the UK). The
resolution in progress is an Oracle Cloud always-free VPS in Stockholm (Sweden is
unrestricted); the account is provisioned but the bot is not yet set up or tested there.
On the horizon are two extensions, Sixth Man (a near-drop-in) and statistical-leader
markets (a different, simulation-based build). The living tracker is `docs/STATUS.md`.

## Environment

Python >= 3.11, managed with `uv`. Install with `uv sync`; exact pinned versions are in
`requirements.txt`.

## Documents

- `docs/STATUS.md` living task tracker: what is done, in flight and next.
- `docs/DECISIONS.md` the reasoning trail: thesis, model choices, and rejected approaches.
- `docs/MECHANICS.md` the full pipeline, every stage and gate, from data to execution.
- `docs/RESULTS.md` the sealed performance and the honest caveats.
- `docs/STRUCTURE.md` a file-by-file inventory for auditing dependencies and redundancy.
