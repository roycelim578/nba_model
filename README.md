# NBA Awards Trader

Fair-value pricing and trading for the NBA voter-decided season awards (MVP, DPOY,
ROTY) on Polymarket. The model predicts voter behaviour, prices each award as a
distribution over candidates, and trades the gap against retail-driven market prices.

See `docs/` for the full picture: `DECISIONS.md` (why the system is built this way),
`MECHANICS.md` (how the pipeline runs), `RESULTS.md` (measured performance), and
`STATUS.md` (the living task tracker).

## Layout

- `scripts/` the code, staged: `common`, `data_pull`, `features`, `modelling`,
  `strategy`, `backtest`. Run modules from the repo root as
  `uv run python -m scripts.<stage>.<module>`.
- `models/` deployed finals, OOF bundles, per-fold boosters, fp and forward-vol
  artefacts. NOT in git (large, load-bearing); a backed-up artefact store. Restore it
  from backup on a fresh clone, or regenerate via the training pipeline.
- `data/` SQLite database (`awards.db`) and caches. NOT in git; restore from backup.
- `out/` run outputs, caches, backtest goldens. NOT in git (regenerable).
- `schema/` database DDL. `tests/` the test suite. `docs/` the documentation.

## Reproducing the sealed backtest

From the repo root, with `models/` and `data/` restored:

```
caffeinate -i uv run python -m scripts.backtest.engine.backtest_singlepass \\
  --season 2025 --out out/sp_2025_shrunk
```

Region and trim settings are supplied by environment variables (see `gate.sh`, which
also diffs the run against the frozen golden). Book budgets come from the pinned
`config.BOOK_WEIGHTS` table; no environment override is needed. The sealed 2025 result
is pooled +$460 (Sharpe 0.71, max drawdown -11.7%).

## Held-out discipline

2024 is a burned development season. 2025 has been spent on the one-shot sealed test
for MVP/DPOY/ROTY. Any new award holds out both 2024 and 2025; `config.SEAL_REGISTRY`
and `assert_not_sealed` enforce this and will refuse to score a sealed season.

## Environment

Python >= 3.11, managed with `uv`. Install with `uv sync`; exact pinned versions are in
`requirements.txt`.
