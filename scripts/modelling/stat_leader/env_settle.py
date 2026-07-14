"""Stat-leader arm: reb-env settlement, parallel in one process.

Settles two questions on evidence before the ship feature set is frozen:

  1. Does reb-env earn its keep? Runs the ship config (own-prior + avail-hier)
     with reb-env on versus off on REB, seed-matched, via the tested scorecard
     scoring, and also runs the ship config on PTS and AST for reference.

  2. Should the environment factor be one-for-all or none? Measures, per stat,
     the common-mode game-environment ICC the same way for reb, pts and ast: the
     share of per-minute-rate variance (residualised against each player's own
     season mean, so it is environment not talent) that is common within a game.
     If pts and ast show ICC comparable to reb, consistency argues for adding
     analogues; if not, reb-only stands on measured grounds.

Parallelism is built in: a ProcessPoolExecutor fans the (config, stat, season)
scoring tasks across cores, and BLAS threads are pinned to one per worker at
import so the workers do not oversubscribe. One command saturates the machine;
no shell loop needed. The prior cache makes both configs share a single fit per
season (fit-once, draw-many).

Run (single invocation, internally parallel):
  caffeinate -i uv run python3 -m scripts.modelling.stat_leader.env_settle \
      --eval-min 2008 --eval-max 2023 --workers 6 --fit-workers 3
"""

from __future__ import annotations

import os

# Pin BLAS to one thread per process BEFORE numpy/mc import; spawned workers
# inherit this environment, so N parallel workers stay single-threaded and do not
# fight over cores.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import logging  # noqa: E402
from collections import defaultdict  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402

import numpy as np  # noqa: E402

try:
    from scripts.common.db import connect
    from scripts.modelling.stat_leader import mc as MC
    from scripts.modelling.stat_leader import scorecard as SC
    from scripts.modelling.stat_leader import prior_cache as PC
except ImportError:  # pragma: no cover
    from db import connect  # type: ignore
    import mc as MC  # type: ignore
    import scorecard as SC  # type: ignore
    import prior_cache as PC  # type: ignore

log = logging.getLogger("stat_leader.env_settle")

STAT_FLOOR = {"pts": 1997, "reb": 1997, "ast": 2013}
STAT_COL = {"reb": "rebounds", "pts": "points", "ast": "assists"}

CONFIGS = [
    {"name": "ship (own+avail, reb_env ON)", "own_prior": True, "avail_hier": True,
     "no_reb_env": False, "corr": False, "corr2": False, "stats": ["reb", "pts", "ast"]},
    {"name": "reb_env OFF", "own_prior": True, "avail_hier": True,
     "no_reb_env": True, "corr": False, "corr2": False, "stats": ["reb"]},
]


def _apply_globals(cfg, B, avail_prior):
    MC.MPG_K = B["mpg_k"]
    MC.GAMES_K = B["games_k"]
    MC.REB_ENV_VAR = 0.0 if cfg["no_reb_env"] else B["reb_env_var"]
    MC.OWN_PRIOR_K = MC.V.REF_MIN if cfg["own_prior"] else None
    for attr, val in (("AVAIL_HIER", cfg["avail_hier"]), ("CORR", cfg["corr"]),
                      ("CORR2", cfg["corr2"]), ("HIER_FANO", False)):
        setattr(MC, attr, val)
    MC._AVAIL_PRIOR = avail_prior if cfg["avail_hier"] else None


def _fit_one(db, season, lookback, refit):
    conn = connect(db)
    try:
        PC.ensure(conn, season, lookback, refit)
    finally:
        conn.close()
    return season


def _score_one(db, cfg, stat, season, lookback, k, field_n):
    conn = connect(db)
    try:
        B, ap = PC.load(conn, season, lookback)
        _apply_globals(cfg, B, ap)
        rows = SC.collect(B, season, stat, k, field_n)
    finally:
        conn.close()
    return cfg["name"], stat, rows


def _env_icc(conn, stat, lo, hi, min_min=10.0):
    """Common-mode game-environment ICC for one stat: variance share of the
    per-minute rate (residualised against each player's own season mean) that is
    shared within a game. One-way random-effects ICC on game groups."""
    col = STAT_COL[stat]
    rows = list(conn.execute(
        f"SELECT nba_api_id pid, season, game_id gid, minutes mn, {col} v "
        f"FROM stg_nba_player_game_logs WHERE season BETWEEN ? AND ? AND minutes>=?",
        (lo, hi, min_min)))
    if not rows:
        return None
    rate = {}
    psum = defaultdict(lambda: [0.0, 0])
    for r in rows:
        x = (r["v"] or 0.0) / r["mn"]
        rate[(r["pid"], r["gid"])] = (r["pid"], r["season"], r["gid"], x)
        psum[(r["pid"], r["season"])][0] += x
        psum[(r["pid"], r["season"])][1] += 1
    pmean = {k: v[0] / v[1] for k, v in psum.items() if v[1] > 0}
    groups = defaultdict(list)
    for pid, season, gid, x in rate.values():
        groups[gid].append(x - pmean[(pid, season)])
    groups = {g: v for g, v in groups.items() if len(v) >= 2}
    if len(groups) < 2:
        return None
    allres = np.concatenate([np.array(v) for v in groups.values()])
    grand = allres.mean()
    ni = np.array([len(v) for v in groups.values()], float)
    gmean = np.array([np.mean(v) for v in groups.values()])
    ssb = float(np.sum(ni * (gmean - grand) ** 2))
    ssw = float(np.sum([np.sum((np.array(v) - np.mean(v)) ** 2) for v in groups.values()]))
    a, N = len(groups), allres.size
    msb = ssb / (a - 1)
    msw = ssw / max(1, (N - a))
    n0 = (N - np.sum(ni ** 2) / N) / (a - 1)
    icc = (msb - msw) / (msb + (n0 - 1) * msw) if (msb + (n0 - 1) * msw) > 0 else 0.0
    return max(0.0, icc), a, int(N)


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="reb-env settlement (parallel).")
    p.add_argument("--db", default="data/awards.db")
    p.add_argument("--eval-min", type=int, default=2008)
    p.add_argument("--eval-max", type=int, default=2023)
    p.add_argument("--fit-lookback", type=int, default=10)
    p.add_argument("--k", type=int, default=MC.DEFAULT_K)
    p.add_argument("--field-n", type=int, default=MC.FIELD_N)
    p.add_argument("--workers", type=int, default=6, help="score-phase parallel workers")
    p.add_argument("--fit-workers", type=int, default=3, help="fit-phase workers (memory-bounded)")
    p.add_argument("--refit", action="store_true", help="force prior re-fit")
    args = p.parse_args(argv)
    seasons = list(range(args.eval_min, args.eval_max + 1))

    try:
        from scripts.common.config import assert_not_sealed
    except ImportError:
        from config import assert_not_sealed  # type: ignore
    for cfg in CONFIGS:
        for st in cfg["stats"]:
            for s in seasons:
                if s >= STAT_FLOOR[st]:
                    assert_not_sealed(MC.STAT_AWARD[st], s)

    # Phase 1: build the prior cache, parallel across distinct seasons (race-free).
    log.info("phase 1: fitting priors for %d seasons (fit-workers=%d)", len(seasons), args.fit_workers)
    with ProcessPoolExecutor(max_workers=args.fit_workers) as ex:
        futs = [ex.submit(_fit_one, args.db, s, args.fit_lookback, args.refit) for s in seasons]
        for f in as_completed(futs):
            f.result()

    # Phase 2: score every (config, stat, season) in parallel; all cache hits.
    tasks = []
    for cfg in CONFIGS:
        for st in cfg["stats"]:
            for s in seasons:
                if s >= STAT_FLOOR[st]:
                    tasks.append((cfg, st, s))
    log.info("phase 2: scoring %d tasks (workers=%d)", len(tasks), args.workers)
    pooled = defaultdict(list)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_score_one, args.db, cfg, st, s, args.fit_lookback,
                          args.k, args.field_n) for cfg, st, s in tasks]
        for f in as_completed(futs):
            name, st, rows = f.result()
            pooled[(name, st)].extend(rows)

    for cfg in CONFIGS:
        for st in cfg["stats"]:
            rows = pooled.get((cfg["name"], st))
            if not rows:
                continue
            print("\n" + "#" * 94)
            print(f"# CONFIG: {cfg['name']}   stat={st}")
            print("#" * 94)
            SC.summary(st, rows)
            SC.phase_bin_report(st, rows)

    # Environment ICC, measured the same way for all three stats.
    conn = connect(args.db)
    print("\n" + "=" * 66)
    print("common-mode game-environment ICC (residualised per-min rate, by game)")
    print("  compare across stats: reb-only is justified only if reb >> pts, ast")
    print("-" * 66)
    print(f"  {'stat':>5} {'ICC':>8} {'games':>8} {'obs':>9}")
    for st in ("reb", "pts", "ast"):
        r = _env_icc(conn, st, args.eval_min, args.eval_max)
        if r:
            icc, a, n = r
            print(f"  {st:>5} {icc:>8.4f} {a:>8d} {n:>9d}")
    print("=" * 66)
    conn.close()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
