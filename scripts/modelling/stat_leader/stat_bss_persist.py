"""Persist per-season, per-book BSS-vs-leaderboard for the stat arm.

This is the stat-arm analogue of the voter OOF bundle that book_weighting reads.
It walks the Monte Carlo across the dev seasons exactly as the scorecard does
(same load_all, same collect, same per-book Beta as ships), then, instead of
pooling into one headline number, computes the leaderboard-anchored Brier skill
score per season and writes it to JSON. stat_book_weighting reads that JSON to
split the stat sub-bankroll the same way book_weighting splits the voter one:
trailing-mean of skill, floor negatives, normalise, shrink halfway to equal.

Scope note: PRA only. STL and BLK are not in this scorecard's stat set and have no
2024 prices, so their within-arm weight is deferred to the STL/BLK-capable scorecard
before the eight-book run. This artefact is sufficient for the 2024 PRA bss-vs-equal
read and the PRA leg of the cross-arm correlation.

Walk-forward and non-sealed by construction: it evaluates only the dev window
(2008-2023 by default) and refits priors on each season's own rolling lookback, so
it never touches a held-out season. Beta matches the shipping flag: on for REB and
AST, off for PTS. British English, no em dashes.

  caffeinate -i uv run python3 -m scripts.modelling.stat_leader.stat_bss_persist \
      --eval-min 2008 --eval-max 2023 --workers 6 --no-progress \
      --out models/stat_leader/bss_by_season_pra.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

try:
    from scripts.common.db import connect
    from scripts.modelling.stat_leader import mc as MC
    from scripts.modelling.stat_leader import calib as CB
    from scripts.modelling.stat_leader.scorecard import (
        collect, STAT_FLOOR, _fit_temperature, _baseline_by_key, _brier, _bss)
except ImportError:  # pragma: no cover
    from db import connect  # type: ignore
    import mc as MC  # type: ignore
    import calib as CB  # type: ignore
    from scorecard import (  # type: ignore
        collect, STAT_FLOOR, _fit_temperature, _baseline_by_key, _brier, _bss)

log = logging.getLogger("stat_leader.bss_persist")

BETA_BY_STAT = {"reb": True, "pts": False, "ast": True}


def _one_season(job):
    """Collect scorecard rows for every active PRA stat in one season. Runs in its
    own process, so it may set the MC module globals freely."""
    db, season, stats, fit_lookback, k, field_n = job
    active = [st for st in stats if season >= STAT_FLOOR[st]]
    if not active:
        return season, {}
    conn = connect(db)
    try:
        MC.AVAIL_HIER = True
        try:
            B = MC.load_all(conn, season, fit_lookback)
        except Exception as e:  # noqa: BLE001
            log.warning("season %d skipped (%s)", season, e)
            return season, {}
        MC.MPG_K = B["mpg_k"]
        MC.GAMES_K = B["games_k"]
        MC.OWN_PRIOR_K = MC.V.REF_MIN
        out = {st: collect(B, season, st, k, field_n) for st in active}
    finally:
        MC.MPG_K = None
        MC.GAMES_K = None
        MC.OWN_PRIOR_K = None
        MC.AVAIL_HIER = False
        conn.close()
    return season, out


def _season_leaderboard_bss(rows):
    """Leaderboard-anchored Brier skill score for one book's rows in one season.
    Reproduces the scorecard's BSS_vs_leaderboard construction on the season slice:
    fit the softmax temperature on the season's snapshots, take that as the no-skill
    baseline, and score P(lead) against it."""
    if len(rows) < 2:
        return float("nan")
    T, _ = _fit_temperature(rows)
    base_key = _baseline_by_key(rows, T)
    pL = np.array([r["p_lead"] for r in rows], float)
    yL = np.array([r["y_lead"] for r in rows], float)
    pB = np.array([base_key[(r["season"], r["snap"], r["pid"])] for r in rows], float)
    return _bss(pL, yL, ref_brier=_brier(pB, yL))


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Persist per-season leaderboard-BSS for PRA.")
    p.add_argument("--db", default="data/awards.db")
    p.add_argument("--eval-min", type=int, default=2008)
    p.add_argument("--eval-max", type=int, default=2023)
    p.add_argument("--fit-lookback", type=int, default=10)
    p.add_argument("--k", type=int, default=MC.DEFAULT_K)
    p.add_argument("--field-n", type=int, default=MC.FIELD_N)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--out", default="models/stat_leader/bss_by_season_pra.json")
    args = p.parse_args(argv)

    stats = ["reb", "pts", "ast"]
    try:
        from scripts.common.config import assert_not_sealed
    except ImportError:
        from config import assert_not_sealed  # type: ignore
    seasons = list(range(args.eval_min, args.eval_max + 1))
    for st in stats:
        for s in seasons:
            assert_not_sealed(MC.STAT_AWARD[st], s)

    jobs = [(args.db, s, stats, args.fit_lookback, args.k, args.field_n) for s in seasons]
    rows_by_stat = defaultdict(list)
    done = 0
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = [ex.submit(_one_season, j) for j in jobs]
        for fut in as_completed(futs):
            season, out = fut.result()
            for st, rws in out.items():
                rows_by_stat[st].extend(rws)
            done += 1
            if not args.no_progress:
                print(f"  seasons done {done}/{len(jobs)}", flush=True)

    result = {}
    for st in stats:
        rows = rows_by_stat.get(st, [])
        if not rows:
            print(f"stat={st}: no rows, skipping")
            continue
        if BETA_BY_STAT[st]:
            rows = CB.calibrate_plead_walkforward(rows)
        by_season = defaultdict(list)
        for r in rows:
            by_season[int(r["season"])].append(r)
        result[MC.STAT_AWARD[st]] = {
            str(s): _season_leaderboard_bss(by_season[s]) for s in sorted(by_season)}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
    print(f"\nwrote {args.out}")
    for aw, d in result.items():
        vals = [v for v in d.values() if v == v]
        mean = float(np.mean(vals)) if vals else float("nan")
        print(f"  {aw}: {len(d)} seasons, mean leaderboard-BSS {mean:+.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
