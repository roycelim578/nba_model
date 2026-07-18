"""Persist per-season model and leaderboard Brier for the stat arm (PRA).

Fix over the first cut: do NOT form a per-season skill ratio. A single season's
leaderboard baseline Brier can collapse toward zero when the naive cumulative
leaderboard is nearly perfect that year, and 1 - model/baseline then explodes.
Instead we store, per season, the model Brier and the leaderboard Brier and the
row count, computed with ONE temperature fit across the whole dev window, and let
stat_book_weighting pool them into a stable trailing skill:

    skill(T) = 1 - sum_{s<T} n_s * B_model_s / sum_{s<T} n_s * B_leaderboard_s

which is the scorecard's pooled BSS-vs-leaderboard restricted to the trailing
window, and never divides by a near-zero denominator. Walk-forward: the weighter
only sums seasons strictly before its target.

Beta matches the shipping flag (on for REB and AST, off for PTS). PRA only; STL and
BLK need the STL/BLK-capable scorecard. British English, no em dashes.

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
        collect, STAT_FLOOR, _fit_temperature, _baseline_by_key)
except ImportError:  # pragma: no cover
    from db import connect  # type: ignore
    import mc as MC  # type: ignore
    import calib as CB  # type: ignore
    from scorecard import collect, STAT_FLOOR, _fit_temperature, _baseline_by_key  # type: ignore

log = logging.getLogger("stat_leader.bss_persist")

BETA_BY_STAT = {"reb": True, "pts": False, "ast": True}


def _one_season(job):
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


def _season_briers(rows_all):
    """One dev-window temperature, then per-season model and leaderboard Brier.
    Returns {season: {"n", "brier_model", "brier_lead"}}."""
    if len(rows_all) < 2:
        return {}
    T, _ = _fit_temperature(rows_all)
    base_key = _baseline_by_key(rows_all, T)
    by_season = defaultdict(list)
    for r in rows_all:
        by_season[int(r["season"])].append(r)
    out = {}
    for s, rws in by_season.items():
        y = np.array([r["y_lead"] for r in rws], float)
        pm = np.array([r["p_lead"] for r in rws], float)
        pb = np.array([base_key[(r["season"], r["snap"], r["pid"])] for r in rws], float)
        out[s] = {"n": int(len(rws)),
                  "brier_model": float(np.mean((pm - y) ** 2)),
                  "brier_lead": float(np.mean((pb - y) ** 2))}
    return out


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Persist per-season model/leaderboard Brier for PRA.")
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
            _, out = fut.result()
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
        by_season = _season_briers(rows)
        result[MC.STAT_AWARD[st]] = {str(s): by_season[s] for s in sorted(by_season)}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
    print(f"\nwrote {args.out}")
    for aw, d in result.items():
        num = sum(v["n"] * v["brier_model"] for v in d.values())
        den = sum(v["n"] * v["brier_lead"] for v in d.values())
        bss = 1.0 - num / den if den > 0 else float("nan")
        print(f"  {aw}: {len(d)} seasons, pooled BSS-vs-leaderboard {bss:+.3f} "
              f"(should match the scorecard)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
