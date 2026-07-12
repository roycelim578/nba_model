"""Stat-leader arm: availability + minutes joint node (top of the trunk).

Projects, per player per snapshot, the JOINT distribution of remaining-season
games played (as a fraction of remaining team games) and minutes per game, by
nonparametric conditional sampling from historical peer outcomes.

Joint sampling preserves the games/minutes covariance (a player easing back from
absence both misses games and plays fewer minutes): we draw (remaining_frac,
remaining_mpg) PAIRS from the same historical player-snapshots in the player's
current state bin, so the empirical covariance comes along automatically.

Nonparametric because absences are bursty/clustered; an independent-Bernoulli
per-game model understates the fat left tail of games-played that drives the
eff_value denominator floor max(G, q).

STATE BIN: banked-availability tercile x form direction (recent minutes vs
season) x currently-absent flag, with min-count backoff to coarser bins.
Fit on training seasons only.

Reads existing tables only (stg_nba_availability_asof, stg_nba_box_asof,
stg_nba_player_game_logs, snapshot_grid). Writes nothing the v1 gate reads.
Self-contained; tested via its walk-forward calibration report (PIT + coverage).

Run:
  uv run python -m scripts.features.stat_leader.availability
  uv run python -m scripts.features.stat_leader.availability --fit-max 2021 --eval-min 2022 --eval-max 2023
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict

import numpy as np

try:
    from scripts.common.db import connect
except ImportError:  # pragma: no cover
    from db import connect  # type: ignore

log = logging.getLogger("stat_leader.availability")

MIN_MPG_ROTATION = 15.0   # season-mpg floor: only project players who were rotation pieces
MIN_POOL = 40             # min historical pairs in a bin before backing off
K_SAMPLES = 400           # MC draws per snapshot for the calibration check


def _load(conn, seasons):
    """Return per (season, snapshot, nba_api_id) banked state + realised remaining
    labels, restricted to rotation players (final season mpg >= floor)."""
    qs = ",".join("?" * len(seasons))
    # banked state: availability + form (mpg) + grid ordinal
    rows = conn.execute(f"""
        SELECT a.season, a.snapshot_date, a.nba_api_id,
               a.games_played_asof, a.team_games_asof,
               a.current_absence_streak, a.missed_last_10_team_games,
               b.mpg_std, b.mpg_l10, g.week_index, g.snapshot_kind
        FROM stg_nba_availability_asof a
        JOIN snapshot_grid g ON g.season=a.season AND g.snapshot_date=a.snapshot_date
        LEFT JOIN stg_nba_box_asof b
          ON b.nba_api_id=a.nba_api_id AND b.season=a.season AND b.snapshot_date=a.snapshot_date
        WHERE a.season IN ({qs}) AND g.snapshot_kind IN ('weekly','ratings')
          AND a.team_games_asof > 0
    """, seasons).fetchall()

    # final team games / games played per (season, player) from availability max
    fin = {}
    for r in conn.execute(f"""SELECT season, nba_api_id,
              MAX(team_games_asof) ftg, MAX(games_played_asof) fgp
           FROM stg_nba_availability_asof WHERE season IN ({qs}) GROUP BY season, nba_api_id""",
           seasons):
        fin[(r["season"], r["nba_api_id"])] = (r["ftg"], r["fgp"])

    # game logs -> per (season, player) ordered (game_date, minutes) for remaining-mpg labels
    glog = defaultdict(list)
    for r in conn.execute(f"""SELECT season, nba_api_id, game_date, minutes
           FROM stg_nba_player_game_logs WHERE season IN ({qs}) AND minutes IS NOT NULL
           ORDER BY nba_api_id, game_date""", seasons):
        glog[(r["season"], r["nba_api_id"])].append((r["game_date"], r["minutes"]))

    # season mpg (rotation filter) = final mpg_std per player
    season_mpg = {}
    for r in conn.execute(f"""SELECT season, nba_api_id, mpg_std FROM stg_nba_box_asof b1
           WHERE season IN ({qs}) AND snapshot_date=(
             SELECT MAX(snapshot_date) FROM stg_nba_box_asof b2
             WHERE b2.nba_api_id=b1.nba_api_id AND b2.season=b1.season)""", seasons):
        season_mpg[(r["season"], r["nba_api_id"])] = r["mpg_std"] or 0.0

    recs = []
    for r in rows:
        key = (r["season"], r["nba_api_id"])
        if season_mpg.get(key, 0.0) < MIN_MPG_ROTATION:
            continue
        ftg, fgp = fin.get(key, (None, None))
        if ftg is None:
            continue
        rem_team = ftg - r["team_games_asof"]
        if rem_team <= 0:
            continue
        # realised remaining games/minutes from logs after this snapshot
        games = glog.get(key, [])
        rem = [m for (d, m) in games if d > r["snapshot_date"] and m and m > 0]
        rem_played = len(rem)
        rem_mpg = float(np.mean(rem)) if rem else None
        rem_frac = min(1.0, rem_played / rem_team)
        avail_rate = r["games_played_asof"] / r["team_games_asof"]
        form = (r["mpg_l10"] or r["mpg_std"] or 0.0) - (r["mpg_std"] or 0.0)
        recs.append({
            "season": r["season"], "snap": r["snapshot_date"], "pid": r["nba_api_id"],
            "avail_rate": avail_rate, "form": form,
            "absent": 1 if (r["current_absence_streak"] or 0) >= 2 else 0,
            "week_index": r["week_index"] if r["week_index"] is not None else -1,
            "rem_team": rem_team, "rem_frac": rem_frac, "rem_mpg": rem_mpg,
        })
    return recs


def _terciles(vals):
    a = np.array([v for v in vals if v is not None])
    return (np.quantile(a, 1/3), np.quantile(a, 2/3)) if len(a) else (0.6, 0.9)


def _bin(rec, tcut):
    at = 0 if rec["avail_rate"] < tcut[0] else (1 if rec["avail_rate"] < tcut[1] else 2)
    fd = 0 if rec["form"] < -0.5 else (2 if rec["form"] > 0.5 else 1)
    return (at, fd, rec["absent"])


def _backoff_keys(b):
    at, fd, absent = b
    return [(at, fd, absent), (at, None, absent), (None, None, absent), (None, None, None)]


def fit(recs, tcut):
    """Build bin -> pool of (rem_frac, rem_mpg) pairs, at every backoff level."""
    pools = defaultdict(list)
    for r in recs:
        b = _bin(r, tcut)
        for k in _backoff_keys(b):
            pools[k].append((r["rem_frac"], r["rem_mpg"]))
    return pools


def sample(pools, rec, tcut, k=K_SAMPLES):
    b = _bin(rec, tcut)
    for key in _backoff_keys(b):
        pool = pools.get(key)
        if pool and len(pool) >= MIN_POOL:
            idx = np.random.randint(0, len(pool), size=k)
            return [pool[i] for i in idx]
    pool = pools.get((None, None, None), [])
    if not pool:
        return []
    idx = np.random.randint(0, len(pool), size=k)
    return [pool[i] for i in idx]


def calib(pools, eval_recs, tcut):
    """PIT + central-80 coverage for games and minutes, split by season phase."""
    def phase(wi):
        return "early" if wi < 8 else ("mid" if wi < 18 else "late")
    acc = defaultdict(lambda: {"pit_g": [], "cov_g": [], "pit_m": [], "cov_m": []})
    for r in eval_recs:
        pairs = sample(pools, r, tcut)
        if not pairs:
            continue
        fr = np.array([p[0] for p in pairs])
        pred_played = fr * r["rem_team"]
        realised = r["rem_frac"] * r["rem_team"]
        ph = phase(r["week_index"])
        for bucket in (ph, "ALL"):
            acc[bucket]["pit_g"].append(float(np.mean(pred_played <= realised)))
            lo, hi = np.quantile(pred_played, [0.1, 0.9])
            acc[bucket]["cov_g"].append(1 if lo <= realised <= hi else 0)
        if r["rem_mpg"] is not None:
            mp = np.array([p[1] for p in pairs if p[1] is not None])
            if len(mp) >= 20:
                for bucket in (ph, "ALL"):
                    acc[bucket]["pit_m"].append(float(np.mean(mp <= r["rem_mpg"])))
                    lo, hi = np.quantile(mp, [0.1, 0.9])
                    acc[bucket]["cov_m"].append(1 if lo <= r["rem_mpg"] <= hi else 0)
    return acc


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="data/awards.db")
    p.add_argument("--fit-min", type=int, default=2013)
    p.add_argument("--fit-max", type=int, default=2021)
    p.add_argument("--eval-min", type=int, default=2022)
    p.add_argument("--eval-max", type=int, default=2023)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args(argv)
    np.random.seed(args.seed)
    conn = connect(args.db)

    fit_seasons = list(range(args.fit_min, args.fit_max + 1))
    eval_seasons = list(range(args.eval_min, args.eval_max + 1))
    fit_recs = _load(conn, fit_seasons)
    eval_recs = _load(conn, eval_seasons)
    conn.close()
    log.info("fit rows=%d (%s)  eval rows=%d (%s)",
             len(fit_recs), fit_seasons, len(eval_recs), eval_seasons)
    if not fit_recs or not eval_recs:
        log.error("no rows; check season ranges / table fill"); return 1

    tcut = _terciles([r["avail_rate"] for r in fit_recs])
    pools = fit(fit_recs, tcut)
    acc = calib(pools, eval_recs, tcut)

    print("\n" + "=" * 70)
    print(f"availability+minutes calibration  fit={fit_seasons} eval={eval_seasons}")
    print(f"avail_rate terciles: {tcut[0]:.3f}, {tcut[1]:.3f}")
    print("well-calibrated: PIT mean ~0.50, coverage@80 ~0.80")
    print("=" * 70)
    print(f"{'phase':>6} {'n':>5} {'PIT_G':>7} {'cov80_G':>8} {'PIT_M':>7} {'cov80_M':>8}")
    for ph in ("early", "mid", "late", "ALL"):
        a = acc[ph]
        if not a["pit_g"]:
            continue
        pg = np.mean(a["pit_g"]); cg = np.mean(a["cov_g"])
        pm = np.mean(a["pit_m"]) if a["pit_m"] else float("nan")
        cm = np.mean(a["cov_m"]) if a["cov_m"] else float("nan")
        print(f"{ph:>6} {len(a['pit_g']):>5} {pg:>7.3f} {cg:>8.3f} {pm:>7.3f} {cm:>8.3f}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
