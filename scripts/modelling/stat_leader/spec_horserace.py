"""Forward-projection horse-race: does decomposition beat a direct rate?

Two decisions:
  PTS  full shot tree (attempts x make% per zone, each leg shrunk separately)
       vs a direct points-per-game rate. If the tree does not project the
       remaining rate more accurately, it adds no mean signal and only error.
  REB  direct rate vs oreb+dreb split (summed) vs pool x share, where pool is
       team rebounds per game (from summed game-log rebounds) and share is the
       player's shrunk fraction of the team pool (the lineup-dependent view).

All specs use the SAME shrink shape as volume.rate_posterior: banked blended to
a snapshot pool by a pseudo-count (light for stable attempt/volume legs, heavy
for volatile make% legs), so the comparison is about STRUCTURE, not the constant.

Target per contender-snapshot: realised remaining per-game rate over the rest of
the season. Metrics: RMSE, MAE, bias, R2, residual-SD, n. Lower RMSE / higher R2
= better mean projection. Residual-SD is the irreducible forward spread each spec
must cover (true coverage calibration is a separate MC step).

Run:
  uv run python3 -m scripts.modelling.stat_leader.spec_horserace --seasons 2016-2023
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict

import numpy as np

try:
    from scripts.common.db import connect
    from scripts.features.stat_leader import nodes as N
except ImportError:  # pragma: no cover
    from db import connect  # type: ignore
    import nodes as N  # type: ignore

K_LIGHT = 3.0    # games, stable attempt/volume legs
K_HEAVY = 60.0   # attempts, volatile make% legs
K_RATE = 10.0    # games, a whole-rate shrink


def _shrink(x, n, pool, k):
    if x is None or pool is None:
        return pool if x is None else x
    return (n * x + k * pool) / (n + k) if (n + k) > 0 else pool


def _pts(d):
    return 2.0 * (d.get("fg2m") or 0.0) + 3.0 * (d.get("fg3m") or 0.0) + (d.get("ftm") or 0.0)


def _metrics(name, pred, real):
    pred, real = np.asarray(pred, float), np.asarray(real, float)
    e = pred - real
    sse = float(np.sum(e ** 2))
    sst = float(np.sum((real - real.mean()) ** 2))
    return (name, np.sqrt(np.mean(e ** 2)), np.mean(np.abs(e)), e.mean(),
            1 - sse / sst if sst > 0 else float("nan"), e.std(), len(e))


def _team_reb_asof(conn, seasons, snaps_by_season):
    qs = ",".join("?" * len(seasons))
    tg = defaultdict(list)
    for r in conn.execute(
            f'SELECT season, team_id, game_id, MIN(game_date) gd, SUM(rebounds) reb '
            f'FROM stg_nba_player_game_logs WHERE season IN ({qs}) '
            f'GROUP BY season, team_id, game_id', seasons):
        tg[(r["season"], r["team_id"])].append((r["gd"], r["reb"] or 0.0))
    out = {}
    for (s, tid), games in tg.items():
        games.sort(key=lambda x: x[0] or "")
        gi = cg = 0
        creb = 0.0
        for snap in sorted(snaps_by_season.get(s, [])):
            while gi < len(games) and games[gi][0] is not None and games[gi][0] <= snap:
                creb += games[gi][1]; cg += 1
                gi += 1
            if cg:
                out[(s, snap, tid)] = creb / cg
    tmap = {}
    for r in conn.execute(
            f'SELECT season, nba_api_id, team_id, COUNT(*) c FROM stg_nba_player_game_logs '
            f'WHERE season IN ({qs}) GROUP BY season, nba_api_id, team_id', seasons):
        k = (r["season"], r["nba_api_id"])
        if k not in tmap or r["c"] > tmap[k][1]:
            tmap[k] = (r["team_id"], r["c"])
    return out, {k: v[0] for k, v in tmap.items()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/awards.db")
    ap.add_argument("--seasons", default="2016-2023")
    a = ap.parse_args(argv)
    lo, hi = (a.seasons.split("-") + [a.seasons])[:2]
    seasons = list(range(int(lo), int(hi) + 1))

    conn = connect(a.db)
    counts, finals, _, _ = N._load(conn, seasons)
    dreb = {}
    for r in conn.execute(
            f'SELECT season, snapshot_date, nba_api_id, dreb FROM stg_nba_player_advanced_asof '
            f'WHERE season IN ({",".join("?"*len(seasons))})', seasons):
        if r["dreb"] is not None:
            dreb[(r["season"], r["snapshot_date"], r["nba_api_id"])] = float(r["dreb"])

    last = {}
    snaps_by_season = defaultdict(set)
    for (s, snap, pid) in counts:
        snaps_by_season[s].add(snap)
        k = (s, pid)
        if k not in last or snap > last[k]:
            last[k] = snap
    team_reb, tmap = _team_reb_asof(conn, seasons, snaps_by_season)

    # ------- PTS horse-race -------
    rows = []
    for (s, snap, pid), d in counts.items():
        gp = d.get("gp_played_asof") or 0.0
        fk = (s, pid)
        if gp < 5 or fk not in last:
            continue
        fd = finals.get(fk); fgp = (fd.get("gp_played_asof") or 0.0) if fd else 0.0
        rem = fgp - gp
        if rem < 10:
            continue
        real = (_pts(fd) - _pts(d)) / rem
        rows.append((s, snap, pid, gp, d, real))
    bysnap = defaultdict(list)
    for r in rows:
        bysnap[(r[0], r[1])].append(r)
    top = set()
    for key, lst in bysnap.items():
        lst.sort(key=lambda r: _pts(r[4]) / r[3], reverse=True)
        for r in lst[:25]:
            top.add((r[0], r[1], r[2]))
    rows = [r for r in rows if (r[0], r[1], r[2]) in top]

    ZONES = [("rim", "fg2a_rim", "fg2m_rim", 2.0), ("mid", "fg2a_mid", "fg2m_mid", 2.0),
             ("three", "fg3a", "fg3m", 3.0), ("ft", "fta", "ftm", 1.0)]
    poolppg, poolatt, poolpct = defaultdict(list), defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))
    for (s, snap, pid, gp, d, real) in rows:
        poolppg[(s, snap)].append(_pts(d) / gp)
        for z, ac, mc, _pv in ZONES:
            poolatt[(s, snap)][z].append((d.get(ac) or 0.0) / gp)
            P = poolpct[(s, snap)][z]; P[0] += (d.get(mc) or 0.0); P[1] += (d.get(ac) or 0.0)
    d_pred, t_pred, real_v = [], [], []
    for (s, snap, pid, gp, d, real) in rows:
        pppg = float(np.mean(poolppg[(s, snap)]))
        d_pred.append(_shrink(_pts(d) / gp, gp, pppg, K_RATE))
        pts = 0.0
        for z, ac, mc, pv in ZONES:
            att = _shrink((d.get(ac) or 0.0) / gp, gp, float(np.mean(poolatt[(s, snap)][z])), K_LIGHT)
            a_bk = d.get(ac) or 0.0
            P = poolpct[(s, snap)][z]
            pool_pct = P[0] / P[1] if P[1] > 0 else 0.0
            pct = _shrink((d.get(mc) or 0.0) / a_bk if a_bk > 0 else pool_pct, a_bk, pool_pct, K_HEAVY)
            pts += pv * att * pct
        t_pred.append(pts); real_v.append(real)
    print("=" * 68)
    print(f"PTS  (remaining points-per-game; n={len(real_v)})")
    print(f"  {'spec':<12}{'RMSE':>8}{'MAE':>8}{'bias':>8}{'R2':>8}{'residSD':>9}")
    for nm, rm, mae, bi, r2, rsd, n in (_metrics("direct", d_pred, real_v),
                                        _metrics("tree", t_pred, real_v)):
        print(f"  {nm:<12}{rm:>8.3f}{mae:>8.3f}{bi:>+8.3f}{r2:>8.3f}{rsd:>9.3f}")

    # ------- REB horse-race -------
    rows = []
    for (s, snap, pid), d in counts.items():
        gp = d.get("gp_played_asof") or 0.0
        fk = (s, pid)
        if gp < 5 or fk not in last:
            continue
        fd = finals.get(fk); fgp = (fd.get("gp_played_asof") or 0.0) if fd else 0.0
        rem = fgp - gp
        if rem < 10:
            continue
        real = ((fd.get("reb") or 0.0) - (d.get("reb") or 0.0)) / rem
        rows.append((s, snap, pid, gp, d, fd, rem, real))
    bysnap = defaultdict(list)
    for r in rows:
        bysnap[(r[0], r[1])].append(r)
    top = set()
    for key, lst in bysnap.items():
        lst.sort(key=lambda r: (r[4].get("reb") or 0.0) / r[3], reverse=True)
        for r in lst[:25]:
            top.add((r[0], r[1], r[2]))
    rows = [r for r in rows if (r[0], r[1], r[2]) in top]

    poolreb, poolo, poold, poolshare = (defaultdict(list) for _ in range(4))
    for (s, snap, pid, gp, d, fd, rem, real) in rows:
        reb = d.get("reb") or 0.0
        poolreb[(s, snap)].append(reb / gp)
        dr = dreb.get((s, snap, pid))
        if dr is not None:
            poold[(s, snap)].append(dr / gp)
            poolo[(s, snap)].append((reb - dr) / gp)
        tid = tmap.get((s, pid))
        tp = team_reb.get((s, snap, tid)) if tid is not None else None
        if tp and tp > 0:
            poolshare[(s, snap)].append((reb / gp) / tp)
    d_pred, sp_pred, pl_pred, real_v = [], [], [], []
    for (s, snap, pid, gp, d, fd, rem, real) in rows:
        reb = d.get("reb") or 0.0
        d_pred.append((_shrink(reb / gp, gp, float(np.mean(poolreb[(s, snap)])), K_RATE), real))
        dr = dreb.get((s, snap, pid))
        if dr is not None and poolo[(s, snap)] and poold[(s, snap)]:
            o = _shrink((reb - dr) / gp, gp, float(np.mean(poolo[(s, snap)])), K_RATE)
            dd = _shrink(dr / gp, gp, float(np.mean(poold[(s, snap)])), K_RATE)
            sp_pred.append((o + dd, real))
        tid = tmap.get((s, pid))
        tp = team_reb.get((s, snap, tid)) if tid is not None else None
        if tp and tp > 0 and poolshare[(s, snap)]:
            sh = _shrink((reb / gp) / tp, gp, float(np.mean(poolshare[(s, snap)])), K_RATE)
            pl_pred.append((tp * sh, real))
        real_v.append(real)
    print("\n" + "=" * 68)
    print(f"REB  (remaining rebounds-per-game; n={len(real_v)})")
    print(f"  {'spec':<12}{'RMSE':>8}{'MAE':>8}{'bias':>8}{'R2':>8}{'residSD':>9}{'n':>7}")
    for nm, pairs in (("direct", d_pred), ("oreb+dreb", sp_pred), ("pool x share", pl_pred)):
        if not pairs:
            print(f"  {nm:<12}  (no rows)"); continue
        p = [x[0] for x in pairs]; rr = [x[1] for x in pairs]
        _, rm, mae, bi, r2, rsd, n = _metrics(nm, p, rr)
        print(f"  {nm:<12}{rm:>8.3f}{mae:>8.3f}{bi:>+8.3f}{r2:>8.3f}{rsd:>9.3f}{n:>7}")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
