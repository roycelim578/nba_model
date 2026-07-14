"""Stat-leader arm: within-season drift diagnostic (why is the elite centred low).

The shrinkage read showed the mean projection is ~98% own banked rate, yet the
coverage log shows the eventual top contenders out-produce their banked rate over
the remainder. That runs against regression to the mean, so this isolates what
produces it, model-agnostically (banked as-of versus realised remainder, no MC
draw), keeping the rate channel and the minutes channel separate because they are
physically different: a per-minute rate cannot rise by playing more, so a low
centre from the minutes channel means contenders play more remaining games or
minutes than a banked-pace extrapolation implies, while a low centre from the rate
channel needs a real cause (usage concentration, a rising league environment) or
is a selection artefact.

Four panels:

  1 by as-of rank (top5 / 6-15 / 16-30). Median of three ratios per phase:
      rate  = realised remaining per-min rate / banked per-min rate
      min   = realised remaining minutes / banked-pace-extrapolated minutes
      games = realised remaining games   / banked-pace-extrapolated games
    Reads whether the low centre is a rate or a minutes phenomenon, for the elite
    the model actually fields.

  2 by an as-of QUALITY axis (--by ts|vol|pra, default ts crossed nothing here;
    ts separates the efficient star from the empty-volume high-minute player, the
    bad-star-on-a-bad-team confounder). Median rate ratio and fraction above one
    per tier per phase. mpg is offered only as a contrast, never the skill axis.

  3 survivorship. The same rate ratio for the AS-OF top5 versus the EVENTUAL
    final top5 over identical snapshots. The gap between them is the selection
    magnitude directly: if eventual-top5 drifts far more than as-of-top5, the
    'drift' is conditioning on the winner, not a real within-season increase.

  4 league baseline. For ALL players with enough games in both halves, the median
    within-player second-half / first-half per-min rate. If contenders (panel 1)
    drift no more than this leaguewide figure, the driver is the environment
    (common-mode, which barely moves P(lead) but centres the absolute projection
    low); if they drift more, it is contender-specific.

Read-only, no retrain, no seed dependence (pure data ratios). Reuses MC.load_all
for a consistent contender field, banked counts and finals.

Run:
  caffeinate -i uv run python3 -m scripts.modelling.stat_leader.drift \
      --stat all --eval-min 2008 --eval-max 2023 --by ts
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict

import numpy as np

try:
    from scripts.common.db import connect
    from scripts.modelling.stat_leader import mc as MC
    from scripts.features.stat_leader import volume as V
except ImportError:  # pragma: no cover
    from db import connect  # type: ignore
    import mc as MC  # type: ignore
    import volume as V  # type: ignore

log = logging.getLogger("stat_leader.drift")

STAT_FLOOR = {"pts": 1997, "reb": 1997, "ast": 2013}
PHASES = ("early", "mid", "late")
NODE = MC.VOL_NODE
FT_POSS = MC.FT_POSS_COEF


def _phase(i, n):
    if n <= 1:
        return "mid"
    f = i / (n - 1)
    return "early" if f < 1 / 3 else ("mid" if f < 2 / 3 else "late")


def _ts(d):
    """As-of true-shooting: pts / (2 * (FGA + 0.44 * FTA)). Efficiency axis,
    orthogonal-ish to volume, so it separates real stars from empty volume."""
    fga = (d.get("fg3a") or 0.0) + (d.get("fg2a") or 0.0)
    fta = d.get("fta") or 0.0
    denom = 2.0 * (fga + FT_POSS * fta)
    return (MC._banked_pts(d) / denom) if denom > 0 else None


def _quality(d, node, by):
    gp = d.get("gp_played_asof") or 0.0
    if by == "ts":
        return _ts(d)
    if by == "mpg":
        return (d.get("min_asof") / gp) if gp else None
    if by == "pra":
        return ((MC._banked_pts(d) + (d.get("reb") or 0.0) + (d.get("ast") or 0.0)) / gp) if gp else None
    # vol: banked per-minute node rate
    vc, vm = V._vol_count(d, node)
    return (vc / vm) if (vc is not None and vm > 0) else None


def _ratios(d, fd, rc, node):
    """(rate, min, games) ratios of realised remainder to banked-pace projection,
    or None if the remainder or banked exposure is too thin to be meaningful."""
    gp = d.get("gp_played_asof") or 0.0
    mn = d.get("min_asof") or 0.0
    vc, vm = V._vol_count(d, node)
    fc, fm = V._vol_count(fd, node)
    if vc is None or fc is None or vm < 100.0 or gp < 5:
        return None
    rem_cnt = fc - vc
    rem_min = fm - vm
    rem_games = (fd.get("gp_played_asof") or 0.0) - gp
    if rem_min < 100.0 or rem_games < 5:
        return None
    banked_rate = vc / vm
    realised_rate = rem_cnt / rem_min if rem_min > 0 else None
    rate_r = (realised_rate / banked_rate) if (banked_rate > 0 and realised_rate is not None) else None
    # banked-pace extrapolation of remaining games and minutes
    banked_team = rc["ftg"] - rc["rem_team"]
    gpg = (gp / banked_team) if banked_team > 0 else None
    proj_games = gpg * rc["rem_team"] if gpg is not None else None
    proj_min = proj_games * (mn / gp) if (proj_games is not None and gp) else None
    games_r = (rem_games / proj_games) if (proj_games and proj_games > 0) else None
    min_r = (rem_min / proj_min) if (proj_min and proj_min > 0) else None
    return rate_r, min_r, games_r


def collect(B, season, stat):
    node = NODE[stat]
    eff_real = MC.realised_eff(B["finals"], B["ftg"], season, stat)
    if not eff_real:
        return []
    ev_top5 = set(sorted(eff_real, key=eff_real.get, reverse=True)[:5])
    snaps = sorted(B["ctx"].keys())
    rows = []
    for si, snap in enumerate(snaps):
        ctx_snap = B["ctx"].get(snap, {})
        field = MC._field_at(B["counts"], ctx_snap, season, snap, stat, MC.FIELD_N)
        if not field:
            continue
        ph = _phase(si, len(snaps))
        for rank, pid in enumerate(field):
            d = B["counts"].get((season, snap, pid))
            fd = B["finals"].get((season, pid))
            rc = ctx_snap.get(pid)
            if not d or not fd or not rc:
                continue
            rr = _ratios(d, fd, rc, node)
            if rr is None:
                continue
            rate_r, min_r, games_r = rr
            rows.append({"phase": ph, "rank": rank, "rate": rate_r, "min": min_r,
                         "games": games_r, "q": _quality(d, node, ARGS_BY),
                         "ev_top5": pid in ev_top5})
    return rows


def _med(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.median(xs)) if xs else float("nan")


def _fracgt1(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return (float(np.mean([x > 1.0 for x in xs])) if xs else float("nan"))


def _rank_bucket(r):
    return "top5" if r < 5 else ("6-15" if r < 15 else "16-30")


def panel1(stat, rows):
    print("  [1] drift by as-of rank  (rate/min/games = realised remainder / banked-pace projection)")
    print(f"    {'bucket':>7} {'phase':>6} {'rate':>6} {'min':>6} {'games':>6} {'rate>1':>7} {'n':>6}")
    for b in ("top5", "6-15", "16-30"):
        for ph in PHASES:
            r = [x for x in rows if _rank_bucket(x["rank"]) == b and x["phase"] == ph]
            if not r:
                continue
            print(f"    {b:>7} {ph:>6} {_med([x['rate'] for x in r]):>6.3f} "
                  f"{_med([x['min'] for x in r]):>6.3f} {_med([x['games'] for x in r]):>6.3f} "
                  f"{_fracgt1([x['rate'] for x in r]):>7.3f} {len(r):>6}")


def panel2(stat, rows, by):
    qs = sorted(x["q"] for x in rows if x["q"] is not None and np.isfinite(x["q"]))
    if len(qs) < 30:
        print(f"  [2] quality axis={by}: too few values")
        return
    lo, hi = np.quantile(qs, 1 / 3), np.quantile(qs, 2 / 3)

    def tier(q):
        if q is None or not np.isfinite(q):
            return None
        return "lo" if q < lo else ("mid" if q < hi else "hi")

    print(f"  [2] rate drift by as-of quality tier ({by}: lo/mid/hi terciles, cuts {lo:.3f}/{hi:.3f})")
    print(f"    {'tier':>5} {'phase':>6} {'rate':>6} {'rate>1':>7} {'n':>6}")
    for t in ("hi", "mid", "lo"):
        for ph in PHASES:
            r = [x for x in rows if tier(x["q"]) == t and x["phase"] == ph]
            if not r:
                continue
            print(f"    {t:>5} {ph:>6} {_med([x['rate'] for x in r]):>6.3f} "
                  f"{_fracgt1([x['rate'] for x in r]):>7.3f} {len(r):>6}")


def panel3(stat, rows):
    print("  [3] survivorship  (rate drift: AS-OF top5 vs EVENTUAL final top5; gap = selection)")
    print(f"    {'group':>12} {'phase':>6} {'rate':>6} {'n':>6}")
    for label, sel in (("as-of top5", lambda x: x["rank"] < 5),
                       ("eventual top5", lambda x: x["ev_top5"])):
        for ph in PHASES:
            r = [x for x in rows if sel(x) and x["phase"] == ph]
            if not r:
                continue
            print(f"    {label:>12} {ph:>6} {_med([x['rate'] for x in r]):>6.3f} {len(r):>6}")


NODE_GAMECOL = {"reb": "rebounds", "pts": "points", "ast": "assists"}


def panel4(conn, stat, seasons):
    """Leaguewide within-player H2/H1 per-min rate, all players with >=10 games in
    each half. Common-mode baseline: contender drift above this is contender-specific."""
    node = NODE[stat]
    ratios = []
    for s in seasons:
        if s < STAT_FLOOR[stat]:
            continue
        by = defaultdict(list)
        for r in conn.execute(
            "SELECT nba_api_id, game_date, minutes, points, rebounds, assists, "
            "fga, fta, turnovers FROM stg_nba_player_game_logs "
            "WHERE season=? AND minutes IS NOT NULL AND minutes>0 "
            "ORDER BY game_date", (s,)):
            by[r["nba_api_id"]].append(dict(r))
        dates = sorted({g["game_date"] for gs in by.values() for g in gs})
        if len(dates) < 20:
            continue
        mid = dates[len(dates) // 2]
        for pid, gs in by.items():
            h1 = [g for g in gs if g["game_date"] < mid]
            h2 = [g for g in gs if g["game_date"] >= mid]
            if len(h1) < 10 or len(h2) < 10:
                continue

            def rate(half):
                cnt = mn = 0.0
                for g in half:
                    m = g["minutes"] or 0.0
                    if m <= 0:
                        continue
                    mn += m
                    if node == "usage":
                        cnt += (g["fga"] or 0.0) + FT_POSS * (g["fta"] or 0.0) + (g["turnovers"] or 0.0)
                    else:
                        cnt += g[NODE_GAMECOL[stat]] or 0.0
                return (cnt / mn) if mn > 0 else None

            r1, r2 = rate(h1), rate(h2)
            if r1 and r2 and r1 > 0:
                ratios.append(r2 / r1)
    print("  [4] league baseline  (all players, within-player H2/H1 per-min rate)")
    print(f"    median={_med(ratios):.3f}  frac>1={_fracgt1(ratios):.3f}  n_players={len(ratios)}")


ARGS_BY = "ts"


def main(argv=None):
    global ARGS_BY
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Stat-leader within-season drift diagnostic.")
    p.add_argument("--db", default="data/awards.db")
    p.add_argument("--stat", default="all", choices=["reb", "pts", "ast", "all"])
    p.add_argument("--eval-min", type=int, default=2008)
    p.add_argument("--eval-max", type=int, default=2023)
    p.add_argument("--fit-lookback", type=int, default=10)
    p.add_argument("--by", default="ts", choices=["ts", "vol", "pra", "mpg"],
                   help="as-of quality axis for panel 2 (ts default; mpg is a contrast only)")
    args = p.parse_args(argv)
    ARGS_BY = args.by

    stats = ["reb", "pts", "ast"] if args.stat == "all" else [args.stat]
    try:
        from scripts.common.config import assert_not_sealed
    except ImportError:
        from config import assert_not_sealed  # type: ignore
    seasons = list(range(args.eval_min, args.eval_max + 1))
    for st in stats:
        for s in seasons:
            assert_not_sealed(MC.STAT_AWARD[st], s)

    pooled = {st: [] for st in stats}
    conn = connect(args.db)
    for s in seasons:
        active = [st for st in stats if s >= STAT_FLOOR[st]]
        if not active:
            continue
        try:
            B = MC.load_all(conn, s, args.fit_lookback)
        except Exception as e:  # noqa: BLE001
            log.warning("season %d skipped (%s)", s, e)
            continue
        for st in active:
            pooled[st].extend(collect(B, s, st))

    for st in stats:
        print("\n" + "=" * 78)
        print(f"stat={st}  within-season drift  seasons={args.eval_min}-{args.eval_max}  quality={args.by}")
        print("  ratio > 1 => realised remainder exceeds the banked-pace projection")
        print("-" * 78)
        if not pooled[st]:
            print("  no rows")
            print("=" * 78)
            continue
        panel1(st, pooled[st])
        print()
        panel2(st, pooled[st], args.by)
        print()
        panel3(st, pooled[st])
        print()
        panel4(conn, st, seasons)
        print("=" * 78)
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
