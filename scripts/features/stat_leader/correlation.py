"""Stat-leader arm: named-driver correlation, stage one (remaining opponent).

The P(lead) defect on REB is an order-statistic problem: the means identify the
leader (separation ~0.78) but independent contender draws let the pack leapfrog,
so rank1's P(lead) sits far below its realised rate. The disciplined fix is to
draw contenders sharing named, measured drivers so the gap between them has the
right (lower) variance, and to attribute the co-movement to concrete causes rather
than a blanket latent factor.

This stage builds the first and cheapest named driver, remaining-opponent
strength, and does the two things drivers do, kept separate:

  MU SHARPENING (this file, applied in the MC). Each contender's remaining
  per-minute rate is nudged by how soft his remaining schedule is: facing weaker
  defences (higher opponent def_rating) over the remainder should raise a scorer's
  or rebounder's projected rate. This is a genuine forward edge a leaderboard
  market misses, valuable independent of correlation, and it induces correlation
  implicitly because contenders with overlapping remaining schedules get correlated
  nudges. The coefficient BETA is MEASURED, not tuned: a within-player regression
  of realised per-game per-minute rate on the opponent's as-of defensive strength,
  pooled over training seasons. A node whose rate does not respond to the opponent
  (BETA ~ 0) simply no-ops, which is the honest outcome for, say, usage volume.

  ATTRIBUTION (this file, printed by main). How dispersed the resulting nudges are
  and how correlated they are across contenders within a snapshot, so we can see
  what fraction of the comove co-movement the opponent driver plausibly carries
  before deciding whether the shared-factor layer (league-trend, opponent-overlap
  covariance) is worth building.

Opponent strength is opponent def_rating from team_records (keyed team_id,
snapshot_date), read strictly as-of. REB, PTS and AST all use the same opponent
axis; a weak defence means more scoring, more missed shots to rebound and more
open passing lanes, so the sign is shared and the magnitude is left to the data.
AST creation has no per-game substrate, so its BETA is measured on assists per
minute as a proxy.

Everything is walk-forward and robust to a missing team_records table: with no
opponent ratings, BETA is empty and opp_z is absent, so the MC nudge is exactly
1.0 and --corr nests to the base engine.

Run (measured coefficients + attribution):
  caffeinate -i uv run python3 -m scripts.features.stat_leader.correlation \
      --stat all --eval-min 2008 --eval-max 2023
"""

from __future__ import annotations

import argparse
import bisect
import logging
import sys
from collections import defaultdict

import numpy as np

try:
    from scripts.common.db import connect
    from scripts.features.stat_leader import remaining_schedule as RS
except ImportError:  # pragma: no cover
    from db import connect  # type: ignore
    import remaining_schedule as RS  # type: ignore

log = logging.getLogger("stat_leader.correlation")

FT_POSS = 0.44
MU_CLIP = (0.90, 1.10)          # nudge is a modest schedule tilt, not a free lever
MIN_REG = 200                    # min game rows before a BETA is trusted
NODE_GAMECOL = {"reb": "rebounds", "pts": "usage", "ast": "assists"}
STAT_NODE = {"reb": "reb", "pts": "usage", "ast": "ast_create"}
STAT_FLOOR = {"pts": 1997, "reb": 1997, "ast": 2013}


def _load_team_drtg(conn):
    """team_records def_rating as-of: {team_id: (dates[], drtg[])} sorted by date,
    plus a leaguewide {date: (mean, sd)} for standardising. Empty if the table or
    column is absent (then correlation no-ops)."""
    try:
        rows = list(conn.execute(
            "SELECT team_id, snapshot_date, def_rating FROM team_records "
            "WHERE def_rating IS NOT NULL ORDER BY team_id, snapshot_date"))
    except Exception as e:  # noqa: BLE001
        log.warning("team_records def_rating unavailable (%s); correlation no-ops", e)
        return {}, []
    by = defaultdict(lambda: ([], []))
    at_date = defaultdict(list)
    for r in rows:
        d, v = r["snapshot_date"], r["def_rating"]
        by[r["team_id"]][0].append(d)
        by[r["team_id"]][1].append(v)
        at_date[d].append(v)
    league = sorted((d, float(np.mean(v)), float(np.std(v) or 1.0)) for d, v in at_date.items())
    return dict(by), league


def _asof(series, date):
    dates, vals = series
    i = bisect.bisect_right(dates, date) - 1
    return vals[i] if i >= 0 else None


def _league_asof(league, date):
    ds = [x[0] for x in league]
    i = bisect.bisect_right(ds, date) - 1
    return (league[i][1], league[i][2]) if i >= 0 else (None, None)


def _opp_z(team_drtg, league, opp_team, date):
    v = _asof(team_drtg.get(opp_team, ([], [])), date)
    m, s = _league_asof(league, date)
    if v is None or m is None or not s:
        return None
    return (v - m) / s


def fit_beta(conn, seasons):
    """Within-player regression of per-game per-minute node rate deviation on the
    opponent's as-of def_rating z-score, pooled over training seasons. Positive
    BETA => softer opponent (higher def_rating) lifts the rate. Per node."""
    team_drtg, league = _load_team_drtg(conn)
    if not team_drtg:
        return {}
    seasons = [s for s in seasons]
    qs = ",".join("?" * len(seasons))
    logs = defaultdict(list)   # (season, pid) -> [(date, opp, minutes, reb, usage, ast)]
    for r in conn.execute(
        f"SELECT season, nba_api_id, game_date, opp_team_id, minutes, rebounds, "
        f"assists, fga, fta, turnovers FROM stg_nba_player_game_logs "
        f"WHERE season IN ({qs}) AND minutes IS NOT NULL AND minutes>0 "
        f"AND opp_team_id IS NOT NULL", seasons):
        usage = (r["fga"] or 0.0) + FT_POSS * (r["fta"] or 0.0) + (r["turnovers"] or 0.0)
        logs[(r["season"], r["nba_api_id"])].append(
            (r["game_date"], r["opp_team_id"], r["minutes"],
             {"rebounds": r["rebounds"] or 0.0, "usage": usage, "assists": r["assists"] or 0.0}))
    beta = {}
    for stat, gcol in NODE_GAMECOL.items():
        X, Y = [], []
        for (s, pid), games in logs.items():
            rates, zs = [], []
            for gdate, opp, mn, cnt in games:
                if mn <= 0:
                    continue
                z = _opp_z(team_drtg, league, opp, gdate)
                if z is None:
                    continue
                rates.append(cnt[gcol] / mn)
                zs.append(z)
            if len(rates) < 10:
                continue
            pm = float(np.mean(rates))
            if pm <= 0:
                continue
            for rt, z in zip(rates, zs):
                X.append(z)
                Y.append(rt / pm - 1.0)          # fractional deviation from player mean
        if len(X) >= MIN_REG:
            X = np.asarray(X); Y = np.asarray(Y)
            b = float(np.polyfit(X, Y, 1)[0])
            beta[STAT_NODE[stat]] = b
    return beta


def attach_opp_z(conn, counts, ctx, season):
    """Fill counts[(season, snap, pid)]['opp_z'] with the standardised softness of
    the contender's remaining schedule, as-of each snapshot. Resolves the player's
    team from his most recent logged game <= snapshot, then averages the as-of
    def_rating z of his remaining opponents. Cached per (snap, team)."""
    team_drtg, league = _load_team_drtg(conn)
    if not team_drtg:
        return
    full = RS.full_schedule(conn, season)
    # player -> sorted [(game_date, team_id)] for as-of team resolution
    pteam = defaultdict(list)
    for r in conn.execute(
        "SELECT nba_api_id, game_date, team_id FROM stg_nba_player_game_logs "
        "WHERE season=? AND team_id IS NOT NULL ORDER BY nba_api_id, game_date", (season,)):
        pteam[r["nba_api_id"]].append((r["game_date"], r["team_id"]))

    def team_asof(pid, snap):
        seq = pteam.get(pid)
        if not seq:
            return None
        i = bisect.bisect_right([d for d, _ in seq], snap) - 1
        return seq[i][1] if i >= 0 else None

    team_z_cache = {}

    def team_z(team, snap):
        key = (snap, team)
        if key in team_z_cache:
            return team_z_cache[key]
        rem = RS.remaining_asof(full, team, snap)
        zs = [_opp_z(team_drtg, league, opp, snap) for _, opp in rem]
        zs = [z for z in zs if z is not None]
        val = float(np.mean(zs)) if zs else None
        team_z_cache[key] = val
        return val

    for snap, players in ctx.items():
        for pid in players:
            key = (season, snap, pid)
            d = counts.get(key)
            if not d:
                continue
            t = team_asof(pid, snap)
            z = team_z(t, snap) if t is not None else None
            if z is not None:
                d["opp_z"] = z


def _phase(i, n):
    if n <= 1:
        return "mid"
    f = i / (n - 1)
    return "early" if f < 1 / 3 else ("mid" if f < 2 / 3 else "late")


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Named-driver correlation stage one (remaining opponent).")
    p.add_argument("--db", default="data/awards.db")
    p.add_argument("--stat", default="all", choices=["reb", "pts", "ast", "all"])
    p.add_argument("--eval-min", type=int, default=2008)
    p.add_argument("--eval-max", type=int, default=2023)
    p.add_argument("--fit-lookback", type=int, default=10)
    args = p.parse_args(argv)
    stats = ["reb", "pts", "ast"] if args.stat == "all" else [args.stat]

    conn = connect(args.db)
    beta = fit_beta(conn, list(range(args.eval_min - args.fit_lookback, args.eval_max + 1)))
    print("\n" + "=" * 74)
    print("measured opponent BETA  (fractional per-min rate change per z of opponent")
    print("def_rating; positive => softer schedule lifts the rate). node <- stat:")
    if not beta:
        print("  team_records def_rating unavailable: correlation no-ops (opp_z absent).")
    for stat in ("reb", "pts", "ast"):
        node = STAT_NODE[stat]
        b = beta.get(node)
        print(f"  {stat:>4} ({node:>10})  BETA={'n/a' if b is None else f'{b:+.4f}'}"
              + ("  [ast measured on assists proxy]" if stat == "ast" else ""))
    print("=" * 74)

    # attribution: dispersion of the nudge and within-snapshot cross-contender
    # correlation of opp_z (the schedule-overlap co-movement the driver induces).
    try:
        from scripts.modelling.stat_leader import mc as MC
    except ImportError:  # pragma: no cover
        import mc as MC  # type: ignore
    for stat in stats:
        node = STAT_NODE[stat]
        b = beta.get(node)
        rows_mu = defaultdict(list)      # phase -> [mu-1]
        snap_z = defaultdict(list)       # (season,snap) -> [opp_z over field]
        for s in range(args.eval_min, args.eval_max + 1):
            if s < STAT_FLOOR[stat]:
                continue
            B = MC.load_all(conn, s, args.fit_lookback)
            attach_opp_z(conn, B["counts"], B["ctx"], s)
            snaps = sorted(B["ctx"].keys())
            for si, snap in enumerate(snaps):
                field = MC._field_at(B["counts"], B["ctx"].get(snap, {}), s, snap, stat, MC.FIELD_N)
                zs = []
                for pid in field:
                    d = B["counts"].get((s, snap, pid))
                    z = d.get("opp_z") if d else None
                    if z is None:
                        continue
                    zs.append(z)
                    if b is not None:
                        mu = min(MU_CLIP[1], max(MU_CLIP[0], 1.0 + b * z))
                        rows_mu[_phase(si, len(snaps))].append(mu - 1.0)
                if len(zs) >= 5:
                    snap_z[(s, snap)] = zs
        print(f"\nstat={stat}  node={node}  attribution")
        print(f"  {'phase':>6} {'mean|mu-1|':>11} {'sd(mu-1)':>10} {'n':>7}")
        for ph in ("early", "mid", "late"):
            v = rows_mu.get(ph, [])
            if not v:
                continue
            a = np.abs(v)
            print(f"  {ph:>6} {float(np.mean(a)):>11.4f} {float(np.std(v)):>10.4f} {len(v):>7}")
        # cross-contender opp_z correlation proxy: variance of the field-mean z
        # relative to the total z variance = share of co-movement the driver shares
        allz = [z for zs in snap_z.values() for z in zs]
        if allz:
            within = float(np.mean([np.var(zs) for zs in snap_z.values()]))
            field_means = [float(np.mean(zs)) for zs in snap_z.values()]
            between = float(np.var(field_means)) if len(field_means) > 1 else 0.0
            icc = between / (between + within) if (between + within) > 0 else 0.0
            print(f"  opp_z shared-share (between-snapshot / total) = {icc:.3f}  "
                  f"(compare to comove ICC for this stat)")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
