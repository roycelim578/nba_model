"""Stat-leader arm: mid-season-trade availability audit (read-only, diagnostic).

Hypothesis: a player traded mid-season gets a broken team_games_asof in
stg_nba_availability_asof (the new team's low count against his cumulative
games-played), so games_played_asof > team_games_asof, availability_pct blows
past 1, rem_team goes non-positive, and the stat-leader field/context drops him
(the Andre Drummond 2019 effect). The voter arm reads the same availability block,
so this also quantifies voter-side exposure.

Mutates nothing. Two modes:
  --trace: dump the realised leader's availability rows for one (stat, season),
           flagging the break signature (games_played_asof > team_games_asof).
  --scan:  across seasons, list players traded mid-season (>1 team in the game
           logs), how many show the break signature, and which were top-N by
           final per-game in any stat (the ones that actually cost the field),
           plus a crude voter-exposure count (traded, broken, >=25 mpg).

Run:
  uv run python3 -m scripts.modelling.stat_leader.trade_audit --trace --stat reb --season 2019
  uv run python3 -m scripts.modelling.stat_leader.trade_audit --scan --eval-min 2008 --eval-max 2023
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict

try:
    from scripts.common.db import connect
    from scripts.modelling.stat_leader import mc as MC
except ImportError:  # pragma: no cover
    from db import connect  # type: ignore
    import mc as MC  # type: ignore

log = logging.getLogger("stat_leader.trade_audit")

STAT_SQL = {"reb": "rebounds", "pts": "points", "ast": "assists"}
TOP_N = 30


def _name_map(conn):
    try:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(players)")]
    except Exception:  # noqa: BLE001
        return {}
    idc = next((c for c in ("nba_api_id", "nba_id", "player_nba_id") if c in cols), None)
    nmc = next((c for c in ("full_name", "display_name", "player_name", "name") if c in cols), None)
    if not idc or not nmc:
        return {}
    return {r[0]: r[1] for r in conn.execute(f"SELECT {idc}, {nmc} FROM players WHERE {idc} IS NOT NULL")}


def _traded(conn, season):
    """{nba_api_id: n_teams} for players with >1 team_id in the season's logs."""
    out = {}
    for r in conn.execute(
        "SELECT nba_api_id, COUNT(DISTINCT team_id) nt FROM stg_nba_player_game_logs "
        "WHERE season=? AND team_id IS NOT NULL GROUP BY nba_api_id HAVING nt>1", (season,)):
        out[r["nba_api_id"]] = r["nt"]
    return out


def _avail_rows(conn, season, pid):
    return list(conn.execute(
        "SELECT snapshot_date, games_played_asof gp, team_games_asof tg, "
        "availability_pct_asof pct, current_absence_streak abs FROM stg_nba_availability_asof "
        "WHERE season=? AND nba_api_id=? ORDER BY snapshot_date", (season, pid)))


def _broken(rows):
    """Snapshots with the break signature: more games played than team games."""
    return [r for r in rows if (r["tg"] or 0) > 0 and (r["gp"] or 0) > (r["tg"] or 0) + 0.5]


def _final_pergame(conn, season, col):
    out = {}
    for r in conn.execute(
        f"SELECT nba_api_id, SUM({col})*1.0/NULLIF(COUNT(*),0) pg, COUNT(*) g, "
        f"AVG(minutes) mpg FROM stg_nba_player_game_logs WHERE season=? "
        f"GROUP BY nba_api_id", (season,)):
        out[r["nba_api_id"]] = (r["pg"] or 0.0, r["g"] or 0, r["mpg"] or 0.0)
    return out


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Mid-season-trade availability audit.")
    p.add_argument("--db", default="data/awards.db")
    p.add_argument("--trace", action="store_true")
    p.add_argument("--scan", action="store_true")
    p.add_argument("--stat", default="reb", choices=["reb", "pts", "ast"])
    p.add_argument("--season", type=int, default=2019)
    p.add_argument("--eval-min", type=int, default=2008)
    p.add_argument("--eval-max", type=int, default=2023)
    p.add_argument("--fit-lookback", type=int, default=10)
    args = p.parse_args(argv)

    conn = connect(args.db)
    names = _name_map(conn)

    if args.trace:
        B = MC.load_all(conn, args.season, args.fit_lookback)
        eff = MC.realised_eff(B["finals"], B["ftg"], args.season, args.stat)
        leader = max(eff, key=eff.get)
        rows = _avail_rows(conn, args.season, leader)
        brk = _broken(rows)
        print("\n" + "=" * 70)
        print(f"trace: {args.stat.upper()} {args.season} realised leader "
              f"{names.get(leader, leader)} (id={leader}); teams in logs="
              f"{_traded(conn, args.season).get(leader, 1)}")
        print(f"  break snapshots (games_played > team_games): {len(brk)} of {len(rows)}")
        print("-" * 70)
        print(f"  {'snapshot':>12} {'gp':>5} {'team_g':>7} {'pct':>6} {'absStreak':>9} {'BREAK':>6}")
        for r in rows:
            flag = "<<<" if (r["tg"] or 0) > 0 and (r["gp"] or 0) > (r["tg"] or 0) + 0.5 else ""
            print(f"  {r['snapshot_date']:>12} {r['gp'] or 0:>5.0f} {r['tg'] or 0:>7.0f} "
                  f"{(r['pct'] or 0):>6.2f} {r['abs'] or 0:>9.0f} {flag:>6}")
        print("=" * 70)

    if args.scan:
        print("\n" + "=" * 78)
        print("scan: mid-season-traded players, availability break signature, and field cost")
        print("-" * 78)
        print(f"  {'season':>6} {'traded':>7} {'broken':>7} {'contenders_broken':>18}")
        costly = []
        vexposed = 0
        for s in range(args.eval_min, args.eval_max + 1):
            traded = _traded(conn, s)
            if not traded:
                continue
            fp = {st: _final_pergame(conn, s, col) for st, col in STAT_SQL.items()}
            ranks = {st: {pid: i for i, (pid, _) in enumerate(
                sorted(v.items(), key=lambda kv: kv[1][0], reverse=True))} for st, v in fp.items()}
            nbrk, ncont = 0, 0
            for pid in traded:
                rows = _avail_rows(conn, s, pid)
                if not _broken(rows):
                    continue
                nbrk += 1
                mpg = max((fp[st].get(pid, (0, 0, 0))[2] for st in STAT_SQL), default=0)
                if mpg >= 25:
                    vexposed += 1
                topstat = next((st for st in STAT_SQL if ranks[st].get(pid, 999) < TOP_N), None)
                if topstat:
                    ncont += 1
                    costly.append((s, pid, topstat, ranks[topstat][pid] + 1))
            print(f"  {s:>6} {len(traded):>7} {nbrk:>7} {ncont:>18}")
        print("-" * 78)
        print(f"  traded+broken players who were top-{TOP_N} in a stat (cost the field):")
        for s, pid, st, rk in sorted(costly):
            print(f"    {s}  {names.get(pid, pid):>26}  {st}  final-rank {rk}")
        print(f"  crude voter-exposure (traded, broken, >=25 mpg): {vexposed}")
        print("=" * 78)
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
