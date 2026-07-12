"""Stat-leader arm: trunk derivations.

Builds the three foundational derived tables every stat-leader node and the
Monte Carlo depend on, from data already in awards.db. No new pulls. Writes only
into the ``stat_`` namespace, so nothing the v1 sealed gate reads is touched.

Season is the STARTING year, inherited verbatim from the game logs.

TABLES BUILT
------------
stat_team_game   (season, team_id, game_id): per team-game box aggregate plus
    offensive and defensive possessions. Offensive possessions use the standard
    OREB-free estimator FGA + 0.44*FTA + TOV, because the game logs carry only a
    total ``rebounds`` column with no offensive/defensive split; this is the
    accepted fallback when OREB is unavailable and is accurate to within a
    possession or two per game. Defensive possessions are the opponent's
    offensive possessions in the same game, resolved via the two team_ids
    present under each game_id.

stat_qualifier   (season, team_id): team scheduled games and the per-game leader
    qualifier q = ceil(0.70 * team_games). This is the soft denominator floor in
    eff_value = season_total / max(games_played, q).

stat_team_fg_asof (season, snapshot_date, team_id): cumulative team field-goal
    makes and attempts (overall and three-point) over games on or before each
    weekly/ratings snapshot. The AST conversion prior is team-FG-excluding-self;
    the ex-self subtraction (minus the player's own cumulative makes/attempts) is
    a trivial downstream step, so this is stored at team level rather than
    materialised per player.

Run from repo root:
  uv run python -m scripts.features.stat_leader.trunk
  uv run python -m scripts.features.stat_leader.trunk --season 2023
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path

try:
    from scripts.common.db import connect, upsert, utcnow_iso
except ImportError:  # pragma: no cover
    from db import connect, upsert, utcnow_iso  # type: ignore

log = logging.getLogger("stat_leader.trunk")

DEFAULT_DB_PATH = Path("data/awards.db")
FT_POSS_COEF = 0.44
QUAL_FRAC = 0.70


def _ensure_schema(conn) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS stat_team_game ("
        "season INTEGER NOT NULL, team_id INTEGER NOT NULL, game_id TEXT NOT NULL, "
        "game_date TEXT, opp_team_id INTEGER, "
        "fga REAL, fgm REAL, fg3a REAL, fg3m REAL, fta REAL, ftm REAL, tov REAL, "
        "team_minutes REAL, off_poss REAL, def_poss REAL, "
        "off_poss_p48 REAL, def_poss_p48 REAL, pulled_at TEXT, "
        "PRIMARY KEY (season, team_id, game_id))"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS stat_qualifier ("
        "season INTEGER NOT NULL, team_id INTEGER NOT NULL, "
        "team_games INTEGER, q INTEGER, pulled_at TEXT, "
        "PRIMARY KEY (season, team_id))"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS stat_team_fg_asof ("
        "season INTEGER NOT NULL, snapshot_date TEXT NOT NULL, team_id INTEGER NOT NULL, "
        "team_fgm_asof REAL, team_fga_asof REAL, team_fg3m_asof REAL, team_fg3a_asof REAL, "
        "team_games_asof INTEGER, pulled_at TEXT, "
        "PRIMARY KEY (season, snapshot_date, team_id))"
    )
    conn.commit()


def _seasons(conn, season: int | None) -> list[int]:
    if season is not None:
        return [season]
    return [r[0] for r in conn.execute(
        "SELECT DISTINCT season FROM stg_nba_player_game_logs ORDER BY season")]


def build_team_game(conn, season: int, pulled_at: str) -> int:
    """Aggregate player game logs to team-game level and compute possessions."""
    rows = conn.execute(
        "SELECT season, team_id, game_id, MIN(game_date) game_date, "
        "MAX(opp_team_id) opp_team_id, "
        "SUM(fga) fga, SUM(fgm) fgm, SUM(fg3a) fg3a, SUM(fg3m) fg3m, "
        "SUM(fta) fta, SUM(ftm) ftm, SUM(turnovers) tov, SUM(minutes) team_minutes "
        "FROM stg_nba_player_game_logs WHERE season = ? "
        "GROUP BY season, team_id, game_id",
        (season,),
    ).fetchall()

    agg = {}
    off_by_game = defaultdict(dict)  # game_id -> {team_id: off_poss}
    for r in rows:
        d = dict(r)
        fga = d["fga"] or 0.0
        fta = d["fta"] or 0.0
        tov = d["tov"] or 0.0
        off_poss = fga + FT_POSS_COEF * fta + tov
        d["off_poss"] = off_poss
        agg[(d["team_id"], d["game_id"])] = d
        off_by_game[d["game_id"]][d["team_id"]] = off_poss

    out = []
    unresolved = 0
    for (team_id, game_id), d in agg.items():
        opps = [t for t in off_by_game[game_id] if t != team_id]
        def_poss = off_by_game[game_id][opps[0]] if len(opps) == 1 else None
        if def_poss is None:
            unresolved += 1
        tm = d["team_minutes"] or 0.0
        game_minutes = tm / 5.0 if tm else 0.0
        p48 = (48.0 / game_minutes) if game_minutes > 0 else None
        out.append({
            "season": season, "team_id": team_id, "game_id": game_id,
            "game_date": d["game_date"], "opp_team_id": d["opp_team_id"],
            "fga": d["fga"], "fgm": d["fgm"], "fg3a": d["fg3a"], "fg3m": d["fg3m"],
            "fta": d["fta"], "ftm": d["ftm"], "tov": d["tov"], "team_minutes": tm,
            "off_poss": d["off_poss"], "def_poss": def_poss,
            "off_poss_p48": (d["off_poss"] * p48) if p48 else None,
            "def_poss_p48": (def_poss * p48) if (p48 and def_poss is not None) else None,
            "pulled_at": pulled_at,
        })
    if out:
        upsert(conn, "stat_team_game", out,
               ["season", "team_id", "game_id"])
    conn.commit()
    if unresolved:
        log.warning("season %d: %d team-games with unresolved opponent (def_poss NULL)",
                    season, unresolved)
    return len(out)


def build_qualifier(conn, season: int, pulled_at: str) -> int:
    rows = conn.execute(
        "SELECT team_id, COUNT(DISTINCT game_id) g FROM stg_nba_player_game_logs "
        "WHERE season = ? GROUP BY team_id",
        (season,),
    ).fetchall()
    out = [{
        "season": season, "team_id": r["team_id"], "team_games": r["g"],
        "q": math.ceil(QUAL_FRAC * r["g"]), "pulled_at": pulled_at,
    } for r in rows]
    if out:
        upsert(conn, "stat_qualifier", out, ["season", "team_id"])
    conn.commit()
    return len(out)


def build_fg_asof(conn, season: int, pulled_at: str) -> int:
    """Cumulative team FG makes/attempts at each weekly/ratings snapshot."""
    grid = [r["snapshot_date"] for r in conn.execute(
        "SELECT snapshot_date FROM snapshot_grid "
        "WHERE season = ? AND snapshot_kind IN ('weekly','ratings') "
        "ORDER BY snapshot_date", (season,))]
    if not grid:
        return 0
    tg = conn.execute(
        "SELECT team_id, game_date, fgm, fga, fg3m, fg3a FROM stat_team_game "
        "WHERE season = ? ORDER BY team_id, game_date", (season,)).fetchall()
    by_team = defaultdict(list)
    for r in tg:
        by_team[r["team_id"]].append(dict(r))

    out = []
    for team_id, games in by_team.items():
        games.sort(key=lambda x: x["game_date"])
        gi = 0
        cfgm = cfga = cfg3m = cfg3a = 0.0
        cg = 0
        for snap in grid:
            while gi < len(games) and games[gi]["game_date"] is not None \
                    and games[gi]["game_date"] <= snap:
                g = games[gi]
                cfgm += g["fgm"] or 0.0; cfga += g["fga"] or 0.0
                cfg3m += g["fg3m"] or 0.0; cfg3a += g["fg3a"] or 0.0
                cg += 1; gi += 1
            out.append({
                "season": season, "snapshot_date": snap, "team_id": team_id,
                "team_fgm_asof": cfgm, "team_fga_asof": cfga,
                "team_fg3m_asof": cfg3m, "team_fg3a_asof": cfg3a,
                "team_games_asof": cg, "pulled_at": pulled_at,
            })
    if out:
        upsert(conn, "stat_team_fg_asof", out,
               ["season", "snapshot_date", "team_id"])
    conn.commit()
    return len(out)


def _summary(conn) -> None:
    print("\n" + "=" * 66)
    print("stat_leader.trunk summary")
    print("=" * 66)
    for tbl in ("stat_team_game", "stat_qualifier", "stat_team_fg_asof"):
        n = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        print(f"  {tbl}: {n} rows")
    print("\n  league mean off_poss_p48 by season (sanity ~98-102):")
    for r in conn.execute(
        "SELECT season, ROUND(AVG(off_poss_p48),1) pace, "
        "COUNT(*) tg, SUM(CASE WHEN def_poss IS NULL THEN 1 ELSE 0 END) unres "
        "FROM stat_team_game GROUP BY season ORDER BY season"):
        flag = f"  UNRESOLVED={r['unres']}" if r["unres"] else ""
        print(f"    {r['season']}: pace={r['pace']}  team_games={r['tg']}{flag}")
    print("\n  qualifier q by season (min/max across teams; 82-game season -> q=58):")
    for r in conn.execute(
        "SELECT season, MIN(team_games) mn, MAX(team_games) mx, "
        "MIN(q) qmn, MAX(q) qmx FROM stat_qualifier GROUP BY season ORDER BY season"):
        print(f"    {r['season']}: team_games {r['mn']}-{r['mx']}  q {r['qmn']}-{r['qmx']}")
    print("=" * 66)


def build(db_path: Path, season: int | None) -> None:
    pulled_at = utcnow_iso()
    conn = connect(db_path)
    try:
        _ensure_schema(conn)
        for s in _seasons(conn, season):
            ntg = build_team_game(conn, s, pulled_at)
            nq = build_qualifier(conn, s, pulled_at)
            nfg = build_fg_asof(conn, s, pulled_at)
            log.info("season %d: team_game=%d qualifier=%d fg_asof=%d", s, ntg, nq, nfg)
        _summary(conn)
    finally:
        conn.close()


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Stat-leader trunk derivations.")
    p.add_argument("--db", default=str(DEFAULT_DB_PATH))
    p.add_argument("--season", type=int, default=None)
    args = p.parse_args(argv)
    build(Path(args.db), args.season)
    return 0


if __name__ == "__main__":
    sys.exit(main())
