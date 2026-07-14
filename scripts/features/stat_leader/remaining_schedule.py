"""Stat-leader arm: as-of remaining-schedule helper (the correlation substrate).

Supplies, for a season and a snapshot date, each team's remaining fixtures
(future game dates and opponents), which the correlation layer turns into
per-contender remaining-opponent strength (mu sharpening) and into shared draws
across contenders who face the same opponents on the same nights (correlation).

TWO SOURCES, one interface. For a COMPLETED season (all development and gating,
2008-2023) the whole fixture list already sits in ``stg_nba_player_game_logs``:
every game has a date, a team and an opponent, so the remaining schedule as-of a
snapshot is simply the games with date strictly after it. No pull is needed to
build or gate the correlation layer; the game logs are the schedule. For the
LIVE season the future half of the schedule is not yet in the logs, so this reads
a forward-schedule table ``stg_nba_schedule`` written by ``schedule_pull.py``.
When both exist for a season the logs take precedence (they are ground truth for
games that have been played) and the forward table fills only dates after the
last logged game.

Team-level, not player-level: a fixture is (game_date, opp_team_id) for a team,
so two contenders on the same team share it exactly and contenders on different
teams share it only through common opponents, which is the correlation we want to
induce rather than invent.

Walk-forward honest: only games with date <= snapshot are treated as banked; only
games with date > snapshot are 'remaining'. Nothing here reads a realised label.

Run (inspection; prints remaining-game counts by team for a snapshot):
  uv run python3 -m scripts.features.stat_leader.remaining_schedule \
      --season 2023 --snapshot 2023-02-01
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict

try:
    from scripts.common.db import connect
except ImportError:  # pragma: no cover
    from db import connect  # type: ignore

log = logging.getLogger("stat_leader.remaining_schedule")

GAME_LOGS = "stg_nba_player_game_logs"
SCHED = "stg_nba_schedule"


def _table_exists(conn, name):
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


def _from_logs(conn, season):
    """Distinct team fixtures (game_date, team_id, opp_team_id) played in the
    season. One row per team per game; both directions of a game appear (each
    team's row), which is what we want for a per-team remaining list."""
    fixtures = defaultdict(list)   # team_id -> [(game_date, opp_team_id), ...]
    seen = set()
    for r in conn.execute(
        f"SELECT DISTINCT game_id, game_date, team_id, opp_team_id "
        f"FROM {GAME_LOGS} WHERE season=? AND team_id IS NOT NULL "
        f"AND opp_team_id IS NOT NULL AND game_date IS NOT NULL", (season,)):
        key = (r["game_id"], r["team_id"])
        if key in seen:
            continue
        seen.add(key)
        fixtures[r["team_id"]].append((r["game_date"], r["opp_team_id"]))
    return fixtures


def _from_forward(conn, season):
    """Forward-schedule fixtures from stg_nba_schedule, expanded to both teams'
    per-team rows. Missing table -> empty."""
    fixtures = defaultdict(list)
    if not _table_exists(conn, SCHED):
        return fixtures
    for r in conn.execute(
        f"SELECT game_date, home_team_id, away_team_id FROM {SCHED} "
        f"WHERE season=? AND game_date IS NOT NULL", (season,)):
        h, a = r["home_team_id"], r["away_team_id"]
        if h is not None and a is not None:
            fixtures[h].append((r["game_date"], a))
            fixtures[a].append((r["game_date"], h))
    return fixtures


def full_schedule(conn, season):
    """Per-team fixture list for the season, logs first then forward-table filling
    dates strictly after each team's last logged game. Each list is date-sorted
    and de-duplicated on (date, opp)."""
    logs = _from_logs(conn, season)
    fwd = _from_forward(conn, season)
    out = {}
    teams = set(logs) | set(fwd)
    for t in teams:
        played = logs.get(t, [])
        last_played = max((d for d, _ in played), default=None)
        future = [(d, o) for (d, o) in fwd.get(t, [])
                  if last_played is None or d > last_played]
        merged = sorted(set(played) | set(future))
        out[t] = merged
    return out


def remaining_asof(full, team_id, snapshot_date):
    """Fixtures for a team with game_date strictly after the snapshot."""
    return [(d, o) for (d, o) in full.get(team_id, []) if d > snapshot_date]


def remaining_opp_strength(remaining, ratings, key="def_rating"):
    """Mean of a per-opponent-team rating over the remaining fixtures and the
    remaining-game count. ``ratings`` is {team_id: {metric: value}} as-of the
    snapshot (supplied by the correlation layer, never a future-leaking value).
    Opponents without a rating are skipped from the mean but still counted."""
    vals = [ratings[o][key] for (_, o) in remaining
            if o in ratings and ratings[o].get(key) is not None]
    n = len(remaining)
    mean = sum(vals) / len(vals) if vals else None
    return mean, n


def team_of(conn, season, nba_api_id, snapshot_date):
    """The team a player was on as of the snapshot: the team of his most recent
    logged game on or before the snapshot. None if he had not yet played."""
    r = conn.execute(
        f"SELECT team_id FROM {GAME_LOGS} WHERE season=? AND nba_api_id=? "
        f"AND game_date<=? AND team_id IS NOT NULL "
        f"ORDER BY game_date DESC LIMIT 1", (season, nba_api_id, snapshot_date)).fetchone()
    return r["team_id"] if r else None


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="As-of remaining-schedule helper.")
    p.add_argument("--db", default="data/awards.db")
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--snapshot", required=True, help="YYYY-MM-DD")
    args = p.parse_args(argv)
    conn = connect(args.db)
    full = full_schedule(conn, args.season)
    src = "logs+forward" if _table_exists(conn, SCHED) else "logs-only"
    rows = sorted(((t, len(remaining_asof(full, t, args.snapshot)), len(full[t]))
                   for t in full), key=lambda x: -x[1])
    conn.close()
    print(f"season={args.season} snapshot={args.snapshot} source={src} teams={len(full)}")
    print(f"  {'team_id':>8} {'remaining':>10} {'season_games':>13}")
    for t, rem, tot in rows:
        print(f"  {t:>8} {rem:>10} {tot:>13}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
