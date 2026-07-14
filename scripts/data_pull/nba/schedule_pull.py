"""Stat-leader arm: live forward-schedule pull -> stg_nba_schedule.

Only needed for the LIVE season, where the future half of the schedule is not yet
in the game logs. For completed seasons the game logs already are the schedule and
``remaining_schedule.full_schedule`` reconstructs it with no pull, so this is not
part of development or gating; it is a deployment-time data pull.

Writes ONLY the nba_api-scoped staging table ``stg_nba_schedule`` keyed by
(game_id). Never touches canonical tables and never reads a realised label. The
table carries one row per game (home/away team ids and the date); the per-team
expansion happens in remaining_schedule.py so both sources share one shape.

Endpoint: nba_api ScheduleLeagueV2 returns the full published season schedule
including unplayed games. If the endpoint name has drifted in the installed
nba_api, adjust IMPORT_ENDPOINT below; the row mapping is defensive about field
names (upper and title case, home/away nesting) and skips rows missing the
natural key rather than inventing one.

Read-only jurisdiction note: pulling the schedule is data, not order placement, so
it is unrestricted everywhere; this does not touch the Polymarket path.

Run:
  uv run python3 -m scripts.data_pull.nba.schedule_pull --season 2025
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone

try:
    from scripts.common.db import connect, upsert
except ImportError:  # pragma: no cover
    from db import connect, upsert  # type: ignore

log = logging.getLogger("nba.schedule_pull")

TABLE = "stg_nba_schedule"
DDL = (
    "CREATE TABLE IF NOT EXISTS stg_nba_schedule ("
    "  game_id TEXT NOT NULL,"
    "  season INTEGER NOT NULL,"
    "  game_date TEXT,"
    "  home_team_id INTEGER,"
    "  away_team_id INTEGER,"
    "  status TEXT,"
    "  pulled_at TEXT,"
    "  PRIMARY KEY (game_id)"
    ")"
)


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _i(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _date(v):
    """Normalise an nba_api schedule date (e.g. '2025-10-21T00:00:00' or
    '10/21/2025 00:00:00') to YYYY-MM-DD."""
    if not v:
        return None
    s = str(v).strip()
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d", "%m/%d/%Y %H:%M:%S", "%m/%d/%Y"):
        try:
            return datetime.strptime(s[:len(fmt) + 4], fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return s[:10] if len(s) >= 10 else None


def _season_str(season: int) -> str:
    return f"{season}-{str(season + 1)[-2:]}"


def fetch_schedule(season: int) -> list[dict]:
    """Return raw game dicts from ScheduleLeagueV2 for the season. Import is local
    so the module loads without nba_api present (e.g. for tests)."""
    from nba_api.stats.endpoints import scheduleleaguev2  # IMPORT_ENDPOINT
    ep = scheduleleaguev2.ScheduleLeagueV2(season=_season_str(season))
    frames = ep.get_data_frames()
    df = frames[0]
    return df.to_dict("records")


def build_rows(records: list[dict], season: int, pulled_at: str) -> tuple[list[dict], int]:
    rows, dropped = [], 0
    for r in records:
        gid = r.get("gameId") or r.get("GAME_ID") or r.get("gameID")
        if not gid:
            dropped += 1
            continue
        home = (r.get("homeTeam_teamId") or r.get("homeTeamId")
                or r.get("HOME_TEAM_ID") or (r.get("homeTeam") or {}).get("teamId"))
        away = (r.get("awayTeam_teamId") or r.get("awayTeamId")
                or r.get("AWAY_TEAM_ID") or (r.get("awayTeam") or {}).get("teamId"))
        gdate = (r.get("gameDate") or r.get("GAME_DATE") or r.get("gameDateEst")
                 or r.get("gameDateTimeEst"))
        rows.append({
            "game_id": str(gid),
            "season": season,
            "game_date": _date(gdate),
            "home_team_id": _i(home),
            "away_team_id": _i(away),
            "status": (r.get("gameStatusText") or r.get("gameStatus") or None),
            "pulled_at": pulled_at,
        })
    return rows, dropped


def pull(conn, season: int) -> int:
    conn.execute(DDL)
    records = fetch_schedule(season)
    rows, dropped = build_rows(records, season, _now())
    if rows:
        upsert(conn, TABLE, rows, ["game_id"])
    conn.commit()
    log.info("season %d: wrote %d schedule rows (%d dropped, no natural key)",
             season, len(rows), dropped)
    return len(rows)


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Live forward-schedule pull -> stg_nba_schedule.")
    p.add_argument("--db", default="data/awards.db")
    p.add_argument("--season", type=int, required=True, help="starting year, e.g. 2025 for 2025-26")
    p.add_argument("--retries", type=int, default=3)
    args = p.parse_args(argv)
    conn = connect(args.db)
    n = 0
    for attempt in range(1, args.retries + 1):
        try:
            n = pull(conn, args.season)
            break
        except Exception as e:  # noqa: BLE001
            log.warning("attempt %d/%d failed: %s", attempt, args.retries, e)
            if attempt < args.retries:
                time.sleep(2 * attempt)
    conn.close()
    return 0 if n else 1


if __name__ == "__main__":
    sys.exit(main())
