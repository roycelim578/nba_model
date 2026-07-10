"""nba_api puller for the NBA Awards Trader.

Pulls the box-score / advanced-stat / team-record / player-reference data that
the model's STATISTICAL features are built from, via the ``nba_api`` library.
Writes ONLY nba_api-scoped staging + the canonical ``teams`` / ``team_records``
tables. Never allocates ``player_id``; never writes ``players`` /
``player_game_logs`` / ``player_advanced_stats`` / ``injuries`` / any ``pm_*``
or ``stg_bref_*`` table (those belong to other chats / the resolution layer).

SEASON CONVENTION
-----------------
Season is stored as the STARTING year. nba_api season strings are "1996-97";
that is stored as season = 1996. The single conversion site is
``season_to_nba_str`` / ``nba_str_to_season``.

BACKFILL FLOOR (pinned at master level)
---------------------------------------
``GAME_LOG_FLOOR = 1996`` for everything: game logs, advanced stats, team
records, all share one floor. The 1996-2009 block enters the training set as
trajectory-featured but NLP-null rows; that is a deliberate, cost-understood
choice. The effective fully-featured backtest window tracks the GAME-LOG floor
(game logs are the sole source for trajectory features and the entire as-of-
snapshot reconstruction), which is exactly why the floor is deep and not 2010.

BACKWARDS-FILL + RESUME (pinned)
--------------------------------
Seasons are pulled NEWEST-FIRST, so a rate-limit stop always leaves a
contiguous recent block (the data the model values most), never a random hole.
Resumability falls out of two things:
  - idempotent upserts on PK (re-running a season never duplicates), so the
    rows already present ARE the de-facto progress ledger; and
  - ``stg_nba_pull_progress``: one row per fully-completed (pull_type, season),
    stamped only after the whole season succeeds. A partial season writes no
    marker and is re-pulled on resume.
Default behaviour is to attempt the full floor in a single unattended run; the
resume design is what makes that overnight grind safe. The end-of-run summary
prints the deepest season reached and any residual per-season failures so the
depth marker can be checked in the morning rather than babysat.

COMPLETION POLICY (pinned)
--------------------------
A season is marked complete even with ``n_failed > 0``; the count is recorded
and surfaced. Recent seasons are never full-repulled to recover a few items.
Resume is TWO passes: pass 1 pulls not-yet-complete seasons; pass 2 runs the
targeted retry for completed seasons carrying ``n_failed > 0`` BEFORE declaring
the backfill done. (See the note in ``run_seasoned_backfill``: the heavy pulls
here use SEASON-WIDE endpoints, so their failures are parse-level rather than
per-item-retryable; the two-pass machinery is implemented faithfully and
degrades to "surface, do not loop" for those, which is the honest behaviour
given the endpoint shape.)

Entry points:
  uv run python -m scripts.data_pull.nba.nba_api_pull all        # everything, full 1996 grind
  uv run python -m scripts.data_pull.nba.nba_api_pull reference   # players + teams only (cheap)
  uv run python -m scripts.data_pull.nba.nba_api_pull game_logs   # the heavy backwards-fill
  uv run python -m scripts.data_pull.nba.nba_api_pull advanced
  uv run python -m scripts.data_pull.nba.nba_api_pull team_records
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import UTC, date, datetime
from pathlib import Path

from tenacity import retry, stop_after_attempt, wait_exponential

try:  # canonical shared helper owned by this chat
    from scripts.common.db import connect, upsert, utcnow_iso
except ImportError:  # pragma: no cover - import path depends on run context
    from db import connect, upsert, utcnow_iso  # type: ignore

log = logging.getLogger("nba_api_pull")

GAME_LOG_FLOOR = 1996  # pinned single floor for all seasoned pulls; reversible config
CACHE_DIR = Path("data/cache/nba_api")
DEFAULT_DB_PATH = Path("data/awards.db")
SEASON_TYPE = "Regular Season"  # awards are regular-season; never pull playoffs here

# ----------------------------------------------------------------------------
# Conference map (E/W). nba_api's static team list carries no conference, and a
# team's conference is effectively stable over the 1996+ window, so a static map
# keyed by the stable abbreviation is correct and avoids an extra network call.
# (Charlotte/New Orleans relocations did not cross conferences in this window.)
# ----------------------------------------------------------------------------
_CONFERENCE_BY_ABBR: dict[str, str] = {
    "ATL": "E", "BOS": "E", "BKN": "E", "CHA": "E", "CHI": "E", "CLE": "E",
    "DET": "E", "IND": "E", "MIA": "E", "MIL": "E", "NYK": "E", "ORL": "E",
    "PHI": "E", "TOR": "E", "WAS": "E",
    "DAL": "W", "DEN": "W", "GSW": "W", "HOU": "W", "LAC": "W", "LAL": "W",
    "MEM": "W", "MIN": "W", "NOP": "W", "OKC": "W", "PHX": "W", "POR": "W",
    "SAC": "W", "SAS": "W", "UTA": "W",
}


# ----------------------------------------------------------------------------
# Season helpers (the only conversion sites)
# ----------------------------------------------------------------------------

def season_to_nba_str(season: int) -> str:
    """Starting-year season int -> nba_api season string. 1996 -> '1996-97'."""
    return f"{season}-{str(season + 1)[-2:]}"


def nba_str_to_season(s: str) -> int:
    """nba_api season string -> starting-year int. '1996-97' -> 1996."""
    return int(s[:4])


def current_season_start_year(today: date | None = None) -> int:
    """The starting year of the most recent NBA season.

    NBA seasons tip off in October. Before October we are still inside the
    season that started the previous calendar year. June 2026 -> 2025.
    """
    today = today or datetime.now(UTC).date()
    return today.year if today.month >= 10 else today.year - 1


# ----------------------------------------------------------------------------
# Cache + retry-wrapped network layer (thin; all nba_api contact is here)
# ----------------------------------------------------------------------------

def _cache_path(key: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", key)
    return CACHE_DIR / f"{safe}.json"


def _cached_records(key: str, fetch, use_cache: bool) -> list[dict]:
    """Return a list-of-dict records for an endpoint call, disk-cached.

    ``fetch`` is a zero-arg callable returning the records (so the retry/backoff
    wraps only the live call, and a cache hit skips the network entirely).
    Caching is mandatory: nba_api rate-limits and times out hard, and the 1996+
    grind must never re-hit a page already pulled during dev iteration.
    """
    cp = _cache_path(key)
    if use_cache and cp.exists():
        return json.loads(cp.read_text())
    records = fetch()
    cp.write_text(json.dumps(records, default=str))
    return records


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, min=2, max=30), reraise=True)
def _endpoint_records(endpoint_cls, **params) -> list[dict]:
    """Instantiate an nba_api endpoint and return its first result set as records.

    Retries with exponential backoff (nba_api times out frequently). Importing
    nba_api lazily keeps the module importable (and unit-testable) in an
    environment where nba_api is absent or cannot reach stats.nba.com.
    """
    ep = endpoint_cls(**params)
    df = ep.get_data_frames()[0]
    # to_dict('records') gives a list of {column: value}; NaN -> we coerce later.
    return df.to_dict("records")


# Each wrapper names the endpoint + params explicitly so the call sites read
# clearly and the cache keys are stable across runs.

def fetch_static_players() -> list[dict]:
    """Local static player universe (id, full_name, is_active). No network."""
    from nba_api.stats.static import players
    return players.get_players()


def fetch_static_teams() -> list[dict]:
    """Local static team list (id, abbreviation, full_name). No network."""
    from nba_api.stats.static import teams
    return teams.get_teams()


def fetch_draft_history(use_cache: bool = True) -> list[dict]:
    """All draft picks ever, one call: PERSON_ID, OVERALL_PICK, SEASON, ..."""
    from nba_api.stats.endpoints import drafthistory
    return _cached_records("draft_history", lambda: _endpoint_records(drafthistory.DraftHistory), use_cache)


def fetch_game_logs(season: int, use_cache: bool = True) -> list[dict]:
    """All players' regular-season game logs for one season (one call, player mode)."""
    from nba_api.stats.endpoints import leaguegamelog
    return _cached_records(
        f"gamelog_{season}",
        lambda: _endpoint_records(
            leaguegamelog.LeagueGameLog,
            season=season_to_nba_str(season),
            season_type_all_star=SEASON_TYPE,
            player_or_team_abbreviation="P",
        ),
        use_cache,
    )


def fetch_advanced_player_stats(season: int, use_cache: bool = True) -> list[dict]:
    """Season-aggregate Advanced player stats for one season (one call)."""
    from nba_api.stats.endpoints import leaguedashplayerstats
    return _cached_records(
        f"advanced_{season}",
        lambda: _endpoint_records(
            leaguedashplayerstats.LeagueDashPlayerStats,
            season=season_to_nba_str(season),
            season_type_all_star=SEASON_TYPE,
            measure_type_detailed_defense="Advanced",
        ),
        use_cache,
    )


def fetch_standings(season: int, use_cache: bool = True) -> list[dict]:
    """Final standings for one season (wins/losses/win_pct/conf rank)."""
    from nba_api.stats.endpoints import leaguestandingsv3
    return _cached_records(
        f"standings_{season}",
        lambda: _endpoint_records(
            leaguestandingsv3.LeagueStandingsV3,
            season=season_to_nba_str(season),
            season_type=SEASON_TYPE,
        ),
        use_cache,
    )


def fetch_advanced_team_stats(season: int, use_cache: bool = True) -> list[dict]:
    """Season-aggregate Advanced team stats (net/off/def rating) for one season."""
    from nba_api.stats.endpoints import leaguedashteamstats
    return _cached_records(
        f"team_advanced_{season}",
        lambda: _endpoint_records(
            leaguedashteamstats.LeagueDashTeamStats,
            season=season_to_nba_str(season),
            season_type_all_star=SEASON_TYPE,
            measure_type_detailed_defense="Advanced",
        ),
        use_cache,
    )


# ----------------------------------------------------------------------------
# Pure value coercion + row builders (no network; unit-tested directly)
# ----------------------------------------------------------------------------

def _f(v) -> float | None:
    """Coerce to float, mapping NaN / blank / None to None (NULL)."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f  # NaN != NaN


def _i(v) -> int | None:
    f = _f(v)
    return None if f is None else int(f)


def _norm_game_date(raw) -> str | None:
    """Normalise a LeagueGameLog GAME_DATE to ISO YYYY-MM-DD.

    The endpoint usually returns ISO already; some historical responses use
    'MON DD, YYYY'. Try ISO first, then the abbreviated-month form.
    """
    if raw is None:
        return None
    s = str(raw).strip()
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d", "%b %d, %Y"):
        try:
            return datetime.strptime(s[: len(fmt) + 6] if "T" in fmt else s, fmt).date().isoformat()
        except ValueError:
            continue
    # Last resort: leading YYYY-MM-DD slice if present.
    m = re.match(r"(\d{4}-\d{2}-\d{2})", s)
    return m.group(1) if m else None


def _ts_pct(pts, fga, fta) -> float | None:
    """True Shooting % = PTS / (2*(FGA + 0.44*FTA)). Per-game, exact, cheap.

    Computed here because the basic box-score endpoint does not return TS%, but
    the inputs are present, so we fill it rather than leaving it NULL. USG% needs
    team totals not present per-game, so usage_rate stays NULL (a season-
    aggregate, sourced via the advanced pull, not fabricated per-game).
    """
    pts_f, fga_f, fta_f = _f(pts), _f(fga), _f(fta)
    if pts_f is None or fga_f is None or fta_f is None:
        return None
    denom = 2.0 * (fga_f + 0.44 * fta_f)
    return None if denom <= 0 else pts_f / denom


def _opp_team_id_from_matchup(matchup: str | None, abbr_to_id: dict[str, int]) -> int | None:
    """Derive opponent team_id from a MATCHUP string like 'OKC vs. DAL' / 'OKC @ DAL'.

    The opponent abbreviation is the trailing token; map it via the static abbr
    index. Unknown/relocated abbreviations resolve to None rather than guessing.
    """
    if not matchup:
        return None
    parts = re.split(r"\s+(?:vs\.?|@)\s+", str(matchup).strip())
    if len(parts) != 2:
        return None
    return abbr_to_id.get(parts[1].strip().upper())


def build_player_rows(static_players: list[dict], draft_records: list[dict], pulled_at: str) -> list[dict]:
    """Build stg_nba_players rows from the static universe + draft history.

    Cheap fields only: nba_api_id, name, is_active, draft_year, draft_position,
    lottery_pick. dob and position are left NULL here (see module/flag notes):
    sourcing them needs a per-player commonplayerinfo sweep of the whole
    historical universe, which is a multi-hour grind that is largely redundant
    with bref's bio scrape for the players who actually matter (voting
    candidates). NULL here means "not yet enriched", not "tried and failed".
    """
    # Index draft info by player id. lottery_pick = overall pick in 1..14.
    draft_by_id: dict[int, dict] = {}
    for d in draft_records:
        pid = _i(d.get("PERSON_ID"))
        if pid is None:
            continue
        overall = _i(d.get("OVERALL_PICK"))
        draft_by_id[pid] = {
            "draft_year": _i(d.get("SEASON")),
            "draft_position": overall,
            # Lottery is the top 14 picks (lottery era covers the whole 1996+ window).
            "lottery_pick": (1 if (overall is not None and overall <= 14) else 0) if overall is not None else None,
        }

    rows: list[dict] = []
    for p in static_players:
        pid = _i(p.get("id"))
        if pid is None:
            continue
        d = draft_by_id.get(pid, {})
        rows.append({
            "nba_api_id": pid,
            "name": p.get("full_name"),
            "position": None,            # deferred enrichment (see docstring)
            "dob": None,                 # deferred enrichment
            "draft_year": d.get("draft_year"),
            "draft_position": d.get("draft_position"),
            "lottery_pick": d.get("lottery_pick"),
            "is_active": 1 if p.get("is_active") else 0,
            "pulled_at": pulled_at,
        })
    return rows


def build_team_rows(static_teams: list[dict], pulled_at: str) -> list[dict]:
    """Build canonical teams rows (team_id, abbr, name, conference)."""
    rows: list[dict] = []
    for t in static_teams:
        abbr = t.get("abbreviation")
        rows.append({
            "team_id": _i(t.get("id")),
            "abbr": abbr,
            "name": t.get("full_name"),
            "conference": _CONFERENCE_BY_ABBR.get(abbr or ""),
        })
    return rows


def build_game_log_rows(records: list[dict], season: int, abbr_to_id: dict[str, int],
                        pulled_at: str) -> tuple[list[dict], int]:
    """Build stg_nba_player_game_logs rows for one season's player-mode logs.

    Returns (rows, n_dropped). A row is dropped (counted, not fatal) when it
    lacks the natural-key fields (nba_api_id, game_id), since those cannot be
    invented. n_dropped feeds the season's n_failed.
    """
    rows: list[dict] = []
    dropped = 0
    for r in records:
        pid = _i(r.get("PLAYER_ID"))
        gid = r.get("GAME_ID")
        if pid is None or not gid:
            dropped += 1
            continue
        rows.append({
            "nba_api_id": pid,
            "game_date": _norm_game_date(r.get("GAME_DATE")),
            "game_id": str(gid),
            "season": season,
            "team_id": _i(r.get("TEAM_ID")),
            "opp_team_id": _opp_team_id_from_matchup(r.get("MATCHUP"), abbr_to_id),
            "minutes": _f(r.get("MIN")),
            "points": _f(r.get("PTS")),
            "rebounds": _f(r.get("REB")),
            "assists": _f(r.get("AST")),
            "steals": _f(r.get("STL")),
            "blocks": _f(r.get("BLK")),
            "turnovers": _f(r.get("TOV")),
            "fga": _f(r.get("FGA")),
            "fgm": _f(r.get("FGM")),
            "fg3a": _f(r.get("FG3A")),
            "fg3m": _f(r.get("FG3M")),
            "fta": _f(r.get("FTA")),
            "ftm": _f(r.get("FTM")),
            "usage_rate": None,          # needs team totals; season-agg only, not per-game
            "ts_pct": _ts_pct(r.get("PTS"), r.get("FGA"), r.get("FTA")),
            "pulled_at": pulled_at,
        })
    return rows, dropped


def build_advanced_rows(records: list[dict], season: int, snapshot_date: str,
                        pulled_at: str) -> tuple[list[dict], int]:
    """Build stg_nba_player_advanced_stats rows from the Advanced player pull.

    CORRECTED COLUMN SET (see module flag): nba_api sources NONE of bpm / vorp /
    ws_per_48 (those are Basketball-Reference metrics, Chat B's staging). What
    nba_api genuinely provides at season level is off/def/net rating, USG%, TS%,
    pace and PIE. ``def_rating`` is the canonical ``drtg`` source. Every column
    written here is really sourced; no dead NULL columns.
    """
    rows: list[dict] = []
    dropped = 0
    for r in records:
        pid = _i(r.get("PLAYER_ID"))
        if pid is None:
            dropped += 1
            continue
        rows.append({
            "nba_api_id": pid,
            "season": season,
            "snapshot_date": snapshot_date,
            "gp": _i(r.get("GP")),
            "min": _f(r.get("MIN")),
            "off_rating": _f(r.get("OFF_RATING")),
            "def_rating": _f(r.get("DEF_RATING")),   # canonical drtg
            "net_rating": _f(r.get("NET_RATING")),
            "usg_pct": _f(r.get("USG_PCT")),
            "ts_pct": _f(r.get("TS_PCT")),
            "pace": _f(r.get("PACE")),
            "pie": _f(r.get("PIE")),
            "pulled_at": pulled_at,
        })
    return rows, dropped


def build_team_record_rows(standings: list[dict], team_adv: list[dict], season: int,
                           snapshot_date: str) -> tuple[list[dict], int]:
    """Build canonical team_records rows by joining standings + advanced team stats.

    standings -> wins/losses/win_pct/conf_rank; advanced -> net/off/def rating.
    For a season-end snapshot, projected_seed == conf_rank (the season resolved
    to that seed) and sos_remaining is NULL (no schedule remains). The in-season
    daily job will pass a real snapshot_date and populate projected_seed /
    sos_remaining separately; the row shape is identical so both coexist.
    """
    net_by_team: dict[int, dict] = {}
    for t in team_adv:
        tid = _i(t.get("TEAM_ID"))
        if tid is None:
            continue
        net_by_team[tid] = {
            "net_rating": _f(t.get("NET_RATING")),
            "off_rating": _f(t.get("OFF_RATING")),
            "def_rating": _f(t.get("DEF_RATING")),
        }

    rows: list[dict] = []
    dropped = 0
    for s in standings:
        tid = _i(s.get("TeamID"))
        if tid is None:
            dropped += 1
            continue
        adv = net_by_team.get(tid, {})
        conf_rank = _i(s.get("PlayoffRank"))
        rows.append({
            "team_id": tid,
            "snapshot_date": snapshot_date,
            "season": season,
            "wins": _i(s.get("WINS")),
            "losses": _i(s.get("LOSSES")),
            "win_pct": _f(s.get("WinPCT")),
            "net_rating": adv.get("net_rating"),
            "off_rating": adv.get("off_rating"),
            "def_rating": adv.get("def_rating"),
            "conf_rank": conf_rank,
            "projected_seed": conf_rank,   # season-end: resolved seed == conf rank
            "sos_remaining": None,         # no schedule remains at season end
        })
    return rows, dropped


# ----------------------------------------------------------------------------
# Progress ledger (resume) + season selection
# ----------------------------------------------------------------------------

def completed_seasons(conn, pull_type: str) -> set[int]:
    """Seasons already marked fully complete for a pull_type (the resume skip set)."""
    cur = conn.execute(
        "SELECT season FROM stg_nba_pull_progress WHERE pull_type = ?", (pull_type,)
    )
    return {row["season"] for row in cur}


def seasons_with_residual_failures(conn, pull_type: str) -> list[int]:
    """Completed seasons carrying n_failed > 0 (targets for the pass-2 retry)."""
    cur = conn.execute(
        "SELECT season FROM stg_nba_pull_progress WHERE pull_type = ? AND n_failed > 0 "
        "ORDER BY season DESC",
        (pull_type,),
    )
    return [row["season"] for row in cur]


def mark_season_complete(conn, pull_type: str, season: int, n_players: int, n_failed: int) -> None:
    """Stamp a season complete. Written ONLY after the whole season succeeds."""
    upsert(
        conn,
        "stg_nba_pull_progress",
        [{
            "pull_type": pull_type,
            "season": season,
            "completed_at": utcnow_iso(),
            "n_players": n_players,
            "n_failed": n_failed,
        }],
        ["pull_type", "season"],
    )


def seasons_to_pull(conn, pull_type: str, floor: int, current: int) -> list[int]:
    """Not-yet-complete seasons, NEWEST-FIRST, floor..current inclusive.

    Newest-first is the load-bearing ordering: a rate-limit stop leaves a
    contiguous recent block, never a hole. Already-complete seasons are skipped.
    """
    done = completed_seasons(conn, pull_type)
    return [s for s in range(current, floor - 1, -1) if s not in done]


# ----------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------

def pull_reference(conn, use_cache: bool = True) -> dict:
    """One-shot (un-seasoned) pulls: stg_nba_players + canonical teams.

    Cheap. Players = static universe joined to draft history. Teams = static 30
    with the conference map.
    """
    pulled_at = utcnow_iso()
    teams_rows = build_team_rows(fetch_static_teams(), pulled_at)
    upsert(conn, "teams", teams_rows, ["team_id"])

    player_rows = build_player_rows(fetch_static_players(), fetch_draft_history(use_cache), pulled_at)
    upsert(conn, "stg_nba_players", player_rows, ["nba_api_id"])

    return {"teams_written": len(teams_rows), "players_written": len(player_rows)}


def _season_end_snapshot(season: int) -> str:
    """Deterministic season-end snapshot_date stamp for a historical season.

    The season starting in year Y ends the following June; we use 30 June Y+1 as
    a stable season-end marker. This is the as-of date for the season-aggregate
    rows, distinct from in-season daily snapshots.
    """
    return date(season + 1, 6, 30).isoformat()


def run_seasoned_backfill(conn, pull_type: str, floor: int, current: int,
                          use_cache: bool = True) -> dict:
    """Two-pass backwards-fill for one seasoned pull_type.

    Pass 1: pull every not-yet-complete season, newest-first. Each season is
            fetched (retry-wrapped), built, upserted, then marked complete with
            (n_players, n_failed). A season that raises after all retries is left
            UNMARKED and reported, so resume re-attempts it.
    Pass 2: for completed seasons carrying n_failed > 0, run the targeted retry
            before declaring done.

    NOTE ON THE TWO-PASS GUARD FOR SEASON-WIDE ENDPOINTS
    ----------------------------------------------------
    The heavy pulls here (game_logs, advanced, team_records) use SEASON-WIDE
    endpoints: one call returns the whole season, so there is no per-ITEM fetch
    to retry. Their n_failed is parse-level (rows missing a natural key that were
    simply not in the payload); re-pulling the season cannot recover them, and we
    never full-repull a recent season anyway. So pass 2 re-runs the season build
    from cache once (cheap, no network) to confirm the residual is stable, then
    SURFACES it rather than looping. The two-pass structure is honoured; it
    degrades correctly to "report, do not loop" given the endpoint shape. A
    future per-player enrichment pull (e.g. commonplayerinfo) would slot into the
    same machinery and genuinely retry per item.
    """
    builder = _SEASONED[pull_type]
    summary = {
        "pull_type": pull_type,
        "floor": floor,
        "current": current,
        "seasons_pulled": [],
        "rows_written": 0,
        "deepest_season_reached": None,
        "season_failures": [],         # seasons that errored out entirely (unmarked)
        "residual_failures": {},       # season -> n_failed for completed-but-imperfect
    }

    # --- Pass 1: pull not-yet-complete seasons, newest-first ---
    todo = seasons_to_pull(conn, pull_type, floor, current)
    for season in todo:
        try:
            rows, n_failed = builder(conn, season, use_cache)
        except Exception as exc:  # noqa: BLE001 - log-and-continue; season stays unmarked
            log.error("%s season %d failed after retries: %s", pull_type, season, exc)
            summary["season_failures"].append((season, str(exc)))
            # Stop descending no further than necessary? No: continue so a single
            # bad season does not abort the whole grind. The hole is reported and
            # re-attempted next run (season unmarked).
            continue
        summary["seasons_pulled"].append(season)
        summary["rows_written"] += len(rows)
        if n_failed:
            summary["residual_failures"][season] = n_failed
        # Deepest season reached = smallest season we have successfully marked.
        d = summary["deepest_season_reached"]
        summary["deepest_season_reached"] = season if d is None else min(d, season)

    # --- Pass 2: targeted retry for completed seasons with residual failures ---
    for season in seasons_with_residual_failures(conn, pull_type):
        try:
            _rows, n_failed = builder(conn, season, use_cache=True)  # cache-only re-build
        except Exception as exc:  # noqa: BLE001
            log.warning("%s season %d pass-2 re-build errored: %s", pull_type, season, exc)
            continue
        if n_failed:
            summary["residual_failures"][season] = n_failed  # stable residual, surfaced

    return summary


# Each seasoned builder: fetch one season, build rows, upsert, mark complete,
# return (rows, n_failed). Marking happens here so a season is complete only
# after its rows are durably written.

def _pull_game_logs_season(conn, season: int, use_cache: bool) -> tuple[list[dict], int]:
    pulled_at = utcnow_iso()
    abbr_to_id = {t.get("abbreviation"): _i(t.get("id")) for t in fetch_static_teams()}
    records = fetch_game_logs(season, use_cache)
    rows, dropped = build_game_log_rows(records, season, abbr_to_id, pulled_at)
    upsert(conn, "stg_nba_player_game_logs", rows, ["nba_api_id", "game_id"])
    n_players = len({r["nba_api_id"] for r in rows})
    mark_season_complete(conn, "game_logs", season, n_players, dropped)
    log.info("game_logs %d: %d rows, %d players, %d dropped", season, len(rows), n_players, dropped)
    return rows, dropped


def _pull_advanced_season(conn, season: int, use_cache: bool) -> tuple[list[dict], int]:
    pulled_at = utcnow_iso()
    snapshot = _season_end_snapshot(season)
    records = fetch_advanced_player_stats(season, use_cache)
    rows, dropped = build_advanced_rows(records, season, snapshot, pulled_at)
    upsert(conn, "stg_nba_player_advanced_stats", rows, ["nba_api_id", "season", "snapshot_date"])
    mark_season_complete(conn, "advanced_stats", season, len(rows), dropped)
    log.info("advanced %d: %d rows, %d dropped", season, len(rows), dropped)
    return rows, dropped


def _pull_team_records_season(conn, season: int, use_cache: bool) -> tuple[list[dict], int]:
    snapshot = _season_end_snapshot(season)
    standings = fetch_standings(season, use_cache)
    team_adv = fetch_advanced_team_stats(season, use_cache)
    rows, dropped = build_team_record_rows(standings, team_adv, season, snapshot)
    upsert(conn, "team_records", rows, ["team_id", "snapshot_date"])
    mark_season_complete(conn, "team_records", season, len(rows), dropped)
    log.info("team_records %d: %d rows, %d dropped", season, len(rows), dropped)
    return rows, dropped


_SEASONED = {
    "game_logs": _pull_game_logs_season,
    "advanced_stats": _pull_advanced_season,
    "team_records": _pull_team_records_season,
}


def run(target: str, db_path: Path = DEFAULT_DB_PATH, floor: int = GAME_LOG_FLOOR,
        current: int | None = None, use_cache: bool = True) -> dict:
    """Top-level dispatch. target in {all, reference, game_logs, advanced, team_records}."""
    current = current if current is not None else current_season_start_year()
    conn = connect(db_path)
    try:
        out: dict = {"target": target, "floor": floor, "current": current}
        if target in ("all", "reference"):
            out["reference"] = pull_reference(conn, use_cache)
        if target in ("all", "team_records"):
            out["team_records"] = run_seasoned_backfill(conn, "team_records", floor, current, use_cache)
        if target in ("all", "advanced"):
            out["advanced_stats"] = run_seasoned_backfill(conn, "advanced_stats", floor, current, use_cache)
        if target in ("all", "game_logs"):
            # Heaviest pull last so the cheaper data is durable before the grind.
            out["game_logs"] = run_seasoned_backfill(conn, "game_logs", floor, current, use_cache)
        return out
    finally:
        conn.close()


def _print_summary(out: dict) -> None:
    """Prominent end-of-run summary: deepest season reached + residual failures."""
    print("\n" + "=" * 70)
    print(f"nba_api_pull summary  target={out['target']}  floor={out['floor']}  current={out['current']}")
    print("=" * 70)
    if "reference" in out:
        r = out["reference"]
        print(f"reference: {r['players_written']} players, {r['teams_written']} teams")
    for key in ("team_records", "advanced_stats", "game_logs"):
        if key not in out:
            continue
        s = out[key]
        deepest = s["deepest_season_reached"]
        print(f"\n[{key}]  rows={s['rows_written']}  seasons_pulled={len(s['seasons_pulled'])}")
        print(f"  DEEPEST SEASON REACHED: {deepest if deepest is not None else 'none (nothing new pulled)'}")
        if s["season_failures"]:
            print(f"  SEASONS THAT FAILED ENTIRELY (re-attempted next run): "
                  f"{[yr for yr, _ in s['season_failures']]}")
        if s["residual_failures"]:
            print("  RESIDUAL PER-SEASON FAILURES (dropped rows within completed seasons):")
            for yr in sorted(s["residual_failures"], reverse=True):
                print(f"    season {yr}: {s['residual_failures'][yr]} dropped")
        else:
            print("  residual failures: none")
    print("=" * 70)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="nba_api puller -> nba_api-scoped staging + teams/team_records")
    p.add_argument("target", nargs="?", default="all",
                   choices=["all", "reference", "game_logs", "advanced", "team_records"])
    p.add_argument("--db", default=str(DEFAULT_DB_PATH))
    p.add_argument("--floor", type=int, default=GAME_LOG_FLOOR,
                   help="Starting-year floor for seasoned pulls (default 1996; reversible config).")
    p.add_argument("--current", type=int, default=None,
                   help="Override current season starting year (default: derived from today).")
    p.add_argument("--no-cache", action="store_true", help="Bypass the on-disk response cache.")
    args = p.parse_args(argv)

    out = run(args.target, Path(args.db), args.floor, args.current, use_cache=not args.no_cache)
    _print_summary(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
