"""As-of eligibility reweight, v2: injury-conditioned avail mixture.

A serve-time-only pricing transform (never a model feature). Supersedes the v1
recency-EMA proxy: the availability question ("will he keep playing?") is now
answered by the injury report's status and reason via the empirical miss-count
distribution, and the per-game qualifying rate p answers only the minutes
question, conditioned on being active. The binomial survives everywhere the
regime is uncertain; season-ending is its degenerate tail.

Per candidate at snapshot t, era-gated to season >= 2023 (P_elig = 1 before the
65-game rule):
  qual_gp  qualifying games so far (>= 20 min, plus up to two 15-20 min games).
  needed   max(0, 65 - qual_gp).
  avail    remaining current-team games (82 - team games played as-of).
  p        purified per-game qualifying rate = fraction of APPEARED games that
           cleared 20 minutes (conditional on being active; near 1 for real
           candidates). NOT a team-games rate: absences are avail's job now.
  state    latest injuries row on or before t, within STALE_DAYS. If the player
           is currently injured (out / doubtful / questionable), P_elig is the
           miss-count mixture p_elig_mixture(status, category, avail, needed, p);
           otherwise (available / probable / stale / not on report) he has no
           current absence and P_elig = binom.sf(needed - 1, avail, p).

The wall (needed > avail -> 0) and clinch (needed == 0 -> 1) fall out of the
binomial and the mixture. Reads only rows dated <= t; realised games are never
read at serve time. Binds on samples.vote_share_pred (and sizing_weights) at the
top of run_award, ahead of the rank floor and the jspeak_floor lift.
"""

from __future__ import annotations

import datetime as dt
import logging

import numpy as np
from scipy.stats import binom

try:
    from scripts.common.injury_categories import classify_reason
    from scripts.strategy.pricing.injury_miss_model import p_elig_mixture, load_distribution
except ImportError:
    from injury_categories import classify_reason
    from injury_miss_model import p_elig_mixture, load_distribution

log = logging.getLogger("eligibility_v2")

THRESHOLD = 65
FULL_SEASON_GAMES = 82
RULE_MIN_SEASON = 2023
QUAL_MINUTES = 20.0
NEAR_MISS_MIN_MINUTES = 15.0
NEAR_MISS_MAX_COUNT = 2
P_CLIP = (0.02, 0.995)
STALE_DAYS = 10
INJURED_STATUSES = {"out", "doubtful", "questionable"}


def rule_active(season: int) -> bool:
    return int(season) >= RULE_MIN_SEASON


# -----------------------------------------------------------------------------
# As-of game-log reads (carried from v1; p purified to played games)
# -----------------------------------------------------------------------------

def _resolve_nba_ids(conn, player_ids) -> dict:
    out = {}
    for pid in player_ids:
        row = conn.execute(
            "SELECT nba_api_id FROM players WHERE player_id = ?", (int(pid),)
        ).fetchone()
        out[int(pid)] = None if row is None or row[0] is None else int(row[0])
    return out


def _current_team_asof(conn, nba_api_id, season, snapshot_date):
    row = conn.execute(
        "SELECT team_id FROM stg_nba_player_game_logs "
        "WHERE season = ? AND nba_api_id = ? AND team_id IS NOT NULL "
        "  AND game_date IS NOT NULL AND minutes IS NOT NULL AND game_date <= ? "
        "ORDER BY game_date DESC, game_id DESC LIMIT 1",
        (int(season), int(nba_api_id), str(snapshot_date)),
    ).fetchone()
    return None if row is None else int(row[0])


def _team_games_played_asof(conn, team_id, season, snapshot_date) -> int:
    row = conn.execute(
        "SELECT COUNT(DISTINCT game_id) AS n FROM stg_nba_player_game_logs "
        "WHERE season = ? AND team_id = ? AND game_date IS NOT NULL AND game_date <= ?",
        (int(season), int(team_id), str(snapshot_date)),
    ).fetchone()
    return int(row["n"] if hasattr(row, "keys") else row[0])


def _count_band(conn, nba_api_id, season, snapshot_date, lo, hi=None) -> int:
    if hi is None:
        sql = ("SELECT COUNT(DISTINCT game_id) AS n FROM stg_nba_player_game_logs "
               "WHERE season=? AND nba_api_id=? AND game_date IS NOT NULL "
               "  AND game_date<=? AND minutes>=?")
        params = (int(season), int(nba_api_id), str(snapshot_date), float(lo))
    else:
        sql = ("SELECT COUNT(DISTINCT game_id) AS n FROM stg_nba_player_game_logs "
               "WHERE season=? AND nba_api_id=? AND game_date IS NOT NULL "
               "  AND game_date<=? AND minutes>=? AND minutes<?")
        params = (int(season), int(nba_api_id), str(snapshot_date), float(lo), float(hi))
    row = conn.execute(sql, params).fetchone()
    return int(row["n"] if hasattr(row, "keys") else row[0])


def _appearances_asof(conn, nba_api_id, season, snapshot_date) -> int:
    row = conn.execute(
        "SELECT COUNT(DISTINCT game_id) AS n FROM stg_nba_player_game_logs "
        "WHERE season=? AND nba_api_id=? AND game_date IS NOT NULL "
        "  AND game_date<=? AND minutes IS NOT NULL",
        (int(season), int(nba_api_id), str(snapshot_date)),
    ).fetchone()
    return int(row["n"] if hasattr(row, "keys") else row[0])


def _qualifying_games(conn, nba_api_id, season, snapshot_date) -> int:
    n_full = _count_band(conn, nba_api_id, season, snapshot_date, QUAL_MINUTES)
    n_near = _count_band(conn, nba_api_id, season, snapshot_date,
                         NEAR_MISS_MIN_MINUTES, QUAL_MINUTES)
    return n_full + min(NEAR_MISS_MAX_COUNT, n_near)


def _purified_p(conn, nba_api_id, season, snapshot_date) -> float:
    """Per-game qualifying rate conditional on being active: of games the player
    appeared in, the fraction that cleared 20 minutes. Absences do not enter p
    (that is avail's job). Neutral high prior when he has not appeared."""
    appeared = _appearances_asof(conn, nba_api_id, season, snapshot_date)
    if appeared <= 0:
        return P_CLIP[1]
    n_full = _count_band(conn, nba_api_id, season, snapshot_date, QUAL_MINUTES)
    return float(np.clip(n_full / appeared, P_CLIP[0], P_CLIP[1]))


# -----------------------------------------------------------------------------
# As-of injury state (from the canonical injuries table, keyed by player_id)
# -----------------------------------------------------------------------------

def _injury_state(conn, player_id, snapshot_date):
    """Latest (status, description) on or before the snapshot, or None if there
    is no report row within STALE_DAYS (treated as recovered / healthy)."""
    row = conn.execute(
        "SELECT snapshot_date, status, description FROM injuries "
        "WHERE player_id = ? AND snapshot_date <= ? "
        "ORDER BY snapshot_date DESC LIMIT 1",
        (int(player_id), str(snapshot_date)),
    ).fetchone()
    if row is None:
        return None
    rdate = row["snapshot_date"] if hasattr(row, "keys") else row[0]
    status = (row["status"] if hasattr(row, "keys") else row[1]) or ""
    desc = row["description"] if hasattr(row, "keys") else row[2]
    try:
        gap = (dt.date.fromisoformat(str(snapshot_date)[:10])
               - dt.date.fromisoformat(str(rdate)[:10])).days
    except ValueError:
        gap = 0
    if gap > STALE_DAYS:
        return None
    return status.strip().lower(), desc


# -----------------------------------------------------------------------------
# The factor
# -----------------------------------------------------------------------------

def eligibility_factor_one(conn, dist, award, season, snapshot_date,
                           player_id, nba_api_id):
    """P_elig for one candidate at a snapshot. Era-gated; injury-conditioned."""
    if not rule_active(season):
        return 1.0
    if nba_api_id is None:
        return 1.0
    team_id = _current_team_asof(conn, nba_api_id, season, snapshot_date)
    if team_id is None:
        return 1.0
    qual_gp = _qualifying_games(conn, nba_api_id, season, snapshot_date)
    needed = max(0, THRESHOLD - qual_gp)
    if needed == 0:
        return 1.0
    avail = max(0, FULL_SEASON_GAMES
                - _team_games_played_asof(conn, team_id, season, snapshot_date))
    if avail <= 0:
        return 0.0
    p = _purified_p(conn, nba_api_id, season, snapshot_date)

    state = _injury_state(conn, player_id, snapshot_date)
    if state is not None and state[0] in INJURED_STATUSES:
        status, desc = state
        category = classify_reason(desc)
        return p_elig_mixture(dist, status, category, avail, needed, p)
    return float(binom.sf(needed - 1, avail, p))


def eligibility_factors(conn, dist, award, season, snapshot_date, player_ids):
    """P_elig per candidate. player_ids are model player_ids. Era-gated to 1.0
    for season < 2023."""
    if not rule_active(season):
        return {int(pid): 1.0 for pid in player_ids}
    nba_ids = _resolve_nba_ids(conn, player_ids)
    out = {}
    for pid in player_ids:
        nid = nba_ids[int(pid)]
        if nid is None:
            log.warning("eligibility_v2: player_id %s has no nba_api_id; "
                        "defaulting P_elig=1.0", pid)
            out[int(pid)] = 1.0
            continue
        out[int(pid)] = eligibility_factor_one(
            conn, dist, award, season, snapshot_date, int(pid), nid)
    return out


# -----------------------------------------------------------------------------
# Vector reweight (binds on vote_share_pred and sizing_weights, before jsfloor)
# -----------------------------------------------------------------------------

def apply_eligibility(v, p_elig):
    v = np.asarray(v, dtype=float)
    pe = np.asarray(p_elig, dtype=float)
    if v.shape != pe.shape:
        raise ValueError(f"shape mismatch: v {v.shape} vs p_elig {pe.shape}")
    w = v * pe
    s = float(w.sum())
    if s <= 0.0:
        log.warning("eligibility_v2: whole group reweighted to zero mass; "
                    "returning the un-gated vector renormalised")
        vs = float(v.sum())
        return v / vs if vs > 0 else v
    return w / s


def reweight_vector(v, player_ids, p_elig_dict):
    pe = np.array([float(p_elig_dict.get(int(pid), 1.0)) for pid in player_ids],
                  dtype=float)
    return apply_eligibility(v, pe)
