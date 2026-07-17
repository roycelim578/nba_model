"""Stat-leader arm: as-of rate-count assembler (Beta/Dirichlet substrate).

Builds per (season, snapshot_date, nba_api_id) the cumulative counts that are the
sufficient statistics for every rate node. A conjugate posterior at MC time is
then just prior_pseudocounts + these banked counts (one addition), which is the
whole point of storing counts rather than ratios here.

COUNTS STORED
-------------
Scoring (PTS branch):
  used_fga, used_ft_trip (= 0.44*FTA, the possession-ending FT trips), used_tov
      -> the allocation Dirichlet (FGA : FT-trip : TOV) over used possessions.
  fg3a, fg3m                 -> FG3% Beta.
  fg2a, fg2m                 -> FG2% Beta (unsplit; the blended fallback).
  fg2a_rim, fg2m_rim, fg2a_mid, fg2m_mid
      -> rim/mid split, APPORTIONED from fg2 by the paint vs midrange POINT-share
         proxy (pct_pts_paint : pct_pts_mr), since the schema has no per-attempt
         zone counts. This assumes similar points-per-attempt across the two
         zones within a player, which is only approximate (rim makes score more),
         so the unsplit fg2 counts are ALSO stored and the PTS node picks split
         vs blended on the calibration evidence.
  fta, ftm                   -> FT% Beta; fta also drives the FT-trip allocation leg.

Rebounding (REB branch):
  oreb_chance_proxy, dreb_chance_proxy: cumulative own-team and opponent missed
      shots while the player's team was on court is not available per-player, so
      the REB node uses oreb_pct/dreb_pct (already share*conversion) scaled by
      trunk pace; we store cumulative team-relative rebound counts here only as
      raw reb totals for the direct-rate fallback.
  reb                        -> raw rebounds (direct-rate fallback).

Playmaking (AST branch, 2013+):
  potential_ast              -> creation-volume trajectory (sticky rate node).
  ast                        -> assists; conversion = ast / potential_ast (Beta,
      prior = team-FG-ex-self from stat_team_fg_asof).

Plus games/minutes banked (gp_played_asof, cumulative minutes) so every rate can
be expressed per-game or per-minute as the node needs.

As-of by construction: everything sums game rows with game_date <= snapshot,
mirroring stg_nba_box_asof. Writes stat_rate_counts_asof in the stat_ namespace;
touches nothing the v1 gate reads.

Run:
  uv run python -m scripts.features.stat_leader.rates
  uv run python -m scripts.features.stat_leader.rates --season 2023
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

try:
    from scripts.common.db import connect, upsert, utcnow_iso
except ImportError:  # pragma: no cover
    from db import connect, upsert, utcnow_iso  # type: ignore

log = logging.getLogger("stat_leader.rates")

DEFAULT_DB_PATH = Path("data/awards.db")
FT_POSS_COEF = 0.44


def _envf(name):
    v = os.environ.get(name)
    return float(v) if v not in (None, "") else None


K_USAGE = _envf("VOL_WINSOR_K_USAGE")
K_REB = _envf("VOL_WINSOR_K_REB")
K_POTAST = _envf("VOL_WINSOR_K_POTAST")
K_STL = _envf("VOL_WINSOR_K_STL")
K_BLK = _envf("VOL_WINSOR_K_BLK")
REF_SEED_MIN = float(os.environ.get("VOL_ROBUST_SEED_MIN", "48"))
REF_HL_MIN = float(os.environ.get("VOL_ROBUST_HL_MIN", "500"))
MIN_GAMES = int(os.environ.get("VOL_WINSOR_MIN_GAMES", "5"))


class _RunRef:
    """EWMA of the (possibly capped) per-minute rate, seeded by a league rate
    with REF_SEED_MIN pseudo-minutes so it is not undefined on game one. Half-life
    REF_HL_MIN minutes: a stable anchor, not itself chasing recent form. Updated
    with the value actually used (raw below the cap, the cap itself when a game
    is clipped), so a single blowout cannot inflate its own reference, but a
    sustained genuine step-change still migrates the reference up over a few
    games, at which point the cap stops firing on it. The current game/window is
    never in its own reference."""
    __slots__ = ("num", "den", "hl")

    def __init__(self, mu0, w0, hl):
        self.num = mu0 * w0
        self.den = w0
        self.hl = hl

    def mean(self):
        return self.num / self.den if self.den > 0 else 0.0

    def update(self, mn, r_used):
        dec = 0.5 ** (mn / self.hl) if self.hl and self.hl > 0 else 1.0
        self.num = self.num * dec + mn * r_used
        self.den = self.den * dec + mn


def _winsor_factor(count_g, mn, ref, k, gp, min_games):
    """Scale factor for a count over a span of minutes (one game, or one
    snapshot window) so its implied rate is capped at k times the running
    reference, upper tail only; below the cap, or with k unset, or before
    min_games of the player's own history have accrued, the game passes through
    unchanged. Always updates the reference with the value actually used, so the
    reference matures from game one even while the cap itself is held off."""
    if mn <= 0.0 or count_g <= 0.0:
        return 1.0
    r = count_g / mn
    if k is None or gp < min_games:
        ref.update(mn, r)
        return 1.0
    cap = k * ref.mean()
    if cap <= 0.0 or r <= cap:
        ref.update(mn, r)
        return 1.0
    ref.update(mn, cap)
    return cap / r

COUNT_COLS = [
    "gp_played_asof", "min_asof",
    "used_fga", "used_ft_trip", "used_tov",
    "fg3a", "fg3m", "fg2a", "fg2m",
    "fg2a_rim", "fg2m_rim", "fg2a_mid", "fg2m_mid",
    "fta", "ftm", "reb", "stl", "blk", "potential_ast_asof", "ast",
]


def _ensure_schema(conn) -> None:
    cols = ", ".join(f"{c} REAL" for c in COUNT_COLS)
    conn.execute(
        f"CREATE TABLE IF NOT EXISTS stat_rate_counts_asof ("
        f"nba_api_id INTEGER NOT NULL, season INTEGER NOT NULL, snapshot_date TEXT NOT NULL, "
        f"{cols}, pulled_at TEXT, "
        f"PRIMARY KEY (nba_api_id, season, snapshot_date))"
    )
    conn.commit()


def _seasons(conn, season):
    if season is not None:
        return [season]
    return [r[0] for r in conn.execute(
        "SELECT DISTINCT season FROM stg_nba_player_game_logs ORDER BY season")]


def _grid(conn, season):
    return [r["snapshot_date"] for r in conn.execute(
        "SELECT snapshot_date FROM snapshot_grid WHERE season=? "
        "AND snapshot_kind IN ('weekly','ratings') ORDER BY snapshot_date", (season,))]


def _load_logs(conn, season):
    """Ordered per-player game rows with the fields the counts need."""
    by = defaultdict(list)
    for r in conn.execute(
        "SELECT nba_api_id, game_date, minutes, points, rebounds, assists, "
        "steals, blocks, turnovers, fga, fgm, fg3a, fg3m, fta, ftm "
        "FROM stg_nba_player_game_logs WHERE season=? AND minutes IS NOT NULL "
        "ORDER BY nba_api_id, game_date", (season,)):
        by[r["nba_api_id"]].append(dict(r))
    return by


def _load_pointshare(conn, season):
    """As-of paint/midrange point shares per (player, snapshot) for the rim/mid
    apportionment. Nearest as-of row is the ext table's own snapshot value."""
    ps = defaultdict(dict)
    for r in conn.execute(
        "SELECT nba_api_id, snapshot_date, pct_pts_paint, pct_pts_mr "
        "FROM stg_nba_player_asof_ext WHERE season=?", (season,)):
        ps[r["nba_api_id"]][r["snapshot_date"]] = (r["pct_pts_paint"], r["pct_pts_mr"])
    return ps


def _load_potast(conn, season):
    """As-of cumulative potential assists proxy: the ext potential_ast is a
    per-game average as-of, so multiply by gp to recover a cumulative count."""
    pa = defaultdict(dict)
    for r in conn.execute(
        "SELECT e.nba_api_id, e.snapshot_date, e.potential_ast, b.gp_played_asof "
        "FROM stg_nba_player_asof_ext e "
        "JOIN stg_nba_box_asof b ON b.nba_api_id=e.nba_api_id AND b.season=e.season "
        "  AND b.snapshot_date=e.snapshot_date "
        "WHERE e.season=?", (season,)):
        if r["potential_ast"] is not None:
            pa[r["nba_api_id"]][r["snapshot_date"]] = r["potential_ast"]  # already cumulative
    return pa


def build_season(conn, season, pulled_at):
    grid = _grid(conn, season)
    if not grid:
        return 0
    logs = _load_logs(conn, season)
    pshare = _load_pointshare(conn, season)
    potast = _load_potast(conn, season)

    _tot_min = _tot_used = _tot_reb = _tot_stl = _tot_blk = 0.0
    for _games in logs.values():
        for _g in _games:
            _mn = _g["minutes"] or 0.0
            if _mn <= 0:
                continue
            _tot_min += _mn
            _tot_used += (_g["fga"] or 0.0) + FT_POSS_COEF * (_g["fta"] or 0.0) + (_g["turnovers"] or 0.0)
            _tot_reb += _g["rebounds"] or 0.0
            _tot_stl += _g["steals"] or 0.0
            _tot_blk += _g["blocks"] or 0.0
    mu0_usage = (_tot_used / _tot_min) if _tot_min > 0 else 0.0
    mu0_reb = (_tot_reb / _tot_min) if _tot_min > 0 else 0.0
    mu0_stl = (_tot_stl / _tot_min) if _tot_min > 0 else 0.0
    mu0_blk = (_tot_blk / _tot_min) if _tot_min > 0 else 0.0
    _tot_potast = _tot_potast_min = 0.0
    for _pid, _vals in potast.items():
        if not _vals:
            continue
        _last = _vals[max(_vals)]
        _pmin = sum((_g["minutes"] or 0.0) for _g in logs.get(_pid, []) if (_g["minutes"] or 0) > 0)
        if _pmin > 0:
            _tot_potast += _last; _tot_potast_min += _pmin
    mu0_potast = (_tot_potast / _tot_potast_min) if _tot_potast_min > 0 else 0.0

    out = []
    for pid, games in logs.items():
        games.sort(key=lambda x: x["game_date"])
        gi = 0
        c = {k: 0.0 for k in COUNT_COLS}
        ref_u = _RunRef(mu0_usage, REF_SEED_MIN, REF_HL_MIN)
        ref_r = _RunRef(mu0_reb, REF_SEED_MIN, REF_HL_MIN)
        ref_s = _RunRef(mu0_stl, REF_SEED_MIN, REF_HL_MIN)
        ref_b = _RunRef(mu0_blk, REF_SEED_MIN, REF_HL_MIN)
        ref_p = _RunRef(mu0_potast, REF_SEED_MIN, REF_HL_MIN)
        _prev_potast_raw = 0.0
        _prev_min_at_potast = 0.0
        for snap in grid:
            while gi < len(games) and games[gi]["game_date"] <= snap:
                g = games[gi]
                mn = g["minutes"] or 0.0
                if mn > 0:
                    c["gp_played_asof"] += 1
                    c["min_asof"] += mn
                    fga = g["fga"] or 0.0; fta = g["fta"] or 0.0; tov = g["turnovers"] or 0.0
                    fg3a = g["fg3a"] or 0.0; fg3m = g["fg3m"] or 0.0
                    fgm = g["fgm"] or 0.0
                    fg2a = fga - fg3a; fg2m = fgm - fg3m
                    reb_g = g["rebounds"] or 0.0
                    stl_g = g["steals"] or 0.0
                    blk_g = g["blocks"] or 0.0
                    used_g = fga + FT_POSS_COEF * fta + tov
                    f_u = _winsor_factor(used_g, mn, ref_u, K_USAGE, c["gp_played_asof"], MIN_GAMES)
                    f_r = _winsor_factor(reb_g, mn, ref_r, K_REB, c["gp_played_asof"], MIN_GAMES)
                    f_s = _winsor_factor(stl_g, mn, ref_s, K_STL, c["gp_played_asof"], MIN_GAMES)
                    f_b = _winsor_factor(blk_g, mn, ref_b, K_BLK, c["gp_played_asof"], MIN_GAMES)
                    c["used_fga"] += fga * f_u
                    c["used_ft_trip"] += FT_POSS_COEF * fta * f_u
                    c["used_tov"] += tov * f_u
                    c["fg3a"] += fg3a; c["fg3m"] += fg3m
                    c["fg2a"] += fg2a; c["fg2m"] += fg2m
                    c["fta"] += fta; c["ftm"] += g["ftm"] or 0.0
                    c["reb"] += reb_g * f_r
                    c["stl"] += stl_g * f_s
                    c["blk"] += blk_g * f_b
                    c["ast"] += g["assists"] or 0.0
                gi += 1
            # rim/mid apportionment of banked fg2 by point-share proxy at this snap
            paint, mr = pshare.get(pid, {}).get(snap, (None, None))
            if paint is not None and mr is not None and (paint + mr) > 0:
                frac_rim = paint / (paint + mr)
            else:
                frac_rim = 0.5
            c["fg2a_rim"] = c["fg2a"] * frac_rim
            c["fg2m_rim"] = c["fg2m"] * frac_rim
            c["fg2a_mid"] = c["fg2a"] * (1 - frac_rim)
            c["fg2m_mid"] = c["fg2m"] * (1 - frac_rim)
            _raw_now = potast.get(pid, {}).get(snap, 0.0)
            _win_min = c["min_asof"] - _prev_min_at_potast
            _win_cnt = _raw_now - _prev_potast_raw
            f_p = _winsor_factor(_win_cnt, _win_min, ref_p, K_POTAST, c["gp_played_asof"], MIN_GAMES)
            c["potential_ast_asof"] += _win_cnt * f_p
            _prev_potast_raw = _raw_now
            _prev_min_at_potast = c["min_asof"]
            if c["gp_played_asof"] > 0:
                row = {"nba_api_id": pid, "season": season, "snapshot_date": snap,
                       "pulled_at": pulled_at}
                row.update({k: c[k] for k in COUNT_COLS})
                out.append(dict(row))
    if out:
        for i in range(0, len(out), 5000):
            upsert(conn, "stat_rate_counts_asof", out[i:i+5000],
                   ["nba_api_id", "season", "snapshot_date"])
    conn.commit()
    return len(out)


def _summary(conn):
    print("\n" + "=" * 66)
    print("stat_rate_counts_asof summary")
    print("=" * 66)
    n = conn.execute("SELECT COUNT(*) FROM stat_rate_counts_asof").fetchone()[0]
    print(f"  rows: {n}")
    print("\n  identity + apportionment sanity on a late-season snapshot (2023):")
    r = conn.execute("""SELECT nba_api_id,
        ROUND(2*fg2m+3*fg3m+ftm,0) pts_from_counts,
        ROUND(fg2a_rim+fg2a_mid,1) rim_plus_mid, ROUND(fg2a,1) fg2a,
        ROUND(potential_ast_asof,0) potast, ROUND(ast,0) ast
        FROM stat_rate_counts_asof
        WHERE season=2023 AND snapshot_date=(SELECT MAX(snapshot_date)
          FROM stat_rate_counts_asof WHERE season=2023)
        ORDER BY pts_from_counts DESC LIMIT 5""").fetchall()
    print(f"  {'pid':>10} {'pts(2fg2+3fg3+ft)':>18} {'rim+mid':>8} {'fg2a':>7} {'potast':>7} {'ast':>6}")
    for x in r:
        print(f"  {x['nba_api_id']:>10} {x['pts_from_counts']:>18} "
              f"{x['rim_plus_mid']:>8} {x['fg2a']:>7} {x['potast']:>7} {x['ast']:>6}")
    print("  (rim+mid must equal fg2a exactly; pts should match the scoring leaders)")
    print("=" * 66)


def build(db_path, season):
    pulled_at = utcnow_iso()
    conn = connect(db_path)
    try:
        _ensure_schema(conn)
        for s in _seasons(conn, season):
            n = build_season(conn, s, pulled_at)
            log.info("season %d: %d rate-count rows", s, n)
        _summary(conn)
    finally:
        conn.close()


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Stat-leader as-of rate-count assembler.")
    p.add_argument("--db", default=str(DEFAULT_DB_PATH))
    p.add_argument("--season", type=int, default=None)
    args = p.parse_args(argv)
    build(Path(args.db), args.season)
    return 0


if __name__ == "__main__":
    sys.exit(main())
