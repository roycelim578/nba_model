"""As-of candidate filter -> candidate_admission.

Computes the admitted-candidate flag per (nba_api_id, season, snapshot_date,
award) for MVP / DPOY / ROTY, using ONLY information available at the snapshot.
The admitted set defines the Plackett-Luce softmax group at train time, so this
filter literally defines the groups the model competes candidates within. It
must therefore be leak-free (every input as-of) and must capture every
high-share finisher (a dropped winner/runner-up is fatal; a dropped thin stray
is negligible).

THE FROZEN FILTER (calibrated against real as-of data; see PHASE2_UPDATE doc):

  MVP  : gp_played>=20 AND (pra_pct>=0.90 OR apg_pct>=0.90)        OR carry-in
  DPOY : gp_played>=20 AND (stocks_pct>=0.90 OR dreb_pct>=0.90)    OR carry-in
  ROTY : rookie-eligible AND gp_played>=15 AND (pra_pct>=0.75 OR apg_pct>=0.75)
         [no carry-in; pool = rookies only; percentiles within rookie pool]

  carry-in = received ANY vote share for the SAME award the prior season.
  All percentiles computed WITHIN the snapshot's rateable pool (UNION limbs).

SOURCING DECISION (load-bearing):
  - pra, apg  : from stg_nba_box_asof (gapless, in-process reconstruction).
  - stocks    : (spg_std + bpg_std) from stg_nba_box_asof, a per-game rate.
                NOT from stg_nba_player_advanced_asof.stl/blk, which are
                Defense-endpoint totals with ~139 transient-failure dead
                snapshots. The box table has NO such gap, so the DPOY primary
                axis is gapless. (Equivalence to the advanced totals/gp was
                cross-checked where both exist.)
  - dreb_pct  : from stg_nba_player_advanced_asof.dreb / gp, the ONLY axis that
                is genuinely advanced-table-only (game logs carry no
                defensive-rebound split). Where dreb is NULL (the dead
                snapshots) the dreb limb simply does not fire; the gapless
                stocks limb carries admission, which D7 showed it does for
                essentially every anchor.

ROOKIE ELIGIBILITY: season == MIN(game-log season) for that nba_api_id. This is
first-APPEARANCE season, NOT draft_year == season (draft-and-stash like Chet
Holmgren, drafted 2022, first games 2023, would break draft-year equality).

POOL DEFINITION: the rateable pool at a snapshot is every player with a box-asof
row at that (season, snapshot_date) clearing the gp floor (gp_played>=20 for
MVP/DPOY, >=15 for ROTY) AND with a non-null axis value. Percentile is
PERCENT_RANK within that pool. For ROTY the pool is restricted to
rookie-eligible players (a rookie competes against rookies, not the league).

LEAK DISCIPLINE: box-asof and advanced-asof are already as-of by construction
(game_date<=snapshot / date_to=snapshot). Carry-in uses prior-season voting
(season-1), which is fully resolved before the current season starts, so it is
legitimately known from the preseason snapshot onward. No feature here reads
current-season outcomes.

Run from project root:
  uv run python -m scripts.features.nba_candidate_filter
  uv run python -m scripts.features.nba_candidate_filter --award DPOY --season 2024
  uv run python -m scripts.features.nba_candidate_filter --trace-dpoy   # D7 re-run
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

try:
    from scripts.common.db import connect, upsert, utcnow_iso as utc_now
except ImportError:  # pragma: no cover
    from db import connect, upsert, utcnow_iso as utc_now  # type: ignore

log = logging.getLogger("nba_candidate_filter")

AWARDS = ("MVP", "DPOY", "ROTY", "6MOTY")

# gp_played floors per award (sample-sanity rateability gate).
GP_FLOOR = {"MVP": 20, "DPOY": 20, "ROTY": 15, "6MOTY": 20}

# production percentile thresholds per award (UNION limbs).
PCT_FLOOR = {"MVP": 0.90, "DPOY": 0.90, "ROTY": 0.75, "6MOTY": 0.60}

# whether the award has a prior-season carry-in limb.
HAS_CARRYIN = {"MVP": True, "DPOY": True, "ROTY": False, "6MOTY": True}

# Bench-gate constants for 6MOTY. Per-game started flags are trustworthy only
# from 2017 (V3 restricts the position field to actual starters then; earlier
# seasons over-label). Pre-2017 the bench gate falls back to an mpg band
# validated at recall 1.0 on the bench-majority set. The starts ratio uses the
# appearance denominator (gp_asof), matching how a start is credited (at tip-off
# regardless of minutes), and sums starts only over rows joinable to the game
# logs so non-appearing listed players do not dilute the count.
STARTS_CLEAN_FROM = 2017
BENCH_MPG_LO = 8.0
BENCH_MPG_HI = 36.0

# Data floor: the first season present in the game logs. ROTY rookie-
# eligibility (first-game-log-season) cannot be computed for this season
# because the prior season is below the floor, so that season's debuts are
# misclassified as rookies and the prior year's genuine rookies leak in via
# the draft-year tolerance (e.g. 1996 admits the 1995 class as false
# rookies). Confirmed empirically that the contamination is confined to this
# one season; 1997+ is clean. ROTY is therefore not produced for the floor
# season. MVP/DPOY are unaffected (they do not depend on rookie detection).
DATA_FLOOR_SEASON = 1996


# ----------------------------------------------------------------------------
# The as-of pool-ranking primitive.
# ----------------------------------------------------------------------------
# A single SQL expression computes percentile-within-(season,snapshot)-pool for
# the per-award axis values. Both the filter (rank the rateable pool to admit)
# and later the loader's relative encodings (rank the admitted set) sit on this
# same primitive: PERCENT_RANK() OVER (PARTITION BY group ORDER BY value), with
# NULL axis values excluded from the pool (a NULL does not clear, and must not
# distort the ranks of present values).
#
# PERCENT_RANK is in [0,1]; for a pool of size n the top value scores 1.0 and
# the bottom 0.0. A >=0.90 gate therefore admits roughly the top decile of the
# pool on that axis, which is the frozen calibration target.


def _axis_pool_cte(award: str) -> str:
    """Return the SQL CTE chain that builds the rateable pool with as-of
    percentiles for one award. Emits a CTE named `ranked` with columns
    (nba_api_id, season, snapshot_date, gp_played, <axis pct columns>).

    The pool, axis sources, and percentile partition differ per award; the
    primitive (PERCENT_RANK within the partition over non-null axis) is shared.
    """
    gp_floor = GP_FLOOR[award]

    if award == "6MOTY":
        # Bench pool: MVP-style pra/apg axes over the gp-floored box pool,
        # intersected with the bench gate. started_asof sums per-game start
        # flags on games on/before the snapshot, joined to the logs so listed
        # non-appearances (started=0 phantoms) drop out. The season switch is
        # applied here so percentiles rank bench players against bench players.
        return f"""
        starts_asof AS (
            SELECT s.nba_api_id, b.season, b.snapshot_date,
                   SUM(CASE WHEN l.game_date <= b.snapshot_date THEN s.started ELSE 0 END) AS started_asof
            FROM stg_nba_game_starts s
            JOIN stg_nba_player_game_logs l
              ON l.nba_api_id = s.nba_api_id AND l.game_id = s.game_id
            JOIN (SELECT DISTINCT season, snapshot_date FROM stg_nba_box_asof) b
              ON b.season = s.season
            GROUP BY s.nba_api_id, b.season, b.snapshot_date
        ),
        pool AS (
            SELECT b.nba_api_id, b.season, b.snapshot_date,
                   b.gp_played_asof AS gp_played,
                   b.pra_std AS pra, b.apg_std AS apg,
                   (COALESCE(b.spg_std,0)+COALESCE(b.bpg_std,0)) AS stocks,
                   b.gp_asof AS gp_asof, b.mpg_std AS mpg,
                   COALESCE(sa.started_asof, 0) AS started_asof
            FROM stg_nba_box_asof b
            JOIN snapshot_grid g
              ON g.season = b.season AND g.snapshot_date = b.snapshot_date
            LEFT JOIN starts_asof sa
              ON sa.nba_api_id = b.nba_api_id AND sa.season = b.season
             AND sa.snapshot_date = b.snapshot_date
            LEFT JOIN (SELECT x.nba_api_id AS nid, gs.season AS sea, CAST(gs.gs AS REAL) / gs.g AS gs_ratio FROM stg_bref_nba_crosswalk x JOIN stg_bref_game_starts gs ON gs.bref_id = x.bref_id WHERE gs.g > 0 AND gs.gs IS NOT NULL) bgs
              ON bgs.nid = b.nba_api_id AND bgs.sea = b.season
            WHERE g.snapshot_kind IN ('weekly','ratings')
              AND b.gp_played_asof >= {gp_floor}
              AND (
                    (b.season >= {STARTS_CLEAN_FROM}
                       AND b.gp_asof > 0
                       AND CAST(COALESCE(sa.started_asof, 0) AS REAL) / b.gp_asof < 0.5)
                 OR (b.season < {STARTS_CLEAN_FROM}
                       AND bgs.gs_ratio IS NOT NULL AND bgs.gs_ratio < 0.5)
                  )
        ),
        ranked AS (
            SELECT p.nba_api_id, p.season, p.snapshot_date, p.gp_played,
                   PERCENT_RANK() OVER (
                       PARTITION BY p.season, p.snapshot_date ORDER BY p.pra
                   ) AS pra_pct,
                   PERCENT_RANK() OVER (
                       PARTITION BY p.season, p.snapshot_date ORDER BY p.apg
                   ) AS apg_pct,
                   PERCENT_RANK() OVER (
                       PARTITION BY p.season, p.snapshot_date ORDER BY p.stocks
                   ) AS stk_pct
            FROM pool p
            WHERE p.pra IS NOT NULL
        )
        """

    if award in ("MVP", "ROTY"):
        # MVP/ROTY axes: pra and apg, both from the box table (gapless).
        # ROTY differs only in the pool restriction (rookies) + gp floor,
        # applied by the caller via the rookie join; the axis SQL is identical.
        base = f"""
        pool AS (
            SELECT b.nba_api_id, b.season, b.snapshot_date,
                   b.gp_played_asof AS gp_played,
                   b.pra_std AS pra, b.apg_std AS apg
            FROM stg_nba_box_asof b
            JOIN snapshot_grid g
              ON g.season = b.season AND g.snapshot_date = b.snapshot_date
            WHERE g.snapshot_kind IN ('weekly','ratings')
              AND b.gp_played_asof >= {gp_floor}
              {{rookie_clause}}
        ),
        ranked AS (
            SELECT p.nba_api_id, p.season, p.snapshot_date, p.gp_played,
                   PERCENT_RANK() OVER (
                       PARTITION BY p.season, p.snapshot_date
                       ORDER BY p.pra
                   ) AS pra_pct,
                   PERCENT_RANK() OVER (
                       PARTITION BY p.season, p.snapshot_date
                       ORDER BY p.apg
                   ) AS apg_pct
            FROM pool p
            WHERE p.pra IS NOT NULL
        )
        """
        return base

    # DPOY: stocks from the box table (gapless), dreb from advanced (gappy).
    # stocks = spg_std + bpg_std (both per-game rates, summed = per-game stocks).
    # dreb_pg = advanced.dreb / advanced.gp (a per-game rate from the total).
    # The two axes come from two tables joined on the as-of key; a player
    # missing the advanced row (or with NULL dreb) still ranks on stocks.
    return f"""
    pool AS (
        SELECT b.nba_api_id, b.season, b.snapshot_date,
               b.gp_played_asof AS gp_played,
               (COALESCE(b.spg_std, 0) + COALESCE(b.bpg_std, 0)) AS stocks,
               CASE WHEN b.spg_std IS NULL AND b.bpg_std IS NULL
                    THEN NULL ELSE 1 END AS stocks_present,
               CASE WHEN a.gp IS NOT NULL AND a.gp > 0 AND a.dreb IS NOT NULL
                    THEN a.dreb / a.gp ELSE NULL END AS dreb_pg
        FROM stg_nba_box_asof b
        JOIN snapshot_grid g
          ON g.season = b.season AND g.snapshot_date = b.snapshot_date
        LEFT JOIN stg_nba_player_advanced_asof a
          ON a.nba_api_id = b.nba_api_id
         AND a.season = b.season
         AND a.snapshot_date = b.snapshot_date
        WHERE g.snapshot_kind IN ('weekly','ratings')
          AND b.gp_played_asof >= {gp_floor}
    ),
    ranked AS (
        SELECT p.nba_api_id, p.season, p.snapshot_date, p.gp_played,
               PERCENT_RANK() OVER (
                   PARTITION BY p.season, p.snapshot_date
                   ORDER BY CASE WHEN p.stocks_present IS NULL THEN NULL ELSE p.stocks END
               ) AS stocks_pct,
               PERCENT_RANK() OVER (
                   PARTITION BY p.season, p.snapshot_date
                   ORDER BY p.dreb_pg
               ) AS dreb_pct,
               p.stocks_present, p.dreb_pg
        FROM pool p
    )
    """


# ----------------------------------------------------------------------------
# Carry-in and rookie-eligibility (identity-joined through players).
# ----------------------------------------------------------------------------

def _carryin_ids(conn, award: str, season: int) -> set[int]:
    """nba_api_ids that received ANY vote share for `award` in `season - 1`.

    Joins award_voting (player_id-keyed) to players for nba_api_id. A player
    with no nba_api_id (synthetic/bref-only) cannot match a box row anyway, so
    dropping them here is harmless.
    """
    rows = conn.execute(
        """
        SELECT DISTINCT p.nba_api_id
        FROM award_voting v
        JOIN players p ON p.player_id = v.player_id
        WHERE v.award = ? AND v.season = ? AND v.vote_share > 0
          AND p.nba_api_id IS NOT NULL
        """,
        (award, season - 1),
    ).fetchall()
    return {r["nba_api_id"] for r in rows}


def _rookie_ids(conn, season: int) -> set[int]:
    """nba_api_ids that are rookie-eligible in `season`.

    Eligible iff FIRST game-log season == season (first-appearance, which keeps
    draft-and-stash correct: Holmgren first appears 2023 though drafted 2022)
    AND the player is not a known veteran whose game-log history merely starts at
    the data floor. The latter is the 1996-data-floor trap: in 1996 every player
    whose career began in or before 1996 has MIN(season)=1996 regardless of true
    rookie status, so first-appearance alone admits ~333 misclassified veterans.

    Disambiguation via draft_year: a true rookie's draft_year is at or within one
    season of first appearance (the one-season tolerance preserves draft-and-
    stash); a misclassified veteran's draft_year predates the season by years. A
    NULL draft_year cannot be disambiguated and is admitted (rare in normal
    seasons; the residual 1996 NULLs are handled by the season-1996 policy
    decided after inspecting the post-fix set).
    """
    rows = conn.execute(
        """
        SELECT g.nba_api_id
        FROM (
            SELECT nba_api_id, MIN(season) AS first_yr
            FROM stg_nba_player_game_logs GROUP BY nba_api_id
        ) g
        LEFT JOIN players p ON p.nba_api_id = g.nba_api_id
        WHERE g.first_yr = ?
          AND (p.draft_year IS NULL OR ? - p.draft_year <= 1)
        """,
        (season, season),
    ).fetchall()
    return {r["nba_api_id"] for r in rows}


# ----------------------------------------------------------------------------
# Filter evaluation per award.
# ----------------------------------------------------------------------------

def _admitted_rows(conn, award: str, season: int, pulled_at: str) -> list[dict]:
    """Compute admitted candidate rows for one (award, season).

    Returns one row per admitted (nba_api_id, season, snapshot_date) with the
    limb that fired, for diagnostics. Non-admitted players produce no row (the
    loader treats absence as 'not a candidate at this snapshot').
    """
    if award == "ROTY" and season == DATA_FLOOR_SEASON:
        # rookie identification is unreliable at the data floor; skip.
        return []

    pct = PCT_FLOOR[award]
    rookie_clause = ""
    rookie_ids: set[int] | None = None
    if award == "ROTY":
        rookie_ids = _rookie_ids(conn, season)
        if not rookie_ids:
            return []
        id_list = ",".join(str(i) for i in rookie_ids)
        rookie_clause = f"AND b.nba_api_id IN ({id_list})"

    cte = _axis_pool_cte(award).format(rookie_clause=rookie_clause)

    if award == "6MOTY":
        sql = f"""
        WITH {cte}
        SELECT nba_api_id, season, snapshot_date, gp_played,
               pra_pct, apg_pct,
               (pra_pct >= ? OR apg_pct >= ? OR stk_pct >= ?) AS production_limb
        FROM ranked
        WHERE season = ?
        """
        rows = conn.execute(sql, (pct, pct, pct, season)).fetchall()
        axis_cols = ("pra_pct", "apg_pct")
    elif award in ("MVP", "ROTY"):
        sql = f"""
        WITH {cte}
        SELECT nba_api_id, season, snapshot_date, gp_played,
               pra_pct, apg_pct,
               (pra_pct >= ? OR apg_pct >= ?) AS production_limb
        FROM ranked
        WHERE season = ?
        """
        rows = conn.execute(sql, (pct, pct, season)).fetchall()
        axis_cols = ("pra_pct", "apg_pct")
    else:  # DPOY
        sql = f"""
        WITH {cte}
        SELECT nba_api_id, season, snapshot_date, gp_played,
               stocks_pct, dreb_pct,
               ((stocks_present IS NOT NULL AND stocks_pct >= ?)
                 OR (dreb_pg IS NOT NULL AND dreb_pct >= ?)) AS production_limb
        FROM ranked
        WHERE season = ?
        """
        rows = conn.execute(sql, (pct, pct, season)).fetchall()
        axis_cols = ("stocks_pct", "dreb_pct")

    carryin = _carryin_ids(conn, award, season) if HAS_CARRYIN[award] else set()

    out: list[dict] = []
    for r in rows:
        prod = bool(r["production_limb"])
        carry = r["nba_api_id"] in carryin
        if not (prod or carry):
            continue
        out.append({
            "nba_api_id": r["nba_api_id"],
            "season": season,
            "snapshot_date": r["snapshot_date"],
            "award": award,
            "admitted": 1,
            "limb_production": int(prod),
            "limb_carryin": int(carry),
            "limb_sticky": 0,
            "axis_a_pct": r[axis_cols[0]],
            "axis_b_pct": r[axis_cols[1]],
            "gp_played_asof": r["gp_played"],
            "pulled_at": pulled_at,
        })

    # Carry-in admits a player from the PRESEASON snapshot onward, even before
    # they have a box row (gp floor unmet). Add preseason/early carry-in rows
    # that the production-pool query (gp-floored) could not have produced.
    if carryin:
        out = _add_carryin_only_rows(conn, award, season, carryin,
                                     {(o["nba_api_id"], o["snapshot_date"]) for o in out},
                                     pulled_at, out)
    return out


def _add_carryin_only_rows(conn, award, season, carryin, existing_keys,
                           pulled_at, out):
    """Admit carry-in players at every grid snapshot where they are not already
    admitted by production. This covers the preseason snapshot and any
    early-season snapshot before the player clears the gp floor: a returning
    vote-getter is a candidate from day zero by reputation."""
    grid = conn.execute(
        "SELECT snapshot_date FROM snapshot_grid WHERE season = ? ORDER BY snapshot_date",
        (season,),
    ).fetchall()
    for cid in carryin:
        for g in grid:
            sd = g["snapshot_date"]
            if (cid, sd) in existing_keys:
                continue
            out.append({
                "nba_api_id": cid, "season": season, "snapshot_date": sd,
                "award": award, "admitted": 1,
                "limb_production": 0, "limb_carryin": 1, "limb_sticky": 0,
                "axis_a_pct": None, "axis_b_pct": None,
                "gp_played_asof": None, "pulled_at": pulled_at,
            })
    return out


def _apply_forward_stick(conn, award: str, season: int, rows: list[dict],
                         pulled_at: str) -> list[dict]:
    """Monotonic-within-season admission: once a player clears the PRODUCTION
    limb at any snapshot, admit them at every LATER grid snapshot in that season
    too, even if their production later dips below the floor.

    Leak-clean: a sticky row at snapshot T is justified entirely by a clearance
    at some snapshot < T (past information only). The admitted set therefore only
    grows within a season, never shrinks ('stays on the ballot once earned').

    Rationale (from the D7 trace): slow-starting anchors (both Draymonds) clear
    production in Jan/Feb after an early miss; forward-stick admits them from
    their first clear onward, eliminating mid-season flicker. It does NOT rescue
    pre-first-clear early rows (no evidence yet) nor never-clearing blind-spot
    anchors (Bridges 2021), which remain correctly missed.

    Only PRODUCTION clears seed the stick (limb_production == 1). Carry-in rows
    do not seed it: carry-in already admits all season by construction, so there
    is nothing to forward-fill, and a carry-in row should not manufacture a
    'sticky' production claim the player never earned on the court.
    """
    # ordered weekly+ratings grid for the season (preseason excluded: no games,
    # nothing to be sticky about, and carry-in already covers preseason)
    grid = [r["snapshot_date"] for r in conn.execute(
        "SELECT snapshot_date FROM snapshot_grid "
        "WHERE season = ? AND snapshot_kind IN ('weekly','ratings') "
        "ORDER BY snapshot_date", (season,)).fetchall()]
    grid_pos = {d: i for i, d in enumerate(grid)}

    # existing admitted keys, and per-player earliest PRODUCTION-clear position
    existing = {(r["nba_api_id"], r["snapshot_date"]) for r in rows}
    first_prod_pos: dict[int, int] = {}
    for r in rows:
        if r["limb_production"] == 1:
            pos = grid_pos.get(r["snapshot_date"])
            if pos is not None:
                pid = r["nba_api_id"]
                if pid not in first_prod_pos or pos < first_prod_pos[pid]:
                    first_prod_pos[pid] = pos

    added: list[dict] = []
    for pid, first_pos in first_prod_pos.items():
        for d in grid[first_pos + 1:]:
            if (pid, d) in existing:
                continue
            added.append({
                "nba_api_id": pid, "season": season, "snapshot_date": d,
                "award": award, "admitted": 1,
                "limb_production": 0, "limb_carryin": 0, "limb_sticky": 1,
                "axis_a_pct": None, "axis_b_pct": None,
                "gp_played_asof": None, "pulled_at": pulled_at,
            })
            existing.add((pid, d))
    return rows + added


def populate(conn, awards=AWARDS, season: int | None = None) -> dict:
    pulled_at = utc_now()
    seasons = ([season] if season is not None
               else [r["season"] for r in conn.execute(
                   "SELECT DISTINCT season FROM snapshot_grid ORDER BY season")])
    summary = {"awards": list(awards), "rows": 0, "by_award": {}}
    for award in awards:
        a_rows = 0
        for s in seasons:
            rows = _admitted_rows(conn, award, s, pulled_at)
            rows = _apply_forward_stick(conn, award, s, rows, pulled_at)
            if rows:
                upsert(conn, "candidate_admission", rows,
                       ["nba_api_id", "season", "snapshot_date", "award"])
                a_rows += len(rows)
        summary["by_award"][award] = a_rows
        summary["rows"] += a_rows
        log.info("%s: %d admitted rows across %d seasons", award, a_rows, len(seasons))
    conn.commit()
    return summary


# ----------------------------------------------------------------------------
# D7 re-trace: DPOY anchors vs the frozen limb, stocks sourced GAPLESSLY from
# the box table. The point of the re-run is to see whether the misses that were
# tied to dead Defense snapshots resolve now that stocks no longer depends on
# the gappy advanced table. Read-only; prints, writes nothing.
# ----------------------------------------------------------------------------

def trace_dpoy(conn) -> None:
    """Print each season's top-3 DPOY finishers at their FIRST gp>=20 snapshot,
    with stocks (box-sourced, gapless) and dreb (advanced, where present)
    percentiles, and whether the frozen 0.90-OR limb clears."""
    sql = """
    WITH dpoy_top3 AS (
        SELECT v.season, v.player_id, v.rank AS dpoy_rank, v.vote_share,
               p.name, p.nba_api_id
        FROM award_voting v JOIN players p ON p.player_id = v.player_id
        WHERE v.award='DPOY' AND v.rank<=3 AND v.season>=2013
          AND p.nba_api_id IS NOT NULL
    ),
    pool AS (
        SELECT b.season, b.snapshot_date, b.nba_api_id,
               (COALESCE(b.spg_std,0)+COALESCE(b.bpg_std,0)) AS stocks,
               CASE WHEN b.spg_std IS NULL AND b.bpg_std IS NULL THEN NULL ELSE 1 END AS sp,
               CASE WHEN a.gp IS NOT NULL AND a.gp>0 AND a.dreb IS NOT NULL
                    THEN a.dreb/a.gp ELSE NULL END AS dreb_pg
        FROM stg_nba_box_asof b
        JOIN snapshot_grid g ON g.season=b.season AND g.snapshot_date=b.snapshot_date
        LEFT JOIN stg_nba_player_advanced_asof a
          ON a.nba_api_id=b.nba_api_id AND a.season=b.season AND a.snapshot_date=b.snapshot_date
        WHERE g.snapshot_kind='weekly' AND b.gp_played_asof>=20
    ),
    first_snap AS (
        SELECT d.season,d.name,d.dpoy_rank,d.vote_share,d.nba_api_id,
               MIN(po.snapshot_date) AS first_date
        FROM dpoy_top3 d JOIN pool po
          ON po.season=d.season AND po.nba_api_id=d.nba_api_id AND po.sp IS NOT NULL
        GROUP BY d.season,d.nba_api_id
    ),
    ranked AS (
        SELECT po.season,po.snapshot_date,po.nba_api_id,po.stocks,po.dreb_pg,
               PERCENT_RANK() OVER (PARTITION BY po.season,po.snapshot_date
                   ORDER BY CASE WHEN po.sp IS NULL THEN NULL ELSE po.stocks END) AS stocks_pct,
               PERCENT_RANK() OVER (PARTITION BY po.season,po.snapshot_date
                   ORDER BY po.dreb_pg) AS dreb_pct
        FROM pool po
    )
    SELECT f.season,f.name,f.dpoy_rank,f.vote_share,f.first_date,
           r.stocks_pct,r.dreb_pct,
           CASE WHEN r.stocks_pct>=0.90 OR r.dreb_pct>=0.90 THEN 'CLEARS' ELSE 'MISS' END AS limb
    FROM first_snap f JOIN ranked r
      ON r.season=f.season AND r.snapshot_date=f.first_date AND r.nba_api_id=f.nba_api_id
    ORDER BY f.season,f.dpoy_rank
    """
    rows = conn.execute(sql).fetchall()
    misses = 0
    print("season name dpoy_rank share first_date stocks_pct dreb_pct limb")
    for r in rows:
        if r["limb"] == "MISS":
            misses += 1
        sp = "None" if r["stocks_pct"] is None else f"{r['stocks_pct']:.3f}"
        dp = "None" if r["dreb_pct"] is None else f"{r['dreb_pct']:.3f}"
        print(f"{r['season']} {r['name']} {r['dpoy_rank']} {r['vote_share']:.3f} "
              f"{r['first_date']} {sp} {dp} {r['limb']}")
    print(f"\nTOTAL: {len(rows)} anchors, {misses} MISS "
          f"(box-sourced gapless stocks; compare to the 3 misses on advanced-sourced stocks)")


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="As-of candidate filter -> candidate_admission.")
    ap.add_argument("--db", type=Path, default=Path("data/awards.db"))
    ap.add_argument("--award", choices=AWARDS, default=None)
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--trace-dpoy", action="store_true", help="D7 re-run: DPOY anchors vs frozen limb, box-sourced stocks (read-only)")
    args = ap.parse_args(argv)
    conn = connect(args.db)
    if args.trace_dpoy:
        trace_dpoy(conn)
        conn.close()
        return 0
    awards = (args.award,) if args.award else AWARDS
    summary = populate(conn, awards=awards, season=args.season)
    conn.close()
    log.info("FILTER SUMMARY: %s", summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
