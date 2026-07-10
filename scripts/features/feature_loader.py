"""Feature loader -> feature_stats_asof (materialised join) + in-memory design matrix.

The Phase 2 convergence point. Two stages (PHASE2_CONTRACT Section 7):

  STAGE 1 (materialise): join the admitted candidate set to every as-of source
  on the shared (nba_api_id, season, snapshot_date) key, plus team context (via
  game-log team resolution) and honours/carryover (via identity hops), into the
  durable table feature_stats_asof. Stable as-of FACTS; expensive join cached
  here, regenerated only when staging changes.

  STAGE 2 (load_design_matrix): read feature_stats_asof and compute RELATIVE
  ENCODINGS and forward-error-engine inputs IN MEMORY per (season, award,
  snapshot) group, emitting the grouped design matrix for the Plackett-Luce
  objective. Volatile/cheap transforms, kept fresh, not stored.

KEY DECISIONS:
  - Admitted set is the softmax group; vote_share is the label, zero-padded over
    admitted-but-unvoted (never renormalised over vote-getters only).
  - Trajectory columns: only _std,_l10,_l30,_ema,_l10_vs_l30,_l10_vs_std. _l5 and
    _l20 are DROPPED (redundant/multicollinear); they stay in the box table but
    are never read here.
  - Team context via current-team-as-of: team_records is (team_id,snapshot_date)
    keyed and neither admission nor box carries team_id, so team is the team_id
    of the player's most recent game with game_date<=snapshot (traded-aware).
  - Relative encodings over the subset of the group that HAS the feature; a
    NULL-feature member is excluded from that feature's rank pool and gets NULL
    encodings (LightGBM missing branches).
  - Carryover is voting-derived ONLY: prior vote share, prior rank, prior-winner
    flag, prior All-NBA/All-Def honour share. No raw prior box/advanced stats.

LEAK DISCIPLINE: sources are as-of by construction; team resolution filters
game_date<=snapshot; carryover reads season-1 (resolved before current season).

Run from project root:
  uv run python -m scripts.features.feature_loader --materialise [--season YYYY]
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

from scripts.features.positional_z import (
    gp_floor as _posz_gp_floor,
    attach_candidate_position,
    annotate_positional_z,
)
_POSZ_GP_FLOOR = _posz_gp_floor()

try:
    # canonical: package-qualified, real db.py exports utcnow_iso (not utc_now)
    from scripts.common.db import connect, upsert, utcnow_iso as utc_now
except ImportError:  # pragma: no cover - standalone / test fallback
    from db import connect, upsert, utcnow_iso as utc_now  # type: ignore

log = logging.getLogger("feature_loader")

# honour enum values in stg_player_honours (confirm against DB; see --check-honours)
HONOUR_ALL_NBA = "ALL_NBA"
HONOUR_ALL_DEF = "ALL_DEF"

_BOX_STATS = ("ppg", "rpg", "apg", "spg", "bpg", "mpg", "pra",
              "fg_pct", "fg3_pct", "ft_pct", "ts_pct", "efg_pct")
_TRAJ = ("std", "ema", "l10_vs_std")  # MA prune: l10 subsumed by ema; l30, l10_vs_l30 dropped


def box_feature_cols() -> list[str]:
    return [f"{s}_{t}" for s in _BOX_STATS for t in _TRAJ]


_ADV_CORE = ("off_rating", "def_rating", "net_rating", "usg_pct", "ts_pct",
             "pace", "pie", "plus_minus", "def_rim_fgm", "def_rim_fga",
             "dreb", "stl", "blk")

_ADV_EXT = ("dd2", "td3", "pfd", "blka", "pct_pts_paint", "pct_pts_3pt",
            "pct_pts_mr", "pct_pts_ft", "pct_pts_fb", "pct_uast_fgm",
            "pct_pts", "pct_ast", "pts_off_tov", "pts_fb", "pts_2nd_chance",
            "pts_paint", "ast_pct", "ast_to", "reb_pct", "oreb_pct",
            "dreb_pct", "poss", "opp_pts_paint", "opp_pts_fb",
            "opp_pts_2nd_chance", "opp_pts_off_tov", "def_ws", "pct_blk",
            "pct_stl", "pct_dreb", "potential_ast", "ast_pts_created",
            "secondary_ast", "time_of_poss", "touches", "avg_sec_per_touch",
            "avg_drib_per_touch", "front_ct_touches", "def_rim_pct",
            "def_rim_freq")

_TEAM = ("wins", "losses", "win_pct", "net_rating", "off_rating",
         "def_rating", "conf_rank", "projected_seed", "sos_remaining")

_AVAIL = ("team_games_asof", "games_played_asof", "games_missed_asof",
          "availability_pct_asof", "missed_last_10_team_games",
          "missed_last_30_team_games", "current_absence_streak",
          "on_65_game_pace_flag")

# ---------------------------------------------------------------------------
# SEASON-OVER-SEASON DELTAS (prior-final levels + in-memory _d1 transforms).
# We materialise the prior-final LEVEL (a durable as-of fact: the season-1
# 'ratings' value); the DELTA <col>_d1 = <col> - prior_<col> is computed in
# memory in Stage 2 and never stored. Level forms only: NOT the within-season
# _l10/_l30/_ema/_vs trajectory forms (those are already change-features; a
# delta-of-a-delta is noise). NOT the carry_* features (already last-season
# facts). ext_ deltas deferred (sparse pre-2017). See schema fragment header.
#
# Each entry is the materialised column name in feature_stats_asof. The prior
# level is prior_<col>; the in-memory delta is <col>_d1. conf_rank / seed are
# ordinal (lower=better) so a NEGATIVE _d1 is improvement; the relative encoder
# treats _d1 as a raw number and the sign convention is handled downstream.
_DELTA_BOX = ("box_ppg_std", "box_rpg_std", "box_apg_std", "box_spg_std",
              "box_bpg_std", "box_mpg_std", "box_pra_std", "box_fg_pct_std",
              "box_fg3_pct_std", "box_ft_pct_std", "box_ts_pct_std",
              "box_efg_pct_std")
_DELTA_ADV = ("adv_off_rating", "adv_def_rating", "adv_net_rating",
              "adv_usg_pct", "adv_ts_pct", "adv_pie")
_DELTA_TEAM = ("team_win_pct", "team_net_rating", "team_off_rating",
               "team_def_rating", "team_conf_rank", "team_projected_seed")


def delta_feature_cols() -> list[str]:
    """Materialised columns that get a season-over-season delta. The prior level
    is prior_<col>; the in-memory delta is <col>_d1."""
    return list(_DELTA_BOX) + list(_DELTA_ADV) + list(_DELTA_TEAM)


def prior_level_cols() -> list[str]:
    """The prior_<col> materialised columns (durable as-of facts)."""
    return [f"prior_{c}" for c in delta_feature_cols()]

# Features that get relative encodings within the group. The key directional
# signals: production (box), impact (advanced), team success. Not every column
# (that would triple ~100 cols); the model architecture calls for relative
# encodings on the stats the PL model needs relativity for.
_HUSTLE = ("defl", "charge", "cont2", "cont3", "conttot", "dloose", "oloose", "scrast")
_HUSTLE_TRAJ = ("std", "ema", "l10_vs_std")
_DEFEND_BANDS = ("overall", "fg3", "fg2")


def hustle_feature_cols() -> list[str]:
    return [f"{p}_{t}" for p in _HUSTLE for t in _HUSTLE_TRAJ]


def defend_feature_cols() -> list[str]:
    pct = [f"dpct_{b}_{t}" for b in _DEFEND_BANDS for t in _HUSTLE_TRAJ]
    vol = [f"dfga_{b}_std" for b in _DEFEND_BANDS]
    return pct + vol


# Wave 1: percentile-bucket bare stat names (heavy-tailed within the candidate set;
# a single outlier leader distorts mean/std, so percentile is robust and z is not).
# Everything else defaults to z (bounded rates, fractions, ratings). Every momentum
# ratio (_l10_vs_std) and every hustle stat is percentile regardless.
_PCT_BOX_STATS = ("spg", "bpg")
_PCT_ADV = ("stl", "blk", "dreb", "def_rim_fgm", "def_rim_fga", "plus_minus")
_PCT_EXT = ("dd2", "td3", "pfd", "blka", "pts_off_tov", "pts_fb", "pts_2nd_chance",
            "pts_paint", "poss", "opp_pts_paint", "opp_pts_fb", "opp_pts_2nd_chance",
            "opp_pts_off_tov", "def_ws", "potential_ast", "ast_pts_created",
            "secondary_ast", "touches", "front_ct_touches", "ast_to",
            "avg_sec_per_touch", "avg_drib_per_touch", "def_rim_freq")
_PCT_AVAIL = ("current_absence_streak", "missed_last_10_team_games",
              "missed_last_30_team_games")


def _relative_kind(col: str) -> str:
    """Return 'z' or 'pct' for a feature's single relative encoding. Deltas (_d1)
    inherit their parent's kind. Unknown families default to 'z'."""
    if col.endswith("_d1"):
        return _relative_kind(col[:-3])
    if col.endswith("_l10_vs_std"):
        return "pct"
    if col.startswith("hus_"):
        return "pct"
    if col.startswith("dfn_dfga_"):
        return "pct"
    if col.startswith("dfn_dpct_"):
        return "z"
    if col.startswith("box_"):
        stat = col[len("box_"):].rsplit("_", 1)[0]
        return "pct" if stat in _PCT_BOX_STATS else "z"
    if col.startswith("adv_"):
        return "pct" if col[len("adv_"):] in _PCT_ADV else "z"
    if col.startswith("ext_"):
        return "pct" if col[len("ext_"):] in _PCT_EXT else "z"
    if col.startswith("team_"):
        return "z"
    if col in _PCT_AVAIL:
        return "pct"
    return "z"


def relative_feature_cols() -> list[str]:
    """Wave 1: universal relative encoding over the full eligible level set. One
    relative per feature (z or percentile per _relative_kind); rank and delta_leader
    dropped. The exclude bucket (carry incl years_repeat, position, era/year/age,
    time-to-resolution, prior_ levels, flags, keys, targets) is simply not listed here.
    Downstream pairwise/VIF/L1 selection prunes from this set; do NOT pre-prune by hand."""
    box = [f"box_{c}" for c in box_feature_cols()]
    adv = [f"adv_{c}" for c in _ADV_CORE]
    ext = [f"ext_{c}" for c in _ADV_EXT]
    hus = [f"hus_{c}" for c in hustle_feature_cols()]
    dfn = [f"dfn_{c}" for c in defend_feature_cols()]
    team = [f"team_{c}" for c in ("win_pct", "net_rating", "off_rating", "def_rating",
                                  "conf_rank", "projected_seed", "sos_remaining")]
    avail = ["games_played_asof", "games_missed_asof", "availability_pct_asof",
             "current_absence_streak", "missed_last_10_team_games",
             "missed_last_30_team_games"]
    deltas = [f"{c}_d1" for c in delta_feature_cols()]
    carried = ["carry_prior_vote_share", "on_65_game_pace_flag"]
    return box + adv + ext + hus + dfn + team + avail + deltas + carried

def materialised_columns() -> list[str]:
    """The full column list of feature_stats_asof, in order, for DDL + insert."""
    keys = ["player_id", "nba_api_id", "season", "snapshot_date", "award"]
    admin = ["limb_production", "limb_carryin", "limb_sticky", "gp_played_asof",
             "week_index", "snapshot_kind"]
    box = [f"box_{c}" for c in box_feature_cols()]
    adv = [f"adv_{c}" for c in _ADV_CORE]
    ext = [f"ext_{c}" for c in _ADV_EXT]
    hustle = [f"hus_{c}" for c in hustle_feature_cols()]
    defend = [f"dfn_{c}" for c in defend_feature_cols()]
    team = ["resolved_team_id"] + [f"team_{c}" for c in _TEAM]
    avail = list(_AVAIL)
    label = ["label_vote_share", "label_won_flag", "label_rank"]
    carry = ["carry_prior_vote_share", "carry_prior_rank", "carry_prior_winner",
             "carry_all_nba_share", "carry_all_def_share", "carry_years_repeat"]
    prior = prior_level_cols()  # season-over-season prior-final levels
    return keys + admin + box + adv + ext + hustle + defend + team + avail + label + carry + prior + ["pulled_at"]


def _build_select(season: int | None) -> tuple[str, dict]:
    box_sel = ", ".join(f"b.{c} AS box_{c}" for c in box_feature_cols())
    adv_sel = ", ".join(f"a.{c} AS adv_{c}" for c in _ADV_CORE)
    ext_sel = ", ".join(f"e.{c} AS ext_{c}" for c in _ADV_EXT)
    hus_sel = ", ".join(f"h.{c} AS hus_{c}" for c in hustle_feature_cols())
    dfn_sel = ", ".join(f"d.{c} AS dfn_{c}" for c in defend_feature_cols())
    team_sel = ", ".join(f"trf.{c} AS team_{c}" for c in _TEAM)
    avail_sel = ", ".join(f"av.{c} AS {c}" for c in _AVAIL)

    # Prior-final SELECTs. The prior-season tables are aliased pb (box), pa
    # (advanced), ptrf (team_records). Strip the family prefix from each delta
    # col to recover the bare staging column name (box_ppg_std -> ppg_std,
    # adv_net_rating -> net_rating, team_win_pct -> win_pct), since the staging
    # tables carry bare column names. The materialised target is prior_<col>.
    def _bare(col: str) -> str:
        for pfx in ("box_", "adv_", "team_"):
            if col.startswith(pfx):
                return col[len(pfx):]
        return col
    prior_box_sel = ", ".join(
        f"pb.{_bare(c)} AS prior_{c}" for c in _DELTA_BOX)
    prior_adv_sel = ", ".join(
        f"pa.{_bare(c)} AS prior_{c}" for c in _DELTA_ADV)
    prior_team_sel = ", ".join(
        f"ptrf.{_bare(c)} AS prior_{c}" for c in _DELTA_TEAM)

    params: dict = {"all_nba": HONOUR_ALL_NBA, "all_def": HONOUR_ALL_DEF}
    season_filter = ""
    if season is not None:
        season_filter = "AND ca.season = :season"
        params["season"] = season

    sql = f"""
    WITH team_res AS (
        SELECT ps.nba_api_id, ps.season, ps.snapshot_date,
               (SELECT l.team_id FROM stg_nba_player_game_logs l
                 WHERE l.nba_api_id = ps.nba_api_id AND l.season = ps.season
                   AND l.game_date <= ps.snapshot_date
                 ORDER BY l.game_date DESC, l.game_id DESC LIMIT 1) AS team_id
        FROM (SELECT DISTINCT nba_api_id, season, snapshot_date
              FROM candidate_admission) ps
    ),
    -- season-1 'ratings' snapshot_date (season_end + 14d), one per season; the
    -- prior-final anchor. NULL for the earliest season (1996: no 1995 row), so
    -- 1996 prior_* fall NULL by construction (data floor). Strictly before any
    -- snapshot of the current season -> no lookahead (same discipline as pv).
    prior_ratings AS (
        SELECT DISTINCT ca.season AS season,
               (SELECT sg.snapshot_date FROM snapshot_grid sg
                 WHERE sg.season = ca.season - 1 AND sg.snapshot_kind = 'ratings'
                 LIMIT 1) AS prior_ratings_date
        FROM candidate_admission ca
    ),
    -- last season's FINAL team for each candidate: most recent game on/before
    -- the season-1 ratings date in season-1. Mirrors team_res, parametrised to
    -- the prior season. NULL if the player did not play last season (injury /
    -- true rookie) -> prior_team_* NULL, which is the correct "no last year".
    team_res_prior AS (
        SELECT ps.nba_api_id, ps.season,
               (SELECT l.team_id FROM stg_nba_player_game_logs l
                 WHERE l.nba_api_id = ps.nba_api_id AND l.season = ps.season - 1
                   AND l.game_date <= pr.prior_ratings_date
                 ORDER BY l.game_date DESC, l.game_id DESC LIMIT 1) AS team_id,
               pr.prior_ratings_date AS prior_ratings_date
        FROM (SELECT DISTINCT nba_api_id, season FROM candidate_admission) ps
        JOIN prior_ratings pr ON pr.season = ps.season
    )
    SELECT
        p.player_id, ca.nba_api_id, ca.season, ca.snapshot_date, ca.award,
        ca.limb_production, ca.limb_carryin, ca.limb_sticky, ca.gp_played_asof,
        g.week_index, g.snapshot_kind,
        {box_sel},
        {adv_sel},
        {ext_sel},
        {hus_sel},
        {dfn_sel},
        tr.team_id AS resolved_team_id,
        {team_sel},
        {avail_sel},
        COALESCE(v.vote_share, 0.0) AS label_vote_share,
        COALESCE(v.won_flag, 0) AS label_won_flag,
        v.rank AS label_rank,
        pv.vote_share AS carry_prior_vote_share,
        pv.rank AS carry_prior_rank,
        CASE WHEN pv.won_flag = 1 THEN 1 ELSE 0 END AS carry_prior_winner,
        hn.share AS carry_all_nba_share,
        hd.share AS carry_all_def_share,
        MAX(0, COALESCE((SELECT COUNT(*) FROM award_voting w
                         WHERE w.player_id = p.player_id AND w.award = ca.award
                           AND w.won_flag = 1 AND w.season < ca.season), 0) - 1)
            AS carry_years_repeat,
        {prior_box_sel},
        {prior_adv_sel},
        {prior_team_sel},
        :pulled_at AS pulled_at
    FROM candidate_admission ca
    JOIN players p ON p.nba_api_id = ca.nba_api_id
    JOIN snapshot_grid g ON g.season = ca.season AND g.snapshot_date = ca.snapshot_date
    LEFT JOIN stg_nba_box_asof b
      ON b.nba_api_id = ca.nba_api_id AND b.season = ca.season AND b.snapshot_date = ca.snapshot_date
    LEFT JOIN stg_nba_player_advanced_asof a
      ON a.nba_api_id = ca.nba_api_id AND a.season = ca.season AND a.snapshot_date = ca.snapshot_date
    LEFT JOIN stg_nba_player_asof_ext e
      ON e.nba_api_id = ca.nba_api_id AND e.season = ca.season AND e.snapshot_date = ca.snapshot_date
    LEFT JOIN stg_nba_hustle_asof h
      ON h.nba_api_id = ca.nba_api_id AND h.season = ca.season AND h.snapshot_date = ca.snapshot_date
    LEFT JOIN stg_nba_defend_asof d
      ON d.nba_api_id = ca.nba_api_id AND d.season = ca.season AND d.snapshot_date = ca.snapshot_date
    LEFT JOIN team_res tr
      ON tr.nba_api_id = ca.nba_api_id AND tr.season = ca.season AND tr.snapshot_date = ca.snapshot_date
    LEFT JOIN team_records trf
      ON trf.team_id = tr.team_id AND trf.snapshot_date = ca.snapshot_date
    LEFT JOIN stg_nba_availability_asof av
      ON av.nba_api_id = ca.nba_api_id AND av.season = ca.season AND av.snapshot_date = ca.snapshot_date
    LEFT JOIN award_voting v
      ON v.player_id = p.player_id AND v.season = ca.season AND v.award = ca.award
    LEFT JOIN award_voting pv
      ON pv.player_id = p.player_id AND pv.season = ca.season - 1 AND pv.award = ca.award
    LEFT JOIN (SELECT bref_id, season, award_share AS share
               FROM stg_player_honours WHERE honour = :all_nba) hn
      ON hn.bref_id = p.bref_id AND hn.season = ca.season - 1
    LEFT JOIN (SELECT bref_id, season, award_share AS share
               FROM stg_player_honours WHERE honour = :all_def) hd
      ON hd.bref_id = p.bref_id AND hd.season = ca.season - 1
    -- ---- prior-final (season-1 'ratings') joins for the delta feature set ----
    -- prior box + advanced: keyed (nba_api_id, season-1, prior_ratings_date).
    LEFT JOIN prior_ratings prg ON prg.season = ca.season
    LEFT JOIN stg_nba_box_asof pb
      ON pb.nba_api_id = ca.nba_api_id AND pb.season = ca.season - 1
         AND pb.snapshot_date = prg.prior_ratings_date
    LEFT JOIN stg_nba_player_advanced_asof pa
      ON pa.nba_api_id = ca.nba_api_id AND pa.season = ca.season - 1
         AND pa.snapshot_date = prg.prior_ratings_date
    -- prior team: resolve last season's final team, then its final team_records
    -- row at the season-1 ratings date (team_records PK is team_id,snapshot_date).
    LEFT JOIN team_res_prior trp
      ON trp.nba_api_id = ca.nba_api_id AND trp.season = ca.season
    LEFT JOIN team_records ptrf
      ON ptrf.team_id = trp.team_id AND ptrf.snapshot_date = trp.prior_ratings_date
    WHERE 1=1 {season_filter}
    """
    return sql, params


def materialise(conn, season: int | None = None) -> dict:
    pulled_at = utc_now()
    sql, params = _build_select(season)
    params["pulled_at"] = pulled_at
    cols = materialised_columns()
    rows = conn.execute(sql, params).fetchall()
    out = [dict(r) for r in rows]
    # upsert in batches keyed on the natural PK
    if out:
        upsert(conn, "feature_stats_asof", out,
               ["nba_api_id", "season", "snapshot_date", "award"])
    conn.commit()
    # integrity tallies
    n_team_null = sum(1 for r in out if r["resolved_team_id"] is None)
    # admitted rows dropped by the inner JOIN players (no identity row). Expected
    # to be ~0; a nonzero count is surfaced loudly so a players-table sync gap
    # cannot hide as a silent row shortfall. These are genuinely unfeaturisable
    # (no identity, hence no features), so we report rather than force them in.
    scope = "" if season is None else " WHERE ca.season = :scope_season"
    dropped = conn.execute(
        f"""SELECT ca.nba_api_id, ca.season, ca.award, ca.snapshot_date
            FROM candidate_admission ca
            LEFT JOIN players p ON p.nba_api_id = ca.nba_api_id
            WHERE p.player_id IS NULL{(' AND ca.season = :scope_season') if season is not None else ''}""",
        ({"scope_season": season} if season is not None else {}),
    ).fetchall()
    if dropped:
        log.warning("materialise dropped %d admitted row(s) with no players identity "
                    "(unfeaturisable): %s", len(dropped),
                    [(d["nba_api_id"], d["season"], d["award"]) for d in dropped[:10]])
    return {"rows": len(out), "team_unresolved": n_team_null,
            "admitted_dropped_no_identity": len(dropped),
            "seasons": "all" if season is None else season}


# ---------------------------------------------------------------------------
# STAGE 2: in-memory relative encodings + forward-error inputs.
# ---------------------------------------------------------------------------

def _percentile_rank(values: list[float]) -> list[float]:
    """PERCENT_RANK semantics over non-null values (NULLs passed as None get
    None back). Matches the filter's ranking primitive."""
    idx = [i for i, v in enumerate(values) if v is not None]
    out: list[float | None] = [None] * len(values)
    n = len(idx)
    if n <= 1:
        for i in idx:
            out[i] = 0.0
        return out
    order = sorted(idx, key=lambda i: values[i])
    # PERCENT_RANK = (rank-1)/(n-1) with ties sharing the lowest rank
    ranks: dict[int, int] = {}
    r = 0
    prev = None
    for pos, i in enumerate(order):
        if prev is None or values[i] != prev:
            r = pos
            prev = values[i]
        ranks[i] = r
    for i in idx:
        out[i] = ranks[i] / (n - 1)
    return out


def _group_relative(group_rows: list[dict], feature_cols: list[str]) -> None:
    """Wave 1: emit ONE standardised relative per feature within this group, in place.
    z (<fc>_z) for bounded rates/fractions/ratings; percentile (<fc>_pct) for heavy-tailed
    magnitude, chosen by _relative_kind. rank and delta_leader are dropped (rank is inverted
    percentile; delta_leader correlates with z). NULL-feature members excluded from each
    feature's pool; a feature absent from the whole group is skipped."""
    for fc in feature_cols:
        if not any(fc in r for r in group_rows):
            continue
        kind = _relative_kind(fc)
        vals = [r.get(fc) for r in group_rows]
        present = [v for v in vals if v is not None]
        if kind == "pct":
            pcts = _percentile_rank(vals)
            for r, pct in zip(group_rows, pcts):
                r[f"{fc}_pct"] = pct
        else:
            mean = sum(present) / len(present) if present else None
            var = (sum((v - mean) ** 2 for v in present) / len(present)
                   if present else None)
            std = math.sqrt(var) if var not in (None, 0) else None
            for r, v in zip(group_rows, vals):
                r[f"{fc}_z"] = ((v - mean) / std) if (v is not None and std) else None


def _season_deltas(group_rows: list[dict]) -> None:
    """Add <col>_d1 = <col> - prior_<col> for each delta-set column, in place.
    None when EITHER the current level or the prior-final level is missing: a
    NULL delta means 'no last year' (true rookie, missed last season, or the
    1996 data floor), which is a different signal from a zero delta ('exactly
    the same as last year'). Do NOT zero-fill; pass None so the tree branches on
    missing. conf_rank / projected_seed are ordinal (lower=better): the raw
    difference is kept, so a NEGATIVE _d1 is improvement; sign handled downstream.
    """
    for col in delta_feature_cols():
        pcol = f"prior_{col}"
        dcol = f"{col}_d1"
        for r in group_rows:
            cur = r.get(col)
            prior = r.get(pcol)
            r[dcol] = (cur - prior) if (cur is not None and prior is not None) else None


def _forward_error_inputs(group_rows: list[dict], season_wmax: float | None = None) -> None:
    """Add the forward-error-engine inputs the model layer (Phase 4) needs:
      - frac_season_elapsed: from week_index over the season's max week.
      - topk_entropy: normalised Shannon entropy of the top-k label-proxy share
        vector (uses current production proxy pra_pct as the share basis, since
        the true label is not available as-of). Governs F's variance.
    z-score standing per feature is already emitted by _group_relative (the _z
    columns), so β has its inputs.
    """
    # frac_season_elapsed: week_index relative to the SEASON max week_index.
    # preseason (week_index == -1) maps to 0.0; in-season weeks span (0, 1].
    wmax = season_wmax
    for r in group_rows:
        wi = r.get("week_index")
        if wi is None or not wmax or wmax <= 0:
            r["frac_season_elapsed"] = None
        elif wi < 0:
            r["frac_season_elapsed"] = 0.0
        else:
            r["frac_season_elapsed"] = wi / wmax
    # Closeness entropy, corrected construction. The legacy version used a
    # PERCENTILE basis (box_pra_std_pct) which is uniform-by-construction and so
    # saturated to ~0.956 for every race (verified degenerate: std 0.004). The
    # fix: build the share vector from z-scored LEVELS over the full group, so a
    # dominant candidate actually spikes and the entropy drops on settled races.
    # Two observed-stat bases computed here (PRA = production, STOCKS = defensive
    # disruption = steals+blocks). Score-entropy (the model-score basis) is NOT
    # here: it needs OOF scores absent at load time, joined separately if built.
    def _entropy_from_levels(levels: list[float]) -> float | None:
        # z-score, exp() to a positive share vector (softmax), normalised Shannon.
        # exp(z) keeps the spike: a +3z candidate dominates the share mass; a flat
        # field gives near-uniform shares and entropy near 1.
        vals = [x for x in levels if x is not None]
        if len(vals) < 2:
            return None
        mu = sum(vals) / len(vals)
        var = sum((x - mu) ** 2 for x in vals) / len(vals)
        sd = math.sqrt(var)
        if sd <= 0:
            return 0.0  # all identical => maximally settled is ill-defined; treat as 0 spread
        z = [(x - mu) / sd for x in vals]
        # clip z to avoid exp overflow on extreme outliers
        ez = [math.exp(max(-10.0, min(10.0, zi))) for zi in z]
        tot = sum(ez)
        ps = [e / tot for e in ez if e > 0]
        h = -sum(p * math.log(p) for p in ps)
        hmax = math.log(len(ps)) if len(ps) > 1 else None
        return (h / hmax) if hmax else 0.0

    pra_levels = [r.get("box_pra_std") for r in group_rows]
    stk_levels = []
    for r in group_rows:
        spg, bpg = r.get("box_spg_std"), r.get("box_bpg_std")
        stk_levels.append((spg + bpg) if (spg is not None and bpg is not None) else None)
    pra_ent = _entropy_from_levels(pra_levels)
    stk_ent = _entropy_from_levels(stk_levels)
    for r in group_rows:
        # keep topk_entropy name = PRA basis (back-compat with existing model);
        # add the two new bases as distinct features for the filter to choose.
        r["topk_entropy"] = pra_ent
        r["entropy_pra"] = pra_ent
        r["entropy_stocks"] = stk_ent
        fr = r.get("frac_season_elapsed")
        r["entropy_pra_x_frac"] = (pra_ent * fr) if (pra_ent is not None and fr is not None) else None
        r["entropy_stocks_x_frac"] = (stk_ent * fr) if (stk_ent is not None and fr is not None) else None


def load_design_matrix(conn, award: str, seasons: list[int] | None = None,
                       model_version: str | None = None,
                       pwin_key: str | None = None,
                       placebo_narrative: bool = False) -> list[dict]:
    q = "SELECT * FROM feature_stats_asof WHERE award = ?"
    args: list = [award]
    if seasons:
        q += f" AND season IN ({','.join('?' * len(seasons))})"
        args += seasons
    rows = [dict(r) for r in conn.execute(q, args).fetchall()]

    # merge score-entropy (materialised from the K=10 OOF pass; null if absent)
    try:
        _se = {(r["nba_api_id"], r["season"], r["snapshot_date"], r["award"]): r["entropy_score"]
               for r in conn.execute(
                   "SELECT nba_api_id, season, snapshot_date, award, entropy_score "
                   "FROM stg_entropy_score_asof WHERE award=?", [award]).fetchall()}
    except Exception:
        _se = {}
    for r in rows:
        r["entropy_score"] = _se.get((r["nba_api_id"], r["season"], r["snapshot_date"], r["award"]))

    # --- first-place-share label (experiment): in-memory from award_voting,
    # no re-materialise. Denominator is the SEASON-WIDE total first-place
    # ballots (counted once per player from the base table), so the share
    # sums to ~1 over the admitted set exactly like label_vote_share. A
    # season with no first-place data at all yields NULL for every row of
    # that season, which run_award/generate_oof drop to keep the arms matched.
    _fpv: dict[tuple, float] = {}
    _fptot: dict[int, float] = {}
    for _x in conn.execute(
            "SELECT season, player_id, first_place_votes FROM award_voting "
            "WHERE award = ?", [award]).fetchall():
        _xd = dict(_x)
        if _xd["first_place_votes"] is not None:
            _fpv[(_xd["player_id"], _xd["season"])] = float(_xd["first_place_votes"])
            _fptot[_xd["season"]] = _fptot.get(_xd["season"], 0.0) + float(_xd["first_place_votes"])
    for r in rows:
        _tot = _fptot.get(r["season"], 0.0)
        _votes = _fpv.get((r["player_id"], r["season"]), 0.0)
        r["label_first_place_votes"] = _votes if _tot else None
        r["label_first_place_share"] = (_votes / _tot) if _tot else None

    # group by (season, snapshot_date)
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["season"], r["snapshot_date"]), []).append(r)

    rel_cols = relative_feature_cols()

    season_wmax_map: dict = {}
    for r in rows:
        wi = r.get("week_index")
        if wi is not None and wi >= 0:
            s = r["season"]
            if wi > season_wmax_map.get(s, -1):
                season_wmax_map[s] = wi

    out: list[dict] = []
    for (season, snap), grp in groups.items():
        _season_deltas(grp)
        _group_relative(grp, rel_cols)
        _forward_error_inputs(grp, season_wmax_map.get(season))
        for _r in grp:
            _es, _fr = _r.get("entropy_score"), _r.get("frac_season_elapsed")
            _r["entropy_score_x_frac"] = (_es * _fr) if (_es is not None and _fr is not None) else None
        # positional normalisation: within-position z vs the as-of league pool.
        # Independent of the bridge; runs before it. Attaches r["position"] (also
        # usable as the LightGBM categorical) and r["<stat>_<frame>_posz"].
        attach_candidate_position(conn, grp, season)
        annotate_positional_z(conn, grp, season, snap, _POSZ_GP_FLOOR)
        gk = f"{award}|{int(season)}|{snap}"
        for r in grp:
            r["group_key"] = gk
        out.extend(grp)
    return out

def migrate_deltas(conn) -> dict:
    """Idempotent ALTER-in-place: add the prior_<col> columns to a populated
    feature_stats_asof without a full rebuild. PRAGMA-guards each ADD COLUMN
    (SQLite has no ADD COLUMN IF NOT EXISTS), in the nba_grid_deadstretch.py
    style. After this, run --materialise to backfill the prior-final levels.
    INTEGER for the two ordinal team columns, REAL for everything else."""
    existing = {r[1] for r in conn.execute(
        "PRAGMA table_info(feature_stats_asof)").fetchall()}
    int_cols = {"prior_team_conf_rank", "prior_team_projected_seed"}
    added = []
    for col in prior_level_cols():
        if col in existing:
            continue
        coltype = "INTEGER" if col in int_cols else "REAL"
        conn.execute(f"ALTER TABLE feature_stats_asof ADD COLUMN {col} {coltype}")
        added.append(col)
    conn.commit()
    return {"added": added, "already_present": len(prior_level_cols()) - len(added)}


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Feature loader: materialise feature_stats_asof.")
    ap.add_argument("--db", type=Path, default=Path("data/awards.db"))
    ap.add_argument("--materialise", action="store_true")
    ap.add_argument("--migrate-deltas", action="store_true",
                    help="idempotent ALTER ADD COLUMN for prior_<col> delta levels")
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--check-honours", action="store_true",
                    help="print distinct honour values to confirm the enum constants")
    args = ap.parse_args(argv)
    conn = connect(args.db)
    if args.check_honours:
        for r in conn.execute("SELECT DISTINCT honour FROM stg_player_honours"):
            print(r[0])
        conn.close()
        return 0
    if args.migrate_deltas:
        summary = migrate_deltas(conn)
        log.info("MIGRATE-DELTAS SUMMARY: %s", summary)
    if args.materialise:
        summary = materialise(conn, season=args.season)
        log.info("MATERIALISE SUMMARY: %s", summary)
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
