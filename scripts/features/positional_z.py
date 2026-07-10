from __future__ import annotations

import logging
import math

log = logging.getLogger("positional_z")

_POSZ_BASE_BOX = ("apg", "spg", "bpg", "rpg")
_BOX_FRAMES = ("std", "ema", "l10_vs_std")
_POSZ_ADV = ("stl", "blk", "dreb", "off_rating", "def_rating", "usg_pct")
_POSZ_EXT = ("ast_pct", "pct_stl", "pct_blk", "reb_pct", "oreb_pct", "dreb_pct",
             "poss", "time_of_poss", "touches", "pct_pts_paint", "pct_pts_3pt",
             "pct_pts_mr")

MIN_POOL = 5
POOL_GP_COL = "gp_played_asof"
_POSITIONS = ("guard", "wing", "big")

def pos_onehot_cols() -> list[str]:
    return [f"pos_is_{p.lower()}" for p in _POSITIONS]


def gp_floor() -> float | None:
    try:
        from scripts.features import nba_candidate_filter as F
    except ImportError:
        try:
            import nba_candidate_filter as F
        except ImportError:
            return None
    for name in ("MIN_GP", "GP_FLOOR", "MIN_GAMES_PLAYED", "MIN_GP_ASOF",
                 "FINISHER_MIN_GP", "MIN_GP_FLOOR"):
        if not hasattr(F, name):
            continue
        val = getattr(F, name)
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return float(val)
        if isinstance(val, dict):
            # per-award floor; the positional pool is award-agnostic, so collapse
            # to the most permissive (min) floor to keep the pool as large as
            # possible. Only numeric dict values count.
            nums = [float(v) for v in val.values()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)]
            if nums:
                return min(nums)
    return None

def posz_target_cols() -> list[str]:
    box = [f"box_{b}_{f}" for b in _POSZ_BASE_BOX for f in _BOX_FRAMES]
    adv = [f"adv_{b}_std" for b in _POSZ_ADV]
    ext = [f"ext_{b}_std" for b in _POSZ_EXT]
    return box + adv + ext


def posz_output_cols() -> list[str]:
    return [f"{c}_posz" for c in posz_target_cols()]


def _family(matrix_col: str) -> str:
    return matrix_col.split("_", 1)[0]


def _bare(matrix_col: str) -> str:
    rest = matrix_col.split("_", 1)[1]
    if matrix_col.startswith(("adv_", "ext_")) and rest.endswith("_std"):
        return rest[:-4]
    return rest


def _mean_std(vals: list[float]) -> tuple[float | None, float | None]:
    if not vals:
        return None, None
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    std = math.sqrt(var) if var not in (None, 0) else None
    return mean, std


_gp_floor_warned = False


def build_position_pool(conn, season: int, snap: str, gp_floor_val: float | None):
    global _gp_floor_warned
    want = {"box": set(), "adv": set(), "ext": set()}
    for tcol in posz_target_cols():
        want[_family(tcol)].add(_bare(tcol))

    box_sel = ", ".join(f"b.{c} AS box__{c}" for c in sorted(want["box"]))
    adv_sel = ", ".join(f"a.{c} AS adv__{c}" for c in sorted(want["adv"]))
    ext_sel = ", ".join(f"e.{c} AS ext__{c}" for c in sorted(want["ext"]))
    sel = ", ".join(s for s in (box_sel, adv_sel, ext_sel) if s)

    gp_clause = ""
    if gp_floor_val is not None:
        gp_clause = f"AND b.{POOL_GP_COL} >= {float(gp_floor_val)}"

    sql = f"""
        SELECT ppm.position AS pos, {sel}
        FROM stg_nba_box_asof b
        JOIN player_position_map ppm ON ppm.nba_api_id = b.nba_api_id
        LEFT JOIN stg_nba_player_advanced_asof a
          ON a.nba_api_id = b.nba_api_id AND a.season = b.season
             AND a.snapshot_date = b.snapshot_date
        LEFT JOIN stg_nba_player_asof_ext e
          ON e.nba_api_id = b.nba_api_id AND e.season = b.season
             AND e.snapshot_date = b.snapshot_date
        WHERE b.season = ? AND b.snapshot_date = ? {gp_clause}
    """
    try:
        fetched = conn.execute(sql, (season, snap)).fetchall()
    except Exception as exc:
        if gp_clause and not _gp_floor_warned:
            log.warning("position pool query failed with GP floor (%s); retrying "
                        "without floor; verify POOL_GP_COL; err=%s", POOL_GP_COL, exc)
            _gp_floor_warned = True
            return build_position_pool(conn, season, snap, None)
        log.error("position pool query failed: %s", exc)
        return {}

    out: dict[str, dict[str, list[float]]] = {}
    for r in fetched:
        pos = r["pos"]
        if pos is None:
            continue
        bucket = out.setdefault(pos, {})
        for key in r.keys():
            if key == "pos":
                continue
            v = r[key]
            if v is None:
                continue
            _fam, bare = key.split("__", 1)
            bucket.setdefault(bare, []).append(float(v))
    return out


def annotate_positional_z(conn, group_rows: list[dict], season: int, snap: str,
                          gp_floor_val: float | None) -> None:
    targets = posz_target_cols()
    pool = build_position_pool(conn, season, snap, gp_floor_val)

    stats: dict[tuple[str, str], tuple[float | None, float | None, int]] = {}
    for pos, cols in pool.items():
        for bare, vals in cols.items():
            mean, std = _mean_std(vals)
            stats[(pos, bare)] = (mean, std, len(vals))

    for r in group_rows:
        pos = r.get("position")
        for tcol in targets:
            out_col = f"{tcol}_posz"
            x = r.get(tcol)
            if x is None or pos is None:
                r[out_col] = None
                continue
            mean, std, n = stats.get((pos, _bare(tcol)), (None, None, 0))
            if n < MIN_POOL or mean is None or std is None:
                r[out_col] = None
            else:
                r[out_col] = (float(x) - mean) / std


def attach_candidate_position(conn, group_rows: list[dict], season: int) -> int:
    ids = sorted({r["nba_api_id"] for r in group_rows
                  if r.get("nba_api_id") is not None})
    if not ids:
        return len(group_rows)
    placeholders = ",".join("?" * len(ids))
    sql = f"""
        SELECT nba_api_id, position
        FROM player_position_map
        WHERE nba_api_id IN ({placeholders})
    """
    pos_by_id: dict = {}
    try:
        for r in conn.execute(sql, tuple(ids)).fetchall():
            pos_by_id[r["nba_api_id"]] = r["position"]
    except Exception as exc:
        log.error("attach_candidate_position failed; verify player_position_map "
                  "exists (run schema/build_position_map.sql): %s", exc)
    missing = 0
    for r in group_rows:
        p = pos_by_id.get(r.get("nba_api_id"))
        r["position"] = p
        for pos in _POSITIONS:
            r[f"pos_is_{pos.lower()}"] = (1 if p == pos else 0) if p is not None else None
        if p is None:
            missing += 1
#    if missing:
#        log.warning("season %d: %d candidate rows without position", season, missing)
    return missing
