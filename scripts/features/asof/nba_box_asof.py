"""As-of box-derived feature reconstruction -> stg_nba_box_asof.

Single deliverable of the box-reconstruction execution chat. Reads the already-
built snapshot_grid (the shared as-of clock) and stg_nba_player_game_logs, and
writes one row per (nba_api_id, season, snapshot_date) carrying that player's
box-derived state computed over ONLY the games on or before the snapshot date.

This module does NOT:
  - allocate player_id, touch players, or do any identity work;
  - add an award column or any award logic (rows are award-agnostic);
  - compute off/def/net rating, USG%, PIE, pace (those land as-of in
    stg_nba_player_advanced_asof / _ext; we must not recompute/approximate them);
  - add team context or relative encodings (loader-on-load concerns).

SEASON CONVENTION: season is the STARTING year, inherited verbatim from
stg_nba_player_game_logs.season. No conversion happens here.

THE TWO COUNTS (do not conflate; this is the GP-match lesson restated):
  gp_asof        = appearances through the snapshot, counted as
                   (minutes IS NOT NULL). Sanity anchor + loader use.
  gp_played_asof = games with minutes > 0 through the snapshot. This is the
                   denominator behind every per-game rate, and the game count
                   that defines the last-N rolling windows. DNPs / 0-minute
                   appearances must never dilute PPG/PRA, so they are excluded
                   from every rate and window.

PERCENTAGES are computed from SUMMED makes/attempts over each window, never as
the mean of per-game percentages (averaging ratios is wrong). A window with zero
attempts yields NULL for that percentage, not 0.

EMA (10-game half-life): an exponentially weighted aggregate over the ordered
minutes>0 game sequence as of the snapshot, most-recent game weight 1.0 and
weight halving every 10 games back (decay = 0.5 ** (1/10) per game step). For
per-game RATE stats the EMA is the weighted mean of the per-game value. For
PERCENTAGES the EMA is weighted_sum(makes) / weighted_sum(attempts) using the
SAME per-game weights, because an EMA of a ratio-of-sums is otherwise ill-
defined; this keeps percentages consistent with the summed-makes/attempts rule.

Run:  uv run python -m scripts.features.asof.nba_box_asof
      uv run python -m scripts.features.asof.nba_box_asof --db data/awards.db
      uv run python -m scripts.features.asof.nba_box_asof --season 2024   # one season only
"""

from __future__ import annotations

import argparse
import logging
import sys

try:  # canonical shared helper in the clean tree
    from scripts.common.db import connect, upsert, utc_now
except ImportError:  # pragma: no cover - db.py may export utcnow_iso instead
    from scripts.common.db import connect, upsert, utcnow_iso as utc_now

log = logging.getLogger("nba_box_asof")

# EMA half-life in games. decay per one-game step so weight halves every 10 games.
EMA_HALFLIFE_GAMES = 10
_EMA_DECAY = 0.5 ** (1.0 / EMA_HALFLIFE_GAMES)

# Rolling window sizes. l10/l30 are primary (contract); l5/l20 are selector-
# adjudicated extras emitted so the L1 stage can decide if they earn their place.
_PRIMARY_WINDOWS = (10, 30)
_EXTRA_WINDOWS = (5, 20)

# Per-game rate stats: (output_prefix, game-log column). PRA handled separately.
_RATE_COLS = (
    ("ppg", "points"),
    ("rpg", "rebounds"),
    ("apg", "assists"),
    ("spg", "steals"),
    ("bpg", "blocks"),
    ("mpg", "minutes"),
)

# Shooting percentages: (output_prefix, numerator-builder, denominator-builder)
# over a window of games, each builder takes summed components. Defined in
# _pct_from_sums; listed here only for column/name generation.
_PCT_PREFIXES = ("fg_pct", "fg3_pct", "ft_pct", "ts_pct", "efg_pct")


# -----------------------------------------------------------------------------
# Column-name generation (single source of truth, matches the DDL exactly)
# -----------------------------------------------------------------------------

def _stat_prefixes() -> list[str]:
    return [p for p, _ in _RATE_COLS] + ["pra"] + list(_PCT_PREFIXES)


def feature_columns() -> list[str]:
    """All feature column names in DDL order, for the row dict + smoke test."""
    cols: list[str] = []
    for pfx in _stat_prefixes():
        cols += [
            f"{pfx}_std", f"{pfx}_l10", f"{pfx}_l30", f"{pfx}_ema",
            f"{pfx}_l10_vs_l30", f"{pfx}_l10_vs_std",
            f"{pfx}_l5", f"{pfx}_l20",
        ]
    return cols


# -----------------------------------------------------------------------------
# Per-game helpers
# -----------------------------------------------------------------------------

def _num(v) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f  # NaN guard


def _safe_div(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or den == 0:
        return None
    return num / den


def _pct_from_sums(prefix: str, s: dict) -> float | None:
    """Compute one shooting percentage from summed components in dict s.

    s carries summed window totals: pts, fga, fgm, fg3a, fg3m, fta, ftm.
    Returns None when the denominator is zero/absent (never 0.0 by default).
    """
    if prefix == "fg_pct":
        return _safe_div(s["fgm"], s["fga"])
    if prefix == "fg3_pct":
        return _safe_div(s["fg3m"], s["fg3a"])
    if prefix == "ft_pct":
        return _safe_div(s["ftm"], s["fta"])
    if prefix == "ts_pct":
        denom = 2.0 * (s["fga"] + 0.44 * s["fta"])
        return _safe_div(s["pts"], denom) if denom > 0 else None
    if prefix == "efg_pct":
        return _safe_div(s["fgm"] + 0.5 * s["fg3m"], s["fga"])
    raise ValueError(f"unknown pct prefix {prefix!r}")


# -----------------------------------------------------------------------------
# Window aggregation over a list of per-game dicts (already minutes>0, ordered
# OLDEST..NEWEST so slicing the last-N is a tail slice)
# -----------------------------------------------------------------------------

def _sum_components(games: list[dict]) -> dict:
    """Sum the shooting components needed for percentages over `games`."""
    keys = ("points", "fga", "fgm", "fg3a", "fg3m", "fta", "ftm")
    tot = {k: 0.0 for k in keys}
    for g in games:
        for k in keys:
            v = g[k]
            if v is not None:
                tot[k] += v
    # remap to the short names _pct_from_sums expects
    return {
        "pts": tot["points"], "fga": tot["fga"], "fgm": tot["fgm"],
        "fg3a": tot["fg3a"], "fg3m": tot["fg3m"], "fta": tot["fta"],
        "ftm": tot["ftm"],
    }


def _rate_mean(games: list[dict], col: str) -> float | None:
    """Simple per-game mean of a rate column over `games` (denominator = len)."""
    if not games:
        return None
    vals = [g[col] for g in games if g[col] is not None]
    if not vals:
        return None
    return sum(vals) / len(games)


def _pra_mean(games: list[dict]) -> float | None:
    if not games:
        return None
    total = 0.0
    n = 0
    for g in games:
        p, r, a = g["points"], g["rebounds"], g["assists"]
        if p is None and r is None and a is None:
            continue
        total += (p or 0.0) + (r or 0.0) + (a or 0.0)
        n += 1
    return total / len(games) if n else None


def _window_rates(games: list[dict]) -> dict[str, float | None]:
    """All rate stats (ppg..mpg + pra) as simple per-game means over `games`."""
    out: dict[str, float | None] = {}
    for pfx, col in _RATE_COLS:
        out[pfx] = _rate_mean(games, col)
    out["pra"] = _pra_mean(games)
    return out


def _window_pcts(games: list[dict]) -> dict[str, float | None]:
    """All shooting percentages over `games`, from summed components."""
    s = _sum_components(games)
    return {pfx: _pct_from_sums(pfx, s) for pfx in _PCT_PREFIXES}


# -----------------------------------------------------------------------------
# EMA over the ordered minutes>0 sequence (most-recent highest weight)
# -----------------------------------------------------------------------------

def _ema_weights(n: int) -> list[float]:
    """Weights for n games ordered OLDEST..NEWEST. Newest weight 1.0, halving
    every EMA_HALFLIFE_GAMES going back. weights[i] = decay**(games-from-newest).
    """
    # index 0 is oldest, index n-1 is newest. distance from newest = (n-1 - i).
    return [_EMA_DECAY ** (n - 1 - i) for i in range(n)]


def _ema_rates(games: list[dict]) -> dict[str, float | None]:
    """EMA of each rate stat: weighted mean of per-game value over all games."""
    out: dict[str, float | None] = {}
    if not games:
        for pfx, _ in _RATE_COLS:
            out[pfx] = None
        out["pra"] = None
        return out
    w = _ema_weights(len(games))
    for pfx, col in _RATE_COLS:
        num = 0.0
        den = 0.0
        for wi, g in zip(w, games):
            v = g[col]
            if v is None:
                continue
            num += wi * v
            den += wi
        out[pfx] = (num / den) if den > 0 else None
    # PRA EMA: weight the per-game PRA total
    num = den = 0.0
    for wi, g in zip(w, games):
        p, r, a = g["points"], g["rebounds"], g["assists"]
        if p is None and r is None and a is None:
            continue
        num += wi * ((p or 0.0) + (r or 0.0) + (a or 0.0))
        den += wi
    out["pra"] = (num / den) if den > 0 else None
    return out


def _ema_pcts(games: list[dict]) -> dict[str, float | None]:
    """EMA of shooting percentages: weighted_sum(makes)/weighted_sum(attempts)
    using the SAME per-game weights, consistent with the summed-ratio rule."""
    if not games:
        return {pfx: None for pfx in _PCT_PREFIXES}
    w = _ema_weights(len(games))
    keys = ("points", "fga", "fgm", "fg3a", "fg3m", "fta", "ftm")
    wsum = {k: 0.0 for k in keys}
    for wi, g in zip(w, games):
        for k in keys:
            v = g[k]
            if v is not None:
                wsum[k] += wi * v
    s = {
        "pts": wsum["points"], "fga": wsum["fga"], "fgm": wsum["fgm"],
        "fg3a": wsum["fg3a"], "fg3m": wsum["fg3m"], "fta": wsum["fta"],
        "ftm": wsum["ftm"],
    }
    return {pfx: _pct_from_sums(pfx, s) for pfx in _PCT_PREFIXES}


# -----------------------------------------------------------------------------
# Build one feature row from a player's ordered minutes>0 games-through-snapshot
# -----------------------------------------------------------------------------

def _delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return a - b


def build_feature_values(played_games: list[dict]) -> dict[str, float | None]:
    """Compute every feature column from the ordered (oldest..newest) list of a
    player's minutes>0 games as of a snapshot. Caller guarantees the list is the
    correct as-of slice; this function applies no date logic.
    """
    std_rates = _window_rates(played_games)
    std_pcts = _window_pcts(played_games)
    ema_rates = _ema_rates(played_games)
    ema_pcts = _ema_pcts(played_games)

    # tail slices for each window size
    def tail(n: int) -> list[dict]:
        return played_games[-n:] if n < len(played_games) else played_games

    win_rates: dict[int, dict] = {}
    win_pcts: dict[int, dict] = {}
    for n in (*_PRIMARY_WINDOWS, *_EXTRA_WINDOWS):
        g = tail(n)
        win_rates[n] = _window_rates(g)
        win_pcts[n] = _window_pcts(g)

    vals: dict[str, float | None] = {}
    for pfx in _stat_prefixes():
        is_pct = pfx in _PCT_PREFIXES
        std = (std_pcts if is_pct else std_rates)[pfx]
        ema = (ema_pcts if is_pct else ema_rates)[pfx]
        l10 = (win_pcts if is_pct else win_rates)[10][pfx]
        l30 = (win_pcts if is_pct else win_rates)[30][pfx]
        l5 = (win_pcts if is_pct else win_rates)[5][pfx]
        l20 = (win_pcts if is_pct else win_rates)[20][pfx]
        vals[f"{pfx}_std"] = std
        vals[f"{pfx}_l10"] = l10
        vals[f"{pfx}_l30"] = l30
        vals[f"{pfx}_ema"] = ema
        vals[f"{pfx}_l10_vs_l30"] = _delta(l10, l30)
        vals[f"{pfx}_l10_vs_std"] = _delta(l10, std)
        vals[f"{pfx}_l5"] = l5
        vals[f"{pfx}_l20"] = l20
    return vals


# -----------------------------------------------------------------------------
# DB reads
# -----------------------------------------------------------------------------

def load_grid(conn, season: int | None) -> list[tuple[int, str]]:
    """Return [(season, snapshot_date), ...] from snapshot_grid, ordered.

    The grid is the shared as-of clock; we read it, never regenerate it.
    """
    if season is None:
        cur = conn.execute(
            "SELECT season, snapshot_date FROM snapshot_grid "
            "ORDER BY season, snapshot_date"
        )
    else:
        cur = conn.execute(
            "SELECT season, snapshot_date FROM snapshot_grid WHERE season = ? "
            "ORDER BY snapshot_date",
            (season,),
        )
    return [(r["season"], r["snapshot_date"]) for r in cur]


_GAMELOG_COLS = (
    "nba_api_id", "game_date", "season", "minutes", "points", "rebounds",
    "assists", "steals", "blocks", "fga", "fgm", "fg3a", "fg3m", "fta", "ftm",
)


def load_season_games(conn, season: int) -> dict[int, list[dict]]:
    """Load one season's game logs grouped by player, ordered oldest..newest.

    Returns {nba_api_id: [game_dict, ...]}. Each game_dict carries game_date,
    minutes, and the box components. Both DNP (minutes NULL) and 0-minute rows
    are kept here; the as-of slicer applies the minutes>0 filter per window so
    the two counts can both be computed from one ordered list.
    """
    cur = conn.execute(
        f"SELECT {', '.join(_GAMELOG_COLS)} FROM stg_nba_player_game_logs "
        "WHERE season = ? ORDER BY nba_api_id, game_date, game_id",
        (season,),
    )
    by_player: dict[int, list[dict]] = {}
    for r in cur:
        g = {k: r[k] for k in _GAMELOG_COLS}
        # numeric coercion (NULL stays None; NaN -> None)
        for k in ("minutes", "points", "rebounds", "assists", "steals", "blocks",
                  "fga", "fgm", "fg3a", "fg3m", "fta", "ftm"):
            g[k] = _num(g[k])
        by_player.setdefault(r["nba_api_id"], []).append(g)
    return by_player


# -----------------------------------------------------------------------------
# Core reconstruction for one season
# -----------------------------------------------------------------------------

def reconstruct_season(conn, season: int, grid_dates: list[str],
                       pulled_at: str) -> list[dict]:
    """Build all stg_nba_box_asof rows for one season across its grid dates.

    For each player and each snapshot_date, slice the player's games to
    game_date <= snapshot_date, split into appearances (minutes not NULL) and
    played (minutes > 0), and emit a row only if the player has >= 1 played game
    as of that date (per the contract's appearance gate). The preseason snapshot
    naturally emits no rows for current-season features (no games yet), which is
    the correct NULL-by-construction behaviour.
    """
    by_player = load_season_games(conn, season)
    rows: list[dict] = []

    for pid, games in by_player.items():
        # games are ordered oldest..newest already.
        for snap in grid_dates:
            # as-of slice: all rows with game_date <= snap. LEAK GATE.
            through = [g for g in games if g["game_date"] is not None
                       and g["game_date"] <= snap]
            if not through:
                continue
            appearances = [g for g in through if g["minutes"] is not None]
            played = [g for g in through if g["minutes"] is not None
                      and g["minutes"] > 0]
            # appearance gate: at least one minutes>0 game on/before snapshot.
            if not played:
                continue

            # ASSERTION (leak discipline): no game in the played slice may post-
            # date the snapshot. Cheap, catches any ordering/filter regression.
            assert all(g["game_date"] <= snap for g in played), (
                f"leak: game after snapshot {snap} for player {pid}"
            )

            vals = build_feature_values(played)
            row = {
                "nba_api_id": pid,
                "season": season,
                "snapshot_date": snap,
                "gp_asof": len(appearances),
                "gp_played_asof": len(played),
                **vals,
                "pulled_at": pulled_at,
            }
            rows.append(row)
    return rows


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------

def build(db_path: str, season: int | None = None, batch: int = 5000) -> dict:
    """Build stg_nba_box_asof for all grid seasons (or one), idempotently.

    Upserts in batches via the shared helper. Returns a summary dict.
    """
    pulled_at = utc_now()
    conn = connect(db_path)
    try:
        grid = load_grid(conn, season)
        if not grid:
            return {"seasons": [], "rows_written": 0,
                    "note": "snapshot_grid empty for requested scope"}

        # group grid dates by season
        dates_by_season: dict[int, list[str]] = {}
        for s, d in grid:
            dates_by_season.setdefault(s, []).append(d)

        total = 0
        seasons_done: list[int] = []
        for s in sorted(dates_by_season):
            rows = reconstruct_season(conn, s, dates_by_season[s], pulled_at)
            for i in range(0, len(rows), batch):
                upsert(conn, "stg_nba_box_asof", rows[i:i + batch],
                       ["nba_api_id", "season", "snapshot_date"])
            total += len(rows)
            seasons_done.append(s)
            log.info("season %d: %d box-asof rows", s, len(rows))

        return {"seasons": seasons_done, "rows_written": total}
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(
        description="Reconstruct as-of box features -> stg_nba_box_asof"
    )
    p.add_argument("--db", default="data/awards.db")
    p.add_argument("--season", type=int, default=None,
                   help="Build a single STARTING-year season only (default: all).")
    args = p.parse_args(argv)

    summary = build(args.db, args.season)
    log.info("done: %d rows across %d season(s)",
             summary["rows_written"], len(summary["seasons"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
