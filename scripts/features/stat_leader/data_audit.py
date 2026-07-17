"""Full player-data audit of awards.db, classified by domain in plain English.

For every user table: domain classification, row count, season span. For the
feature-candidate tables (the ones a leading indicator could draw on) it goes
column by column: fill rate (fraction non-null) and FIRST populated season, the
hard floor on any feature built from that column.

Read-only. Run from repo root:
  uv run python3 -m scripts.features.stat_leader.data_audit
  uv run python3 -m scripts.features.stat_leader.data_audit --deep-all   # cols for every table
"""
from __future__ import annotations

import argparse
import sys

try:
    from scripts.common.db import connect
except ImportError:  # pragma: no cover
    from db import connect  # type: ignore

DOMAINS = [
    ("box counting (per-game)", ["player_game_logs"]),
    ("advanced / efficiency", ["advanced", "entropy"]),
    ("tracking (touches, passing, drives)", ["asof_ext"]),
    ("hustle (charges, loose balls)", ["hustle"]),
    ("defence (defended shots, rim)", ["defend"]),
    ("availability / minutes / starts", ["availability", "box_asof", "game_starts",
                                         "boxstart", "starts_asof"]),
    ("team / pace / ratings", ["team_game", "qualifier", "team_records",
                               "prior_ratings", "team_res", "standings"]),
    ("stat substrate (derived)", ["stat_rate_counts", "feature_stats", "feature_selection"]),
    ("market / price", ["pm_", "corpus_"]),
    ("voting / honours (labels)", ["award_voting", "player_honours", "dpoy",
                                   "candidate_admission", "honours"]),
    ("identity", ["players", "player_position_map", "name_resolution",
                  "bref_players", "aliases"]),
    ("injuries", ["injuries"]),
    ("schedule / grid", ["snapshot_grid", "schedule"]),
    ("model artefacts / progress", ["model_predictions", "pull_progress",
                                    "prediction", "progress"]),
]
FEATURE_TABLES = ["stg_nba_player_game_logs", "stg_nba_player_advanced_asof",
                  "stg_nba_player_asof_ext", "stg_nba_hustle_asof",
                  "stg_nba_defend_asof", "stg_nba_box_asof",
                  "stg_nba_availability_asof"]


def _classify(name):
    for dom, keys in DOMAINS:
        if any(k in name for k in keys):
            return dom
    return "other / unclassified"


def _cols(conn, t):
    return [r[1] for r in conn.execute(f'PRAGMA table_info("{t}")')]


def _has(conn, t, col):
    return col in _cols(conn, t)


def _span(conn, t):
    if not _has(conn, t, "season"):
        return None
    r = conn.execute(f'SELECT MIN(season), MAX(season), COUNT(DISTINCT season) FROM "{t}"').fetchone()
    return r if r and r[0] is not None else None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/awards.db")
    ap.add_argument("--deep-all", action="store_true")
    a = ap.parse_args(argv)
    conn = connect(a.db)

    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
    grouped = {}
    for t in tables:
        grouped.setdefault(_classify(t), []).append(t)

    print("=" * 74)
    print(f"DATA AUDIT  {a.db}   ({len(tables)} tables)")
    print("=" * 74)
    for dom, _ in DOMAINS + [("other / unclassified", [])]:
        ts = grouped.get(dom)
        if not ts:
            continue
        print(f"\n### {dom}")
        for t in sorted(set(ts)):
            try:
                n = conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
            except Exception:
                n = -1
            sp = _span(conn, t)
            sptxt = f"seasons {sp[0]}-{sp[1]} ({sp[2]})" if sp else "no season col"
            print(f"  {t:<34} {n:>10,} rows   {sptxt}")

    deep = tables if a.deep_all else [t for t in FEATURE_TABLES if t in tables]
    print("\n" + "=" * 74)
    print("COLUMN-LEVEL: fill rate and first populated season (the feature floor)")
    print("=" * 74)
    for t in deep:
        cols = _cols(conn, t)
        try:
            n = conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
        except Exception:
            continue
        if not n:
            continue
        has_season = _has(conn, t, "season")
        print(f"\n### {t}   ({n:,} rows)")
        print(f"  {'column':<26} {'fill%':>6}  {'first_season':>12}")
        for c in cols:
            if c in ("season", "snapshot_date", "nba_api_id", "player_id",
                     "team_id", "game_id"):
                continue
            try:
                nn = conn.execute(f'SELECT COUNT("{c}") FROM "{t}"').fetchone()[0]
            except Exception:
                continue
            fill = 100.0 * nn / n
            fs = ""
            if has_season and nn:
                try:
                    fs = conn.execute(
                        f'SELECT MIN(season) FROM "{t}" WHERE "{c}" IS NOT NULL').fetchone()[0]
                except Exception:
                    fs = ""
            flag = "" if fill > 95 else ("  <-- partial" if fill > 5 else "  <-- empty/near-empty")
            print(f"  {c:<26} {fill:>5.1f}  {str(fs):>12}{flag}")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
