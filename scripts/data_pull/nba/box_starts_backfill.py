"""Per-game started-flag backfill for the NBA Awards Trader.

Sources the per-game starter/bench split that no season-wide nba_api endpoint
exposes, by calling the traditional box score once per game and reading the
starting ``position`` (non-empty for the five starters, empty off the bench).

Uses ``BoxScoreTraditionalV3`` (V2 stopped publishing as of the 2025-26 season).
V3 exposes ``personId`` / ``position``; the row builder tolerates the older
``PLAYER_ID`` / ``START_POSITION`` naming too, so a fallback source stays drop-in.

Writes ONLY the nba_api-scoped staging table ``stg_nba_game_starts`` keyed by
(nba_api_id, game_id); never touches canonical ``player_game_logs`` and never
allocates ``player_id`` (resolution's job), matching ``nba_api_pull``'s
contract. The game universe is the distinct (game_id, season) set already in
``stg_nba_player_game_logs`` over the requested range, so no game outside the
log layer's coverage is ever pulled. Resume is a per-game ledger
(``stg_nba_boxstart_progress``), stamped only after a game's rows are durably
upserted, so a mid-run stop re-attempts only unfinished games and a re-run is a
no-op. Newest-first, so a stop leaves a contiguous recent block.

Run:
  caffeinate -i uv run python -m scripts.data_pull.nba.box_starts_backfill
  uv run python -m scripts.data_pull.nba.box_starts_backfill --rate-only
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from tenacity import retry, stop_after_attempt, wait_exponential

try:
    from scripts.common.db import connect, upsert, utcnow_iso
except ImportError:  # pragma: no cover
    from db import connect, upsert, utcnow_iso  # type: ignore

log = logging.getLogger("box_starts_backfill")

DEFAULT_DB_PATH = Path("data/awards.db")
GAME_LOG_FLOOR = 1996


def _ensure_schema(conn) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS stg_nba_game_starts ("
        "nba_api_id INTEGER NOT NULL, game_id TEXT NOT NULL, season INTEGER, "
        "started INTEGER, pulled_at TEXT, PRIMARY KEY (nba_api_id, game_id))"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS stg_nba_boxstart_progress ("
        "game_id TEXT PRIMARY KEY, season INTEGER, n_rows INTEGER, completed_at TEXT)"
    )
    conn.commit()


def games_to_pull(conn, floor: int, current: int) -> list[tuple[str, int]]:
    """Distinct (game_id, season) from the log staging, newest-first, minus done."""
    done = {r[0] for r in conn.execute("SELECT game_id FROM stg_nba_boxstart_progress")}
    cur = conn.execute(
        "SELECT DISTINCT game_id, season FROM stg_nba_player_game_logs "
        "WHERE season BETWEEN ? AND ? ORDER BY season DESC, game_id DESC",
        (floor, current),
    )
    return [(gid, season) for gid, season in cur if gid not in done]


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, min=2, max=30), reraise=True)
def _box_records(game_id: str) -> list[dict]:
    from nba_api.stats.endpoints import boxscoretraditionalv3
    ep = boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=game_id)
    try:
        return ep.player_stats.get_data_frame().to_dict("records")
    except AttributeError:  # pragma: no cover - accessor shape guard
        return ep.get_data_frames()[0].to_dict("records")


def _started(val) -> int:
    return 1 if val is not None and str(val).strip() != "" else 0


def _pull_one_game(conn, game_id: str, season: int, pulled_at: str) -> int:
    records = _box_records(game_id)
    rows = []
    for r in records:
        pid = r.get("personId", r.get("PLAYER_ID"))
        if pid is None:
            continue
        pos = r.get("position", r.get("START_POSITION"))
        rows.append({
            "nba_api_id": int(pid),
            "game_id": str(game_id),
            "season": season,
            "started": _started(pos),
            "pulled_at": pulled_at,
        })
    if rows:
        upsert(conn, "stg_nba_game_starts", rows, ["nba_api_id", "game_id"])
    upsert(conn, "stg_nba_boxstart_progress",
           [{"game_id": str(game_id), "season": season, "n_rows": len(rows),
             "completed_at": utcnow_iso()}], ["game_id"])
    conn.commit()
    return len(rows)


def run(db_path: Path, floor: int, current: int, sleep_s: float) -> dict:
    conn = connect(db_path)
    try:
        _ensure_schema(conn)
        todo = games_to_pull(conn, floor, current)
        log.info("games to pull: %d (floor=%d current=%d)", len(todo), floor, current)

        try:
            from tqdm import tqdm
            iterator = tqdm(todo, desc="box_starts", unit="game")
        except ImportError:  # pragma: no cover
            iterator = todo

        summary = {"games": 0, "rows": 0, "failures": []}
        for game_id, season in iterator:
            try:
                n = _pull_one_game(conn, game_id, season, utcnow_iso())
            except Exception as exc:  # noqa: BLE001
                log.error("game %s (season %d) failed: %s", game_id, season, exc)
                summary["failures"].append((game_id, str(exc)))
                continue
            summary["games"] += 1
            summary["rows"] += n
            if hasattr(iterator, "set_postfix"):
                iterator.set_postfix(rows=summary["rows"], fails=len(summary["failures"]))
            time.sleep(sleep_s)
        return summary
    finally:
        conn.close()


def _print_started_rate(db_path: Path) -> None:
    """Per-season started-rate sanity check. An all-zero early season => bad source."""
    conn = connect(db_path)
    try:
        cur = conn.execute(
            "SELECT season, COUNT(*), SUM(started), ROUND(AVG(started), 3) "
            "FROM stg_nba_game_starts GROUP BY season ORDER BY season")
        print("\nseason | rows | started | started_rate")
        for season, n, s, rate in cur:
            print(f"  {season} | {n} | {s} | {rate}")
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Per-game started-flag box-score backfill -> stg_nba_game_starts")
    p.add_argument("--db", default=str(DEFAULT_DB_PATH))
    p.add_argument("--floor", type=int, default=GAME_LOG_FLOOR)
    p.add_argument("--current", type=int, default=2025)
    p.add_argument("--sleep", type=float, default=0.6)
    p.add_argument("--rate-only", action="store_true")
    args = p.parse_args(argv)

    if args.rate_only:
        _print_started_rate(Path(args.db))
        return 0

    summary = run(Path(args.db), args.floor, args.current, args.sleep)
    log.info("done: %d games, %d rows, %d failures", summary["games"], summary["rows"], len(summary["failures"]))
    if summary["failures"]:
        log.warning("failures (first 20): %s", summary["failures"][:20])
    _print_started_rate(Path(args.db))
    return 0


if __name__ == "__main__":
    sys.exit(main())
