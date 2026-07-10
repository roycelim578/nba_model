"""Database connection + upsert helpers.

LOCAL THIN HELPER, pinned-interface compatible.

This is a minimal local implementation of the shared DB helper. Per the Phase 1
contract, the canonical `src/data/db.py` is owned by the nba_api chat (Chat C).
If/when that module lands, delete this file and import from it instead: the
public surface here (connect, upsert, utcnow_iso) is deliberately identical, so
the swap is a one-line import change, not a rewrite.

Pinned interface:
    connect(db_path) -> sqlite3.Connection   # foreign_keys=ON, WAL
    upsert(conn, table, rows, pk_cols)       # idempotent INSERT..ON CONFLICT
    utcnow_iso()                             # pulled_at stamp source
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path


def connect(db_path: str | Path) -> sqlite3.Connection:
    """Open a SQLite connection with foreign keys and WAL enabled.

    Matches the pinned shared signature. Row factory is set to sqlite3.Row so
    callers can use column-name access.
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def utcnow_iso() -> str:
    """UTC timestamp in ISO-8601, used to stamp `pulled_at`-style columns."""
    return datetime.now(timezone.utc).isoformat()


def upsert(
    conn: sqlite3.Connection,
    table: str,
    rows: Sequence[dict],
    pk_cols: Sequence[str],
) -> int:
    """Idempotent bulk upsert: INSERT ... ON CONFLICT(pk) DO UPDATE.

    rows: list of dicts with identical key sets. Empty -> no-op.
    pk_cols: the conflict target. Non-PK columns are overwritten on conflict.

    Returns number of rows submitted. Re-running with the same data must never
    duplicate or error (the whole point of the ON CONFLICT clause).
    """
    if not rows:
        return 0

    cols = list(rows[0].keys())
    # Guard: every row must share the same column set, or the executemany
    # parameter binding silently misaligns.
    for r in rows:
        if set(r.keys()) != set(cols):
            raise ValueError(
                f"upsert row column mismatch for table {table}: "
                f"expected {sorted(cols)}, got {sorted(r.keys())}"
            )

    placeholders = ", ".join(f":{c}" for c in cols)
    col_list = ", ".join(cols)
    conflict = ", ".join(pk_cols)
    # Update every non-PK column from the would-be-inserted row (excluded.*).
    update_cols = [c for c in cols if c not in pk_cols]
    if update_cols:
        set_clause = ", ".join(f"{c} = excluded.{c}" for c in update_cols)
        do_clause = f"DO UPDATE SET {set_clause}"
    else:
        # PK-only table (no other columns to update): conflict is a no-op.
        do_clause = "DO NOTHING"

    sql = (
        f"INSERT INTO {table} ({col_list}) VALUES ({placeholders}) "
        f"ON CONFLICT({conflict}) {do_clause}"
    )
    conn.executemany(sql, rows)
    conn.commit()
    return len(rows)


utc_now = utcnow_iso


utc_now = utcnow_iso
