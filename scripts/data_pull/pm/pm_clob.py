"""Polymarket CLOB price puller: writes pm_prices ONLY.

Two row provenances, never conflated (pinned at master level):

  history_agg rows: from /prices-history (and /batch-prices-history). The
    endpoint returns ONLY {t, p}. We write p into hist_price, record the
    fidelity/interval that actually returned data, and leave ltp/mid/bid/ask/
    sizes/volume_24h NULL. We never synthesise a depth value.

  book_snapshot rows: from the live orderbook (/book, /midpoint, /price,
    /spread). Populate ltp/mid/bid/ask/sizes/volume_24h; hist_price NULL.
    Used by the daily-incremental in-season job.

Critical operational facts (verified against live CLOB spec):
  - There is NO historical depth source. Orderbook endpoints are live, point-in-
    time only. So historical bid/ask/sizes do not exist and must stay NULL.
  - fidelity defaults to 1 minute, which returns EMPTY on old/resolved markets.
    We sweep coarse-to-fine and record what worked, so coverage is data not guess.
  - Resolved tokens can return blank even at coarse fidelity. Sparse/missing is a
    finding to REPORT, never a gap to interpolate.

Entry points:
  uv run python -m scripts.data_pull.pm.pm_clob backfill   # /prices-history for all candidates
  uv run python -m scripts.data_pull.pm.pm_clob daily       # live book snapshot for open markets
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from scripts.common.db import connect, upsert, utc_now

CLOB_BASE = "https://clob.polymarket.com"
CACHE_DIR = Path("data/cache/pm")
DB_PATH = Path("data/awards.db")

# Which market awards get a price-history backfill. The NBA tag returns ~21k
# OTHER props/game-lines alongside the few hundred award-market candidates; we
# only want prices for the markets the model trades. CHAMPIONSHIP is included
# because it is in scope for the narrative thesis (and cheap). Widen this set if
# you ever want OTHER prices too.
_AWARD_FILTER: tuple[str, ...] = ("MVP", "DPOY", "ROTY", "CHAMPIONSHIP")

# Coarse-to-fine sweep. We try the coarsest first (most likely to return data on
# old/resolved markets), and stop at the FIRST granularity that returns points.
# Each tuple is (interval, fidelity_minutes). 'max' = full available history.
_FIDELITY_SWEEP: list[tuple[str, int]] = [
    ("max", 60 * 24),   # daily
    ("max", 60 * 12),   # 12h
    ("max", 60 * 6),    # 6h
    ("max", 60),        # hourly
    ("max", 1),         # minute (rarely works on old markets)
]

_BATCH_SIZE = 50  # /batch-prices-history accepts multiple markets per request


def _cache_path(name: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe = name.replace("/", "_").replace("?", "_").replace("&", "_").replace("=", "_")
    return CACHE_DIR / f"{safe}.json"


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=30))
def _get(path: str, params: dict) -> dict:
    resp = requests.get(f"{CLOB_BASE}{path}", params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _candidates(conn) -> list[tuple[str, str]]:
    """Return (market_id, candidate_id) pairs for AWARD markets only.

    candidate_id IS the CLOB token id. We deliberately restrict the price-history
    backfill to the award markets the model actually trades (MVP/DPOY/ROTY plus
    CHAMPIONSHIP), not the ~21k OTHER props/game-lines that share the NBA tag.
    Pulling history for OTHER markets is the bulk of the runtime and feeds
    nothing downstream. To pull everything, pass a wider set via _AWARD_FILTER.
    """
    placeholders = ", ".join("?" for _ in _AWARD_FILTER)
    rows = conn.execute(
        f"SELECT c.market_id, c.candidate_id "
        f"FROM pm_candidates c JOIN pm_markets m ON c.market_id = m.market_id "
        f"WHERE m.award IN ({placeholders})",
        tuple(_AWARD_FILTER),
    ).fetchall()
    return [(r["market_id"], r["candidate_id"]) for r in rows]


def _open_candidates(conn) -> list[tuple[str, str]]:
    placeholders = ", ".join("?" for _ in _AWARD_FILTER)
    rows = conn.execute(
        f"SELECT c.market_id, c.candidate_id "
        f"FROM pm_candidates c JOIN pm_markets m ON c.market_id = m.market_id "
        f"WHERE m.status = 'open' AND m.award IN ({placeholders})",
        tuple(_AWARD_FILTER),
    ).fetchall()
    return [(r["market_id"], r["candidate_id"]) for r in rows]


# -----------------------------------------------------------------------------
# History pull (history_agg)
# -----------------------------------------------------------------------------

def _fetch_history_one(token_id: str, use_cache: bool) -> tuple[list[dict], str | None, int | None]:
    """Coarse-to-fine /prices-history sweep for one token.

    Returns (points, interval_used, fidelity_used). points is the {t,p} list;
    empty if NO granularity returned data (a coverage finding, reported upward).
    """
    for interval, fidelity in _FIDELITY_SWEEP:
        params = {"market": token_id, "interval": interval, "fidelity": fidelity}
        key = f"hist_{token_id}_{interval}_{fidelity}"
        cp = _cache_path(key)
        if use_cache and cp.exists():
            data = json.loads(cp.read_text())
        else:
            try:
                data = _get("/prices-history", params)
            except Exception:  # noqa: BLE001 - try next granularity, log at caller
                continue
            cp.write_text(json.dumps(data))
        hist = data.get("history") if isinstance(data, dict) else None
        if hist:
            return hist, interval, fidelity
    return [], None, None


def _history_rows(market_id: str, token_id: str, points: list[dict],
                  interval: str | None, fidelity: int | None) -> list[dict]:
    stamp = utc_now()
    rows = []
    for pt in points:
        t = pt.get("t")
        p = pt.get("p")
        if t is None or p is None:
            continue
        ts = datetime.fromtimestamp(int(t), tz=timezone.utc).isoformat()
        rows.append({
            "market_id": market_id,
            "candidate_id": token_id,
            "timestamp": ts,
            "price_type": "history_agg",
            "hist_price": float(p),
            "ltp": None, "mid": None, "bid": None, "ask": None,
            "bid_size": None, "ask_size": None, "volume_24h": None,
            "fidelity": fidelity,
            "interval": interval,
            "pulled_at": stamp,
        })
    return rows


def backfill_history(conn, use_cache: bool = True) -> dict:
    """Pull /prices-history for every candidate token. Reports coverage."""
    cands = _candidates(conn)
    written = 0
    no_data: list[str] = []
    fidelity_used: dict[str, int] = {}
    for market_id, token_id in cands:
        points, interval, fidelity = _fetch_history_one(token_id, use_cache)
        if not points:
            no_data.append(token_id)
            continue
        rows = _history_rows(market_id, token_id, points, interval, fidelity)
        upsert(conn, "pm_prices", rows, ["market_id", "candidate_id", "timestamp"])
        written += len(rows)
        label = f"{interval}/{fidelity}"
        fidelity_used[label] = fidelity_used.get(label, 0) + 1
    return {
        "candidates_total": len(cands),
        "candidates_with_data": len(cands) - len(no_data),
        "candidates_no_data": len(no_data),
        "price_rows_written": written,
        "granularity_distribution": fidelity_used,
    }


# -----------------------------------------------------------------------------
# Live book snapshot (book_snapshot)
# -----------------------------------------------------------------------------

def _fetch_book(token_id: str) -> dict:
    """Live top-of-book + midpoint for one token. Best-effort per field."""
    out: dict = {"bid": None, "ask": None, "bid_size": None, "ask_size": None,
                 "mid": None, "ltp": None}
    try:
        book = _get("/book", {"token_id": token_id})
        bids = book.get("bids") or []
        asks = book.get("asks") or []
        if bids:
            out["bid"] = float(bids[0]["price"])
            out["bid_size"] = float(bids[0]["size"])
        if asks:
            out["ask"] = float(asks[0]["price"])
            out["ask_size"] = float(asks[0]["size"])
    except Exception:  # noqa: BLE001
        pass
    try:
        mid = _get("/midpoint", {"token_id": token_id})
        out["mid"] = float(mid["mid"])
    except Exception:  # noqa: BLE001
        pass
    try:
        last = _get("/last-trade-price", {"token_id": token_id})
        out["ltp"] = float(last["price"])
    except Exception:  # noqa: BLE001
        pass
    return out


def daily_snapshot(conn) -> dict:
    """Live book snapshot for open-market candidates. price_type=book_snapshot."""
    cands = _open_candidates(conn)
    stamp = utc_now()
    rows = []
    failures: list[str] = []
    for market_id, token_id in cands:
        try:
            book = _fetch_book(token_id)
            rows.append({
                "market_id": market_id,
                "candidate_id": token_id,
                "timestamp": stamp,
                "price_type": "book_snapshot",
                "hist_price": None,
                "ltp": book["ltp"], "mid": book["mid"],
                "bid": book["bid"], "ask": book["ask"],
                "bid_size": book["bid_size"], "ask_size": book["ask_size"],
                "volume_24h": None,  # populate if/when a live vol field is wired
                "fidelity": None, "interval": None,
                "pulled_at": stamp,
            })
        except Exception as e:  # noqa: BLE001
            failures.append(f"{token_id}: {e}")
    if rows:
        upsert(conn, "pm_prices", rows, ["market_id", "candidate_id", "timestamp"])
    return {"snapshots_written": len(rows), "failures": failures}


def run(mode: str, db_path: Path = DB_PATH, use_cache: bool = True) -> dict:
    conn = connect(db_path)
    try:
        if mode == "backfill":
            return backfill_history(conn, use_cache=use_cache)
        elif mode == "daily":
            return daily_snapshot(conn)
        raise ValueError(f"unknown mode {mode!r}; use 'backfill' or 'daily'")
    finally:
        conn.close()


def main(argv: list[str]) -> int:
    mode = argv[1] if len(argv) > 1 else "daily"
    print(json.dumps(run(mode), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))