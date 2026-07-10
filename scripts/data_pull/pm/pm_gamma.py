"""Polymarket Gamma puller: market discovery + candidate population.

Writes ONLY pm_markets and pm_candidates. Never touches players, name_resolution,
stg_ tables, or pm_prices (that is pm_clob.py's job). Never assigns player_id.

Structure decision (pinned at master level):
  - One pm_markets row per binary sub-market. A PM award event ("NBA MVP
    2025-26") contains many binary sub-markets, one per player ("Will Jokic win
    MVP?"). Each sub-market is its own market_id.
  - pm_candidates.candidate_id == the CLOB token id (NOT the outcome index).
    Both Yes and No outcomes get their own candidate row, keyed by their token
    id, with outcome ('Yes'/'No') and outcome_index recorded.
  - player_id is left strictly NULL; the resolution chat owns that backfill.

Entry points:
  uv run python -m scripts.data_pull.pm.pm_gamma backfill   # all NBA-relevant history
  uv run python -m scripts.data_pull.pm.pm_gamma daily       # open markets only, in-season
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from scripts.data_pull.pm import pm_classify
from scripts.common.db import connect, upsert, utc_now

GAMMA_BASE = "https://gamma-api.polymarket.com"
CACHE_DIR = Path("data/cache/pm")
DB_PATH = Path("data/awards.db")

# NBA award events live under NBA-related tags/slugs. We filter server-side by
# the NBA tag_id so we fetch only NBA markets (a few hundred) instead of the
# whole site (tens of thousands), which both avoids Gamma's offset ceiling and
# is far faster. _looks_nba remains as a secondary client-side safety net. We
# still let the classifier (incl. OTHER) decide the award; we do NOT pre-filter
# out non-award NBA markets.
_NBA_TAG_ID = 745  # Polymarket Gamma tag id for NBA (verified against /tags)
_NBA_KEYWORDS = ("nba", "basketball")


# -----------------------------------------------------------------------------
# HTTP with caching + retries
# -----------------------------------------------------------------------------

def _cache_path(name: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe = name.replace("/", "_").replace("?", "_").replace("&", "_").replace("=", "_")
    return CACHE_DIR / f"{safe}.json"


class _StopPaginate(Exception):
    """Raised on a 422 offset-ceiling response: a clean end-of-pagination, not
    a transient error, so it must bypass tenacity's retry."""


@retry(
    retry=retry_if_exception_type(requests.RequestException),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=30),
)
def _get(path: str, params: dict | None = None) -> object:
    """GET against Gamma with retry/backoff. Raises on non-200 to trigger retry.

    A 422 (offset past available depth) is NOT transient: retrying it wastes
    four attempts on a deterministic failure. We convert it to _StopPaginate,
    which is not in the retry set, so it propagates immediately for the
    paginator to treat as end-of-pages.
    """
    resp = requests.get(f"{GAMMA_BASE}{path}", params=params or {}, timeout=30)
    if resp.status_code == 422:
        raise _StopPaginate()
    resp.raise_for_status()
    return resp.json()


def _get_cached(path: str, params: dict | None, cache_key: str, use_cache: bool) -> object:
    """GET with optional disk cache replay for dev iterations."""
    cp = _cache_path(cache_key)
    if use_cache and cp.exists():
        return json.loads(cp.read_text())
    data = _get(path, params)
    cp.write_text(json.dumps(data))
    return data


# -----------------------------------------------------------------------------
# Discovery
# -----------------------------------------------------------------------------

def _paginate_events(closed: bool, use_cache: bool) -> list[dict]:
    """Page through /events filtered to the NBA tag, one `closed` value.

    Award markets on Polymarket are EVENTS ('NBA MVP 2025-26') that contain
    per-player binary sub-markets in an event['markets'] array. The flat
    /markets list is dominated by props (draft order, retirements) and does not
    surface the award sub-markets well, so discovery goes through /events and
    walks each event's markets, which is Polymarket's documented pattern for
    multi-outcome award events.

    `closed` MUST be passed explicitly: Gamma defaults closed=false when the
    param is omitted, so the resolved PAST-season award markets (the ones with
    price history to backtest against) are silently excluded unless we ask for
    closed=true. Backfill therefore calls this once for each of True and False.

    422 on a too-deep offset is treated as a clean end-of-pagination.
    """
    out: list[dict] = []
    limit = 100
    offset = 0
    while True:
        params: dict = {
            "limit": limit,
            "offset": offset,
            "tag_id": _NBA_TAG_ID,
            "closed": str(closed).lower(),
        }
        key = f"events_tag-{_NBA_TAG_ID}_closed-{closed}_off-{offset}"
        try:
            batch = _get_cached("/events", params, key, use_cache)
        except _StopPaginate:
            break
        if not isinstance(batch, list) or not batch:
            break
        out.extend(batch)
        if len(batch) < limit:
            break
        offset += limit
    return out


def _markets_from_events(events: list[dict]) -> list[dict]:
    """Flatten the sub-markets out of a list of events.

    Each event carries a 'markets' array of binary sub-markets. We stamp each
    sub-market with the parent event's title and slug under reserved keys so the
    classifier can read the award name from the EVENT (where 'MVP'/'DPOY'/'ROTY'
    actually live) even when the sub-market's own question is just a player
    framing. Downstream parsing/classification is unchanged otherwise.
    """
    out: list[dict] = []
    for ev in events:
        ev_title = ev.get("title") or ev.get("question")
        ev_slug = ev.get("slug")
        for m in ev.get("markets", []) or []:
            m = dict(m)  # shallow copy so we don't mutate the cached payload
            m["_event_title"] = ev_title
            m["_event_slug"] = ev_slug
            out.append(m)
    return out


def _discover_nba_markets(mode: str, use_cache: bool) -> list[dict]:
    """Return the flat list of NBA sub-markets for the given mode.

    backfill: both closed and open events (full history Gamma exposes).
    daily: open events only.
    """
    if mode == "backfill":
        events = _paginate_events(closed=True, use_cache=use_cache)
        events += _paginate_events(closed=False, use_cache=use_cache)
    else:  # daily
        events = _paginate_events(closed=False, use_cache=use_cache)
    return _markets_from_events(events)


def _looks_nba(market: dict) -> bool:
    """Heuristic NBA-relevance gate on a raw market dict.

    We keep this permissive: better to capture an OTHER NBA market than to miss
    an award market because of an unusual title. Checks slug, question, and any
    tag/category fields Gamma exposes.
    """
    hay = " ".join(
        str(market.get(k, "")).lower()
        for k in ("slug", "question", "title", "description", "category",
                  "_event_title", "_event_slug")
    )
    # tags may be a list of dicts or strings depending on Gamma shape.
    for t in market.get("tags", []) or []:
        hay += " " + (t.get("slug", "") + " " + t.get("label", "") if isinstance(t, dict) else str(t)).lower()
    return any(kw in hay for kw in _NBA_KEYWORDS)


# -----------------------------------------------------------------------------
# Parsing one market -> rows
# -----------------------------------------------------------------------------

def _parse_outcomes(market: dict) -> list[tuple[str, str, int]]:
    """Extract (token_id, outcome_name, outcome_index) per outcome.

    Gamma returns `outcomes` and `clobTokenIds` as JSON-encoded string arrays
    that map 1:1 by index. We zip them. If either is missing/malformed, return
    empty (caller logs and skips); we never fabricate token ids.
    """
    raw_outcomes = market.get("outcomes")
    raw_tokens = market.get("clobTokenIds")
    try:
        outcomes = json.loads(raw_outcomes) if isinstance(raw_outcomes, str) else (raw_outcomes or [])
        tokens = json.loads(raw_tokens) if isinstance(raw_tokens, str) else (raw_tokens or [])
    except (json.JSONDecodeError, TypeError):
        return []
    if not outcomes or not tokens or len(outcomes) != len(tokens):
        return []
    return [(str(tok), str(name), i) for i, (name, tok) in enumerate(zip(outcomes, tokens))]


def _season_from(market: dict) -> int | None:
    """Best-effort STARTING-year season stamp (2025-26 -> 2025).

    Looks for a 4-digit year or YYYY-YY span in slug/question. Returns None if
    nothing parseable; we do not guess a season we can't see.
    """
    import re
    hay = f"{market.get('slug','')} {market.get('question','')}"
    # YYYY-YY span -> starting year
    m = re.search(r"(20\d{2})\s*[-/]\s*\d{2}", hay)
    if m:
        return int(m.group(1))
    # bare year
    m = re.search(r"\b(20\d{2})\b", hay)
    if m:
        return int(m.group(1))
    return None


def _player_from_question(question: str | None, event_title: str | None) -> str | None:
    """Extract the player name from an award sub-market question.

    Binary award sub-markets phrase the player in the QUESTION, not the outcome
    token (the outcome is just 'Yes'/'No'). The consistent Polymarket phrasing is
    'Will <Player> win the <YYYY-YY> NBA MVP?' (and DPOY/ROTY variants). We pull
    the span between 'Will ' and ' win'. Returns None if the pattern does not
    match, so a non-player question never yields a bogus name.
    """
    import re
    text = (question or event_title or "").strip()
    if not text:
        return None
    m = re.search(r"\bwill\s+(.+?)\s+win\b", text, flags=re.IGNORECASE)
    if m:
        name = m.group(1).strip()
        if name and name.lower() not in ("the", "a", "an"):
            return name
    return None


def _market_rows(market: dict) -> tuple[dict | None, list[dict]]:
    """Build (pm_markets row, [pm_candidates rows]) from a raw Gamma market.

    Returns (None, []) if the market has no usable outcome tokens, so the caller
    can log and continue rather than write a market with no tradable candidates.
    """
    market_id = str(market.get("id") or market.get("conditionId") or "").strip()
    if not market_id:
        return None, []

    # Award name lives on the parent EVENT ('NBA MVP 2025-26'), not necessarily
    # on the per-player sub-market question. Classify from the event title/slug
    # first (stamped on by _markets_from_events), falling back to the market's
    # own question/title/slug.
    award, award_raw = pm_classify.classify_award(
        market.get("_event_title") or market.get("question") or market.get("title"),
        market.get("_event_slug") or market.get("slug"),
    )

    status = "closed" if market.get("closed") else ("open" if market.get("active") else None)
    if market.get("umaResolutionStatus") == "resolved" or market.get("resolved"):
        status = "resolved"

    stamp = utc_now()
    mrow = {
        "market_id": market_id,
        "condition_id": market.get("conditionId"),
        "slug": market.get("slug"),
        "award": award,
        "season": _season_from(market),
        "status": status,
        "start_date": market.get("startDate"),
        "end_date": market.get("endDate"),
        "resolution_date": market.get("umaEndDate") or market.get("endDate"),
        "description": market.get("description"),
        "award_raw": award_raw,
        "pulled_at": stamp,
    }

    # For binary award sub-markets the outcome token is 'Yes'/'No' and the player
    # lives in the question; parse it so the resolver has a name to match. For
    # multi-outcome markets the outcome name IS the player, so keep it. OTHER
    # markets have no player to parse, so candidate_name stays the raw outcome.
    player_name = None
    if award in (pm_classify.MVP, pm_classify.DPOY, pm_classify.ROTY):
        player_name = _player_from_question(
            market.get("question") or market.get("title"),
            market.get("_event_title"),
        )

    crows: list[dict] = []
    for token_id, outcome_name, idx in _parse_outcomes(market):
        is_yesno = outcome_name in ("Yes", "No")
        # candidate_name carries the player where we can identify one: the parsed
        # question player for Yes/No award markets, else the outcome name itself.
        cand_name = player_name if (is_yesno and player_name) else outcome_name
        crows.append({
            "market_id": market_id,
            "candidate_id": token_id,       # CLOB token id is the natural key
            "candidate_name": cand_name,
            "player_id": None,               # STRICTLY NULL: resolution chat owns
            "outcome": outcome_name if is_yesno else None,
            "outcome_index": idx,
            "pulled_at": stamp,
        })
    return mrow, crows


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------

def run(mode: str, db_path: Path = DB_PATH, use_cache: bool = True) -> dict:
    """Run discovery+population. mode in {'backfill','daily'}.

    backfill: all markets (open + closed/resolved), full history Gamma exposes.
    daily: open markets only, for the in-season incremental job.

    Returns a summary dict (counts + failures) for the findings note.
    """
    if mode not in ("backfill", "daily"):
        raise ValueError(f"unknown mode {mode!r}; use 'backfill' or 'daily'")
    raw = _discover_nba_markets(mode, use_cache=use_cache)

    nba = [m for m in raw if _looks_nba(m)]
    conn = connect(db_path)
    m_rows: list[dict] = []
    c_rows: list[dict] = []
    failures: list[str] = []
    award_counts: dict[str, int] = {}

    for m in nba:
        try:
            mrow, crows = _market_rows(m)
            if mrow is None or not crows:
                failures.append(f"no usable outcomes: {m.get('slug') or m.get('id')}")
                continue
            m_rows.append(mrow)
            c_rows.extend(crows)
            award_counts[mrow["award"]] = award_counts.get(mrow["award"], 0) + 1
        except Exception as e:  # noqa: BLE001 - collect, never crash the whole run
            failures.append(f"parse error {m.get('slug') or m.get('id')}: {e}")

    upsert(conn, "pm_markets", m_rows, ["market_id"])
    upsert(conn, "pm_candidates", c_rows, ["market_id", "candidate_id"])
    conn.close()

    summary = {
        "mode": mode,
        "raw_markets": len(raw),
        "nba_markets": len(nba),
        "markets_written": len(m_rows),
        "candidates_written": len(c_rows),
        "award_counts": award_counts,
        "failures": failures,
    }
    return summary


def main(argv: list[str]) -> int:
    mode = argv[1] if len(argv) > 1 else "daily"
    summary = run(mode)
    print(json.dumps(summary, indent=2))
    if summary["failures"]:
        print(f"\n{len(summary['failures'])} item(s) failed (logged above), run continued.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))