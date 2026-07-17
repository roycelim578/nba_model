"""Cross-sport stat-leader forward-vol corpus builder (NBA EXCLUDED).

Builds the corpus the stat-leader forward-vol refit trains on, mirroring the
voter corpus schema but keyed on other-sport, season-long "will X lead <league>
in <stat>" markets. The traded NBA books must NOT enter this fit, so NBA (Gamma
tag 745) is excluded at discovery and a client-side net drops any NBA-flavoured
market that leaks through a cross-listed tag.

Self-contained by design (pinned at master level): discovered series are written
STRAIGHT into corpus_market_stat / corpus_price_daily_stat and NEVER routed
through pm_markets / pm_candidates. The NBA data layer stays clean and the stat
corpus stands alone. The classifier is not used at all.

Discovery mirrors pm_gamma: page /events filtered server-side by a tag id
(closed true and false), flatten each event['markets'] array, read the
index-aligned outcomes / clobTokenIds. Tag ids for the target leagues are
discovered from /tags first. History reuses pm_clob._fetch_history_one /
_history_rows verbatim, then resamples the {t,p} points to a daily close.

One SERIES per tradable outcome token: the Yes leg of a binary sub-market, or
every player-outcome of a multi-outcome "who leads" market (the No leg is
dropped). Series group into a LOO event via event_slug = LEAGUE|STAT|SEASON, so
a held-out unit is a whole league-stat-season race, mirroring the voter corpus
holding out an award-season.

Network / DB imports are lazy (inside the functions that use them) so the pure
selection, resample and slug logic imports and unit-tests offline without
requests / tenacity / the DB. WAL: run as the SINGLE writer during the pull.

  uv run python3 -m scripts.data_pull.pm.pm_corpus_stat --tags-only            # dry run: print matched tags
  uv run python3 -m scripts.data_pull.pm.pm_corpus_stat --limit 30             # cautious first pull
  caffeinate -i uv run python3 -m scripts.data_pull.pm.pm_corpus_stat --no-progress   # full pull
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

GAMMA_BASE = "https://gamma-api.polymarket.com"
CACHE_DIR = Path("data/cache/pm")
DB_PATH = Path("data/awards.db")

NBA_TAG_ID = 745  # excluded: the traded book must never enter the vol fit
# Word-boundary match so the net catches NBA without also catching WNBA (whose
# slug contains the substring 'nba'). Kept as a belt-and-braces text net at the
# market level even though the tag allowlist below already excludes NBA.
_RX_NBA = re.compile(r"\bnba\b", re.I)

# Curated tag-id allowlist per target league, pinned from a live /tags dump (the
# same way NBA=745 was verified). Keyword-walking /tags was abandoned: acronym
# substrings (influenza/inflation/conflict/netflix/SNHL) and shared competition
# names (Basketball/Table-Tennis Champions League, cricket Premier Leagues) both
# leaked the wrong sport into the corpus. An explicit id list is precise and
# reviewable. For the four count sports the base league tag is unambiguous; for
# soccer the big-five leagues plus Champions League and the World Cup are where
# Golden Boot / top-scorer markets concentrate. Extend by adding ids here.
LEAGUE_TAG_IDS: dict[str, list[int]] = {
    "WNBA": [100254],
    "MLB": [100381, 100380],
    "NFL": [450],
    "NHL": [899],
    "SOCCER": [82, 306, 103043,     # Premier League / EPL / English Premier League
               780, 100618, 1494, 102070,   # La Liga / Serie A / Bundesliga / Ligue 1
               1234, 102469,        # Champions League / UEFA Champions League
               100350, 100834,      # Soccer / Intl. Soccer
               102232, 102350],     # FIFA World Cup / 2026 FIFA World Cup
}
assert NBA_TAG_ID not in {t for ids in LEAGUE_TAG_IDS.values() for t in ids}

# Season-long stat-leader inclusion. Deliberately broad: the season-context
# exclusion below and the --tags-only / --limit dry runs are the safety valves,
# and the vol model is stat-agnostic so over-capture of leader markets is benign.
RX_STAT_INCLUDE = re.compile(
    r"lead(s|er)?\b.{0,20}\bin\b|league leader|"
    r"\btop scorer\b|scoring (title|champion|leader|crown)|most points|"
    r"\bgolden boot\b|most goals|top (goal ?scorer|assister)|most assists|"
    r"\bart ross\b|\brocket richard\b|maurice[- ]richard|"
    r"batting (title|champion|crown)|most (home runs|hrs?|rbis?|strikeouts|hits|stolen bases|saves)|"
    r"home run (title|leader|champion|crown)|"
    r"most (sacks|interceptions|receptions|receiving yards|rushing yards|passing yards|touchdowns|tackles)|"
    r"(rebound|assist|block|steal|point|goal|save|tackle)s?\s+(title|leader|champion|crown)|"
    r"\b(ppg|rpg|apg|bpg|spg)\b",
    re.I,
)

# Family label for reporting + the STAT axis of event_slug. Coarse; a miss into
# 'other' does not affect the stat-agnostic vol fit, only the report granularity.
_FAMILY_RX: list[tuple[str, str]] = [
    ("points", r"point|scoring|ppg|top scorer|golden boot|most goals|goal|rocket richard|maurice"),
    ("rebounds", r"rebound|\brpg\b"),
    ("assists", r"assist|\bapg\b|art ross"),
    ("blocks", r"block|\bbpg\b"),
    ("steals", r"steal|\bspg\b"),
    ("home_runs", r"home run|\bhrs?\b"),
    ("rbi", r"\brbis?\b"),
    ("strikeouts", r"strikeout"),
    ("batting", r"batting (title|champion|crown)|most hits|stolen bases"),
    ("sacks", r"sack"),
    ("interceptions", r"interception"),
    ("yards", r"(receiving|rushing|passing) yards"),
    ("receptions", r"reception"),
    ("touchdowns", r"touchdown"),
    ("tackles", r"tackle"),
    ("saves", r"\bsaves?\b"),
]

# Context filter mirrored from pm_clob_targeted, widened cross-sport. Only
# 'season' context is kept; single-game and series/postseason props are dropped.
_RX_SINGLE_GAME = re.compile(r"\btonight\b|\btoday\b|\bgame[ -]?\d\b|\bweek[ -]?\d+\b|\bmatchday\b", re.I)
_RX_SERIES = re.compile(
    r"\bseries\b|\bplayoff|\bpostseason\b|world series|super ?bowl|stanley cup|"
    r"conference (final|semi)|\bfinals?\b|\bvs\.?\b|\bwild ?card\b|group stage|knockout",
    re.I,
)
_RX_TITLE_OUTCOME = re.compile(r"win the .*(final|championship|title|cup|bowl|series)", re.I)


# ---------------------------------------------------------------- HTTP (lazy)

def _cache_path(name: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe = name.replace("/", "_").replace("?", "_").replace("&", "_").replace("=", "_")
    return CACHE_DIR / f"{safe}.json"


class _StopPaginate(Exception):
    """A 422 offset-ceiling response: clean end-of-pagination, not transient."""


def _make_get():
    """Build the retrying GET, importing requests / tenacity lazily so the module
    imports offline. Mirrors pm_gamma._get semantics (422 -> _StopPaginate)."""
    import requests
    from tenacity import (retry, retry_if_exception_type, stop_after_attempt,
                          wait_exponential)

    @retry(retry=retry_if_exception_type(requests.RequestException),
           stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=30))
    def _get(path: str, params: dict | None = None):
        resp = requests.get(f"{GAMMA_BASE}{path}", params=params or {}, timeout=30)
        if resp.status_code == 422:
            raise _StopPaginate()
        resp.raise_for_status()
        return resp.json()

    return _get


def _get_cached(get, path, params, cache_key, use_cache):
    cp = _cache_path(cache_key)
    if use_cache and cp.exists():
        return json.loads(cp.read_text())
    data = get(path, params)
    cp.write_text(json.dumps(data))
    return data


# ------------------------------------------------------------- tag discovery

def _paginate_tags(get, use_cache) -> list[dict]:
    out: list[dict] = []
    limit, offset = 100, 0
    while True:
        key = f"tags_off-{offset}"
        try:
            batch = _get_cached(get, "/tags", {"limit": limit, "offset": offset}, key, use_cache)
        except _StopPaginate:
            break
        if not isinstance(batch, list) or not batch:
            break
        out.extend(batch)
        if len(batch) < limit:
            break
        offset += limit
    return out


def _tag_labels(get, ids: set[int], use_cache) -> dict[int, str]:
    """Resolve {tag_id: label} for the pinned ids from /tags, for the confirm
    print. Pages /tags once (cached); missing ids surface as '(id not found)'."""
    labels: dict[int, str] = {}
    for t in _paginate_tags(get, use_cache):
        tid = t.get("id")
        try:
            tid_i = int(tid)
        except (TypeError, ValueError):
            continue
        if tid_i in ids:
            labels[tid_i] = str(t.get("label") or t.get("slug") or "")
    return {i: labels.get(i, "(id not found)") for i in ids}


# ------------------------------------------------------- market discovery

def _paginate_events(get, tag_id: int, closed: bool, use_cache: bool) -> list[dict]:
    out: list[dict] = []
    limit, offset = 100, 0
    while True:
        params = {"limit": limit, "offset": offset, "tag_id": tag_id,
                  "closed": str(closed).lower()}
        key = f"events_tag-{tag_id}_closed-{closed}_off-{offset}"
        try:
            batch = _get_cached(get, "/events", params, key, use_cache)
        except _StopPaginate:
            break
        if not isinstance(batch, list) or not batch:
            break
        out.extend(batch)
        if len(batch) < limit:
            break
        offset += limit
    return out


def _markets_from_events(events: list[dict], league: str) -> list[dict]:
    """Flatten sub-markets, stamping the parent event title/slug and the league
    the tag walk came from (under reserved keys) for selection and slug-building."""
    out: list[dict] = []
    for ev in events:
        ev_title = ev.get("title") or ev.get("question")
        ev_slug = ev.get("slug")
        for m in ev.get("markets", []) or []:
            m = dict(m)
            m["_event_title"] = ev_title
            m["_event_slug"] = ev_slug
            m["_league"] = league
            out.append(m)
    return out


# ----------------------------------------------------------- pure selection

def _normalise(*parts) -> str:
    joined = " ".join(str(p) for p in parts if p)
    joined = joined.replace("-", " ").replace("_", " ")
    return re.sub(r"\s+", " ", joined).strip().lower()


def _context(text: str) -> str:
    if _RX_SINGLE_GAME.search(text):
        return "single_game"
    if _RX_TITLE_OUTCOME.search(text):
        return "season"
    if _RX_SERIES.search(text):
        return "finals_series"
    return "season"


def _stat_family(text: str) -> str:
    for fam, rx in _FAMILY_RX:
        if re.search(rx, text, re.I):
            return fam
    return "other"


def _looks_nba(text: str) -> bool:
    return bool(_RX_NBA.search(text))


def _season_from(text: str) -> int | None:
    m = re.search(r"(20\d{2})\s*[-/]\s*\d{2}", text)
    if m:
        return int(m.group(1))
    m = re.search(r"\b(20\d{2})\b", text)
    return int(m.group(1)) if m else None


def _parse_outcomes(market: dict) -> list[tuple[str, str, int]]:
    """(token_id, outcome_name, outcome_index) per outcome; [] if malformed. The
    outcomes / clobTokenIds are index-aligned JSON string arrays (as pm_gamma)."""
    raw_o = market.get("outcomes")
    raw_t = market.get("clobTokenIds")
    try:
        outcomes = json.loads(raw_o) if isinstance(raw_o, str) else (raw_o or [])
        tokens = json.loads(raw_t) if isinstance(raw_t, str) else (raw_t or [])
    except (json.JSONDecodeError, TypeError):
        return []
    if not outcomes or not tokens or len(outcomes) != len(tokens):
        return []
    return [(str(tok), str(name), i) for i, (name, tok) in enumerate(zip(outcomes, tokens))]


def _series_from_market(market: dict) -> list[dict]:
    """Zero or more series descriptors from one raw market. A series is one
    tradable outcome token: the Yes leg of a binary sub-market, or each
    player-outcome of a multi-outcome market. The No leg is dropped (its price is
    1 - Yes and would double-count the race). Non-stat, non-season, single-game,
    series/postseason and NBA-flavoured markets yield nothing."""
    text = _normalise(market.get("_event_title"), market.get("_event_slug"),
                       market.get("question"), market.get("title"),
                       market.get("slug"), market.get("description"))
    if not text or _looks_nba(text):
        return []
    if not RX_STAT_INCLUDE.search(text):
        return []
    if _context(text) != "season":
        return []
    league = str(market.get("_league") or "OTHER").upper()
    stat = _stat_family(text)
    season = _season_from(text)
    event_slug = f"{league}|{stat}|{season if season is not None else 'NA'}"
    out: list[dict] = []
    for token_id, outcome_name, idx in _parse_outcomes(market):
        if outcome_name.strip().lower() == "no":
            continue
        out.append({
            "token_id": token_id,
            "outcome": outcome_name,
            "league": league,
            "stat_family": stat,
            "season": season,
            "event_slug": event_slug,
            "slug": market.get("slug"),
        })
    return out


def _to_daily(points: list[dict]) -> list[tuple[str, float]]:
    """Resample {t (unix s), p} points to one price per UTC calendar day (the
    day's last observation), sorted ascending. Mirrors the voter corpus daily
    grain. Malformed points are skipped, never interpolated."""
    by_day: dict[str, tuple[int, float]] = {}
    for pt in points:
        t, p = pt.get("t"), pt.get("p")
        if t is None or p is None:
            continue
        try:
            ts = int(t)
            price = float(p)
        except (TypeError, ValueError):
            continue
        day = datetime.fromtimestamp(ts, tz=timezone.utc).date().isoformat()
        prev = by_day.get(day)
        if prev is None or ts >= prev[0]:
            by_day[day] = (ts, price)
    return [(d, v[1]) for d, v in sorted(by_day.items())]


# --------------------------------------------------------------- schema / IO

_SCHEMA = """
CREATE TABLE IF NOT EXISTS corpus_market_stat (
  market_id  TEXT PRIMARY KEY,
  event_slug TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS corpus_price_daily_stat (
  market_id TEXT NOT NULL,
  day       TEXT NOT NULL,
  yes_price REAL NOT NULL,
  PRIMARY KEY (market_id, day)
);
CREATE INDEX IF NOT EXISTS ix_cpds_market ON corpus_price_daily_stat(market_id);
"""


def _ensure_schema(conn):
    conn.executescript(_SCHEMA)
    conn.commit()


# ------------------------------------------------------------- orchestration

def run(tags_only=False, limit=None, use_cache=True, progress=True,
        db_path: Path = DB_PATH) -> dict:
    """Discover cross-sport stat-leader series and pull their daily history into
    the stat corpus tables. --tags-only stops after printing matched tag ids."""
    from scripts.common.db import connect, upsert
    from scripts.data_pull.pm.pm_clob import _fetch_history_one, _history_rows  # noqa: F401

    get = _make_get()

    all_ids = {t for ids in LEAGUE_TAG_IDS.values() for t in ids}
    labels = _tag_labels(get, all_ids, use_cache)
    print("pinned league tags (NBA 745 excluded):")
    for lg, ids in LEAGUE_TAG_IDS.items():
        shown = ", ".join(f"{t}:{labels.get(t, '?')}" for t in ids)
        print(f"  {lg:>7}: {shown}")
    if tags_only:
        return {"pinned_tag_ids": LEAGUE_TAG_IDS,
                "resolved_labels": {str(k): v for k, v in labels.items()}}

    # discover + select series across every pinned non-NBA tag
    series: dict[str, dict] = {}  # token_id -> descriptor (dedup across tags)
    for lg, ids in LEAGUE_TAG_IDS.items():
        for tag_id in ids:
            evs = _paginate_events(get, tag_id, True, use_cache)
            evs += _paginate_events(get, tag_id, False, use_cache)
            for m in _markets_from_events(evs, lg):
                for s in _series_from_market(m):
                    series.setdefault(s["token_id"], s)
    tokens = list(series.values())
    if limit:
        tokens = tokens[:limit]

    fam_tally: dict[str, int] = {}
    league_tally: dict[str, int] = {}
    for s in tokens:
        fam_tally[s["stat_family"]] = fam_tally.get(s["stat_family"], 0) + 1
        league_tally[s["league"]] = league_tally.get(s["league"], 0) + 1
    print(f"selected {len(tokens)} candidate series across leagues {league_tally} "
          f"families {fam_tally}")

    conn = connect(db_path)
    try:
        _ensure_schema(conn)
        m_rows, p_rows = [], []
        lengths: list[int] = []
        no_data = 0
        for i, s in enumerate(tokens, 1):
            tok = s["token_id"]
            points, interval, fidelity = _fetch_history_one(tok, use_cache)
            daily = _to_daily(points) if points else []
            if not daily:
                no_data += 1
            else:
                m_rows.append({"market_id": tok, "event_slug": s["event_slug"]})
                for day, price in daily:
                    p_rows.append({"market_id": tok, "day": day, "yes_price": price})
                lengths.append(len(daily))
            if progress and i % 25 == 0:
                print(f"  ... {i}/{len(tokens)} tokens, {len(m_rows)} series, "
                      f"{no_data} empty", flush=True)
        if m_rows:
            upsert(conn, "corpus_market_stat", m_rows, ["market_id"])
        if p_rows:
            upsert(conn, "corpus_price_daily_stat", p_rows, ["market_id", "day"])
    finally:
        conn.close()

    lengths.sort()
    ge12 = sum(1 for n in lengths if n >= 12)
    def _pct(q):
        return lengths[min(len(lengths) - 1, int(q * len(lengths)))] if lengths else 0
    summary = {
        "series_candidates": len(tokens),
        "series_with_data": len(m_rows),
        "series_no_data": no_data,
        "series_ge_12_days": ge12,
        "price_rows_written": len(p_rows),
        "league_tally": league_tally,
        "family_tally": fam_tally,
        "length_distribution": {"min": lengths[0] if lengths else 0,
                                "p25": _pct(0.25), "median": _pct(0.5),
                                "p75": _pct(0.75), "max": lengths[-1] if lengths else 0},
    }
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Cross-sport stat-leader vol corpus (NBA excluded)")
    ap.add_argument("--tags-only", action="store_true", help="print matched tag ids and stop")
    ap.add_argument("--limit", type=int, default=None, help="cap series for a cautious first pull")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--no-progress", action="store_true")
    ap.add_argument("--db", default=str(DB_PATH))
    args = ap.parse_args(argv)
    summary = run(tags_only=args.tags_only, limit=args.limit,
                  use_cache=not args.no_cache, progress=not args.no_progress,
                  db_path=Path(args.db))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
