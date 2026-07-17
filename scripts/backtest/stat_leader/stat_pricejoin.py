"""Price join for the stat-leader books (pts / reb / ast), the analogue of
backtest_pricejoin.load_tradeable. Per (book, season) it resolves each stat-leader
market to a model player_id, pairs the YES/NO legs, and attaches the D+1 execution
price under the SAME strict no-lookahead cutoff as the voter join: the chosen
execution timestamp must be strictly after the snapshot's inclusive feature cutoff
(price stamps 00:00 UTC, so the first valid price is stamped >= D+1). Snapshot-
candidates with no price at or after D+1 within tolerance DROP OUT rather than
reaching forward to a later price.

Routing. Stat markets are reclassified into the award tags PTS_LEADER / REB_LEADER
/ AST_LEADER. Selection routes on the tag. Until that reclassification lands the
markets still carry award='OTHER', so an interim slug fallback selects the OTHER
subset by stat family and season context, mirroring pm_clob_targeted. Once the tag
re-run has happened the fallback is never reached and behaviour is identical.

Identity is additive and read-only. pm_candidates.player_id is not populated for
these markets, so the player is parsed from the market slug and resolved via the
EXISTING players table and aliases.yaml, mirroring promote_award.py: alias hint,
then an unambiguous single normalised-name match. Ambiguous or unmatched names are
reported and dropped, never guessed. resolve_identity.py is NEVER run; only its
pure normalise_name / load_aliases are reused. The DB is read only.

Fair value (calibrated P(lead)) and the CVaR pool are NOT carried on the leg: they
live on the StatSamples object keyed by player_id, the single source of truth for
fair value. The master's Phase 3 joins samples to these legs by player_id and
builds the tradeable_mask from per-snapshot price presence.

  load_tradeable_stat(conn, book, season, snapshot_dates) -> StatJoin
    .frames             { snapshot_date: SnapshotFrame }   (mirrors the voter dict)
    .market_player_ids  [player_id]   resolved season-level set (the union input)
    .resolved           { market_id: player_id }
    .unresolved         [ (market_id, parsed_name, reason, slug) ]
    .routing            "tag" | "slug-fallback"
  SnapshotFrame: .date, .candidates (list of CandidateLeg)
  CandidateLeg: player_id, name, yes_cid, no_cid, yes_exec_price, no_exec_price,
                exec_timestamp
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from scripts.data_pull.identity.resolve_identity import load_aliases, normalise_name

STAT_BOOK_AWARD = {"pts": "PTS_LEADER", "reb": "REB_LEADER", "ast": "AST_LEADER", "stl": "STL_LEADER", "blk": "BLK_LEADER"}

# Interim slug fallback only (real routing is the award tag above). Mirrors the
# pm_clob_targeted family vocabulary. A market belongs to a book if its season-
# context text matches the book's specific family, or matches the generic
# lead-the-league frame and carries the book's stat token.
_FAMILY_RX = {
    "pts": re.compile(r"scoring (title|champion|leader)|most points|lead(s|er)?.{0,15}\bin scoring\b|"
                      r"lead(s|er)?.{0,12}\bpoints\b|top scorer|ppg leader", re.I),
    "reb": re.compile(r"rebound|\brpg\b|top rebounder", re.I),
    "ast": re.compile(r"assist|\bapg\b", re.I),
    "stl": re.compile(r"steal|\bspg\b|top thief", re.I),
    "blk": re.compile(r"block|\bbpg\b|top shot ?blocker", re.I),
}
_GENERIC_RX = re.compile(r"lead the (nba|league) in|league leader", re.I)
_STAT_TOKEN_RX = {
    "pts": re.compile(r"\b(points|scoring|ppg)\b", re.I),
    "reb": re.compile(r"\b(rebounds?|rebounding|rpg)\b", re.I),
    "ast": re.compile(r"\b(assists?|apg)\b", re.I),
    "stl": re.compile(r"\b(steals?|spg)\b", re.I),
    "blk": re.compile(r"\b(blocks?|bpg)\b", re.I),
}
_RX_SINGLE_GAME = re.compile(r"\btonight\b|\bgame[ -]?\d\b|\btoday\b", re.I)
_RX_SERIES = re.compile(r"\bseries\b|\bin the .*finals|conference finals|playoff|\bvs\b|"
                        r"top scorer in|top rebounder|most .* in the .*finals", re.I)
_RX_TITLE_OUTCOME = re.compile(r"win the .*(finals|conference|championship|title)", re.I)

# will-<player>-lead-the-nba-in-<stat>; capture the player chunk before the verb.
_SLUG_PLAYER_RX = re.compile(r"^will-(?P<name>.+?)-(?:leads?|to-lead|win)-", re.I)
_DESC_PLAYER_RX = re.compile(r"^will\s+(?P<name>.+?)\s+(?:leads?|to\s+lead|win)\b", re.I)


@dataclass
class CandidateLeg:
    player_id: int
    name: str
    yes_cid: str
    no_cid: str | None
    yes_exec_price: float
    no_exec_price: float
    exec_timestamp: str


@dataclass
class SnapshotFrame:
    date: str
    candidates: list = field(default_factory=list)


@dataclass
class StatJoin:
    book: str
    season: int
    frames: dict
    market_player_ids: list
    resolved: dict
    unresolved: list
    routing: str


def _rows(conn, sql, params=()):
    return [dict(r) for r in conn.execute(sql, params).fetchall()]


def _norm_text(*parts):
    joined = " ".join(str(p) for p in parts if p)
    joined = joined.replace("-", " ").replace("_", " ")
    return re.sub(r"\s+", " ", joined).strip().lower()


def _is_season_context(text):
    if _RX_SINGLE_GAME.search(text):
        return False
    if _RX_TITLE_OUTCOME.search(text):
        return True
    if _RX_SERIES.search(text):
        return False
    return True


def _belongs_to_book(market, book):
    text = _norm_text(market["slug"], market["description"])
    if not _is_season_context(text):
        return False
    if _FAMILY_RX[book].search(text):
        return True
    if _GENERIC_RX.search(text) and _STAT_TOKEN_RX[book].search(text):
        return True
    return False


def _parse_player(slug, description=""):
    m = _SLUG_PLAYER_RX.match(slug or "")
    if m:
        return m.group("name").replace("-", " ").strip()
    m = _DESC_PLAYER_RX.match(description or "")
    if m:
        return m.group("name").strip()
    return None


_FIELD_RX = re.compile(r"\b(?:another|any\s+other|other)\s+player\b|\bthe\s+field\b", re.I)


def _squash(name):
    return normalise_name(name).replace("'", "").replace("\u2019", "").replace(".", "")


def _load_name_index(conn):
    idx = {}
    for r in conn.execute("SELECT player_id, name FROM players"):
        idx.setdefault(_squash(r["name"]), []).append((r["player_id"], r["name"]))
    return idx


def _select_markets(conn, book, season):
    tag = STAT_BOOK_AWARD[book]
    recs = _rows(conn,
                 "SELECT market_id, COALESCE(slug,'') AS slug, "
                 "COALESCE(description,'') AS description, COALESCE(award,'') AS award "
                 "FROM pm_markets WHERE season=? AND award=?", (season, tag))
    if recs:
        return recs, "tag"
    other = _rows(conn,
                  "SELECT market_id, COALESCE(slug,'') AS slug, "
                  "COALESCE(description,'') AS description, COALESCE(award,'') AS award "
                  "FROM pm_markets WHERE season=? AND award='OTHER'", (season,))
    return [m for m in other if _belongs_to_book(m, book)], "slug-fallback"


def resolve_markets(conn, book, season, manual=None, aliases=None):
    """Resolve the book's markets for the season to player_ids, read-only against
    players. manual is an optional {market_id: player_id} disambiguation pin.
    aliases overrides the on-disk seed (keyed by normalised nickname); None loads
    aliases.yaml. Returns (recs, resolved{market_id:pid}, unresolved[...], routing)."""
    manual = manual or {}
    recs, routing = _select_markets(conn, book, season)
    aliases = load_aliases() if aliases is None else aliases
    aliases = {_squash(k): v for k, v in aliases.items()}
    name_idx = _load_name_index(conn)
    resolved, unresolved = {}, []
    for m in recs:
        mid = m["market_id"]
        if mid in manual:
            resolved[mid] = int(manual[mid])
            continue
        raw = _parse_player(m["slug"], m["description"])
        if not raw:
            unresolved.append((mid, None, "could not parse player from slug", m["slug"]))
            continue
        if _FIELD_RX.search(raw):
            unresolved.append((mid, raw, "field leg (residual mass, not a player)", m["slug"]))
            continue
        norm = _squash(raw)
        if norm in aliases:
            norm = _squash(aliases[norm])
        cands = name_idx.get(norm, [])
        if len(cands) == 1:
            resolved[mid] = cands[0][0]
        elif not cands:
            unresolved.append((mid, raw, "no normalised-name match", m["slug"]))
        else:
            names = sorted({c[1] for c in cands})
            unresolved.append((mid, raw, f"ambiguous ({len(cands)} namesakes: {names})", m["slug"]))
    return recs, resolved, unresolved, routing


def _side_of(outcome_side, outcome):
    s = (outcome_side or outcome or "").strip().upper()
    return "YES" if s == "YES" else ("NO" if s == "NO" else "?")


def _legs_for_market(conn, market_id):
    yes_cid = no_cid = None
    for r in _rows(conn, "SELECT candidate_id, candidate_name, outcome, outcome_side "
                         "FROM pm_candidates WHERE market_id=?", (market_id,)):
        side = _side_of(r.get("outcome_side"), r.get("outcome"))
        if side == "YES" and yes_cid is None:
            yes_cid = r["candidate_id"]
        elif side == "NO" and no_cid is None:
            no_cid = r["candidate_id"]
    return yes_cid, no_cid


def _series(conn, candidate_id):
    return [(r["timestamp"], r["hist_price"]) for r in _rows(
        conn, "SELECT timestamp, hist_price FROM pm_prices "
              "WHERE price_type='history_agg' AND candidate_id=? ORDER BY timestamp",
        (candidate_id,))]


def _parse_ts(ts):
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def _first_price_at_or_after(rows, cutoff_dt, tol_dt):
    for ts, px in rows:
        t = _parse_ts(ts)
        if t >= cutoff_dt:
            if t <= tol_dt:
                return float(px), ts
            return None, None
    return None, None


def load_tradeable_stat(conn, book, season, snapshot_dates, exec_lag_days=1,
                        forward_tol_days=6, manual=None, aliases=None):
    """Assemble tradeable stat-leader legs per snapshot under the D+1 no-lookahead
    seal. snapshot_dates is the list of model snapshot dates to price against
    (supplied by the caller, e.g. the samples producer's MC snapshot grid).
    Returns a StatJoin. Read-only against the DB."""
    if book not in STAT_BOOK_AWARD:
        raise ValueError(f"unknown stat book {book!r}; expected one of {sorted(STAT_BOOK_AWARD)}")
    recs, resolved, unresolved, routing = resolve_markets(conn, book, season, manual, aliases)

    legs = {}  # player_id -> (name, yes_cid, no_cid, yes_series)
    seen_pid = {}
    name_by_pid = {}
    for r in conn.execute("SELECT player_id, name FROM players"):
        name_by_pid[r["player_id"]] = r["name"]
    for m in recs:
        mid = m["market_id"]
        pid = resolved.get(mid)
        if pid is None:
            continue
        yes_cid, no_cid = _legs_for_market(conn, mid)
        if yes_cid is None:
            unresolved.append((mid, name_by_pid.get(pid), "no YES candidate leg", m["slug"]))
            continue
        yrows = _series(conn, yes_cid)
        if pid in seen_pid and not yrows:
            continue
        if pid in seen_pid and legs.get(pid) and legs[pid][3]:
            continue  # keep the first market for this pid that carries prices
        legs[pid] = (name_by_pid.get(pid, str(pid)), yes_cid, no_cid, yrows)
        seen_pid[pid] = mid

    frames = {}
    for snap in snapshot_dates:
        cutoff = _parse_ts(snap + "T00:00:00+00:00")
        exec_from = cutoff + timedelta(days=exec_lag_days)
        tol = exec_from + timedelta(days=forward_tol_days)
        for pid, (name, yes_cid, no_cid, yrows) in legs.items():
            if not yrows:
                continue
            ypx, yts = _first_price_at_or_after(yrows, exec_from, tol)
            if ypx is None:
                continue
            assert _parse_ts(yts) >= exec_from, f"seal violation {yts} < {exec_from}"
            frames.setdefault(snap, SnapshotFrame(date=snap)).candidates.append(
                CandidateLeg(player_id=pid, name=name, yes_cid=yes_cid, no_cid=no_cid,
                             yes_exec_price=ypx, no_exec_price=1.0 - ypx, exec_timestamp=yts))

    return StatJoin(book=book, season=season, frames=dict(sorted(frames.items())),
                    market_player_ids=sorted(legs.keys()), resolved=resolved,
                    unresolved=unresolved, routing=routing)
