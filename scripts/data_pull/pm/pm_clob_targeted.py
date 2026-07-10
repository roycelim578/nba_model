"""Targeted CLOB price backfill for stat-leader and secondary-award markets.

pm_clob.backfill_history restricts to _AWARD_FILTER = (MVP, DPOY, ROTY,
CHAMPIONSHIP). The stat-leader, Sixth Man, and MIP markets are all bucketed
award='OTHER', so that backfill never queried them: their empty price history is
a collection-scope gap, not a Polymarket gap. The token ids already exist in
pm_candidates (Gamma writes candidates for every NBA market), so this module
just points the SAME fetch-and-write path at a targeted OTHER subset.

Selection is by slug category (stat leaders, Sixth Man, MIP) and restricted to
season-long markets, excluding finals-series and single-game props that share
the vocabulary ('top scorer in nba finals', 'triple double tonight'). It reuses
pm_clob._fetch_history_one, _history_rows, and the pm_prices upsert verbatim, so
provenance (price_type='history_agg') and schema are identical to the award
backfill and nothing downstream needs to change.

Writes pm_prices, and (unless --no-season-backfill) fills pm_markets.season
where it is NULL, deriving the starting-year season from the earliest price
timestamp. Many OTHER slugs carry no year ('will-x-lead-the-nba-in-scoring'),
so Gamma left their season NULL, which would break the walk-forward fold key.
The backfill is a general idempotent pass over every null-season priced market
and only ever fills NULLs, never overwriting an existing season.

  uv run python -m scripts.data_pull.pm.pm_clob_targeted --limit 20 --yes-only    # cautious first pass
  uv run python -m scripts.data_pull.pm.pm_clob_targeted --yes-only                # stat leaders + 6M + MIP, season only
  uv run python -m scripts.data_pull.pm.pm_clob_targeted --families sixth_man,most_improved
  uv run python -m scripts.data_pull.pm.pm_clob_targeted --seasons 2024,2025 --yes-only --no-progress
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

from scripts.common.db import connect, upsert
from scripts.data_pull.pm.pm_clob import _fetch_history_one, _history_rows

DB_PATH = Path("data/awards.db")

FAMILY_PATTERNS: dict[str, str] = {
    "scoring_leader": r"scoring (title|champion|leader)|most points|"
                      r"lead(s|er)?.{0,15}\bin scoring\b|lead(s|er)?.{0,12}\bpoints\b|top scorer|ppg leader",
    "rebounds_leader": r"rebound|\brpg\b|top rebounder",
    "assists_leader": r"assist|\bapg\b",
    "blocks_leader": r"\bblock(s|ed|ing)?\b|\bbpg\b",
    "steals_leader": r"\bsteal(s|ing)?\b|\bspg\b",
    "generic_stat_leader": r"lead the (nba|league) in|league leader",
    "sixth_man": r"sixth[ -]?man|6th man|\b6moy\b",
    "most_improved": r"most improved|\bmip\b",
}
STAT_FAMILIES = ["scoring_leader", "rebounds_leader", "assists_leader",
                 "blocks_leader", "steals_leader", "generic_stat_leader"]

_RX_SINGLE_GAME = re.compile(r"\btonight\b|\bgame[ -]?\d\b|\btoday\b", re.I)
_RX_SERIES = re.compile(r"\bseries\b|\bin the .*finals|conference finals|playoff|\bvs\b|"
                        r"top scorer in|top rebounder|most .* in the .*finals", re.I)
_RX_TITLE_OUTCOME = re.compile(r"win the .*(finals|conference|championship|title)", re.I)


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


def _select_targets(rows, families, seasons, contexts, yes_only):
    """Pure selection: given candidate rows (dicts with market_id, candidate_id,
    outcome, slug, description, season, award), return the matching subset plus a
    per-family tally. Kept pure so it is unit-testable without the live API/DB."""
    compiled = [(f, re.compile(FAMILY_PATTERNS[f], re.I)) for f in families]
    selected, tally = [], {f: 0 for f in families}
    seen_family = {}
    for r in rows:
        if r["award"] != "OTHER":
            continue
        if seasons and r["season"] not in seasons:
            continue
        if yes_only and (r["outcome"] or "").lower() != "yes":
            continue
        text = _normalise(r["slug"], r["description"])
        if _context(text) not in contexts:
            continue
        matched = [f for f, rx in compiled if rx.search(text)]
        if not matched:
            continue
        selected.append(r)
        primary = matched[0]
        tally[primary] += 1
        seen_family[r["market_id"]] = primary
    return selected, tally


def _load_candidates(conn):
    rows = conn.execute(
        "SELECT c.market_id, c.candidate_id, c.outcome, "
        "COALESCE(m.slug,'') AS slug, COALESCE(m.description,'') AS description, "
        "m.season, COALESCE(m.award,'') AS award "
        "FROM pm_candidates c JOIN pm_markets m ON c.market_id = m.market_id"
    ).fetchall()
    return [dict(r) for r in rows]


def _season_from_iso(first_ts: str) -> int | None:
    """Derive the starting-year season from a market's earliest price timestamp.

    NBA futures for a season open no earlier than roughly July and resolve the
    following spring, so a first-price month >= 7 means the season is that
    calendar year, and Jan-Jun means the prior year (2025-04 -> 2024 season).
    This holds for stat-leader and award markets; it would misdate a next-season
    future that somehow opened in June, which this target set does not contain.
    """
    try:
        dt = datetime.fromisoformat(first_ts)
    except (TypeError, ValueError):
        return None
    return dt.year if dt.month >= 7 else dt.year - 1


def _backfill_seasons(conn) -> dict:
    """Fill pm_markets.season from the earliest price timestamp where it is NULL.

    General idempotent pass over every null-season market that has price history,
    not just this run's targets. Guarded with 'AND season IS NULL' so re-runs and
    already-dated markets are never touched."""
    rows = conn.execute(
        "SELECT p.market_id AS market_id, MIN(p.timestamp) AS first_ts "
        "FROM pm_prices p JOIN pm_markets m ON p.market_id = m.market_id "
        "WHERE m.season IS NULL AND p.hist_price IS NOT NULL "
        "GROUP BY p.market_id"
    ).fetchall()
    filled, unresolved = 0, []
    for r in rows:
        season = _season_from_iso(r["first_ts"])
        if season is None:
            unresolved.append(r["market_id"])
            continue
        cur = conn.execute(
            "UPDATE pm_markets SET season = ? WHERE market_id = ? AND season IS NULL",
            (season, r["market_id"]),
        )
        filled += cur.rowcount
    conn.commit()
    return {"null_season_priced_markets": len(rows), "seasons_filled": filled,
            "unresolved": unresolved}


def run(families, seasons, contexts, yes_only, limit, use_cache, progress,
        season_backfill=True, db_path: Path = DB_PATH) -> dict:
    conn = connect(db_path)
    try:
        rows = _load_candidates(conn)
        targets, tally = _select_targets(rows, families, seasons, contexts, yes_only)
        if limit:
            targets = targets[:limit]
        print(f"selected {len(targets)} token(s) across families: "
              f"{json.dumps({k: v for k, v in tally.items() if v}, sort_keys=True)}")
        if seasons:
            print(f"  seasons filter: {sorted(seasons)}")
        print(f"  contexts: {sorted(contexts)} | yes_only={yes_only} | limit={limit}")

        written, no_data, done = 0, [], 0
        fidelity_used: dict[str, int] = {}
        for r in targets:
            mid, tok = r["market_id"], r["candidate_id"]
            points, interval, fidelity = _fetch_history_one(tok, use_cache)
            if not points:
                no_data.append(tok)
            else:
                prows = _history_rows(mid, tok, points, interval, fidelity)
                upsert(conn, "pm_prices", prows, ["market_id", "candidate_id", "timestamp"])
                written += len(prows)
                label = f"{interval}/{fidelity}"
                fidelity_used[label] = fidelity_used.get(label, 0) + 1
            done += 1
            if progress and done % 25 == 0:
                print(f"  ... {done}/{len(targets)} tokens, {written} rows, "
                      f"{len(no_data)} empty", flush=True)

        season_report = _backfill_seasons(conn) if season_backfill else {"skipped": True}

        return {
            "tokens_selected": len(targets),
            "tokens_with_data": len(targets) - len(no_data),
            "tokens_no_data": len(no_data),
            "price_rows_written": written,
            "granularity_distribution": fidelity_used,
            "family_tally": {k: v for k, v in tally.items() if v},
            "season_backfill": season_report,
        }
    finally:
        conn.close()


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default=None,
                    help="comma list; default = stat leaders + sixth_man + most_improved")
    ap.add_argument("--seasons", default=None, help="comma list of starting years, e.g. 2024,2025")
    ap.add_argument("--contexts", default="season",
                    help="comma list of season,finals_series,single_game (default season)")
    ap.add_argument("--yes-only", action="store_true", help="only pull the Yes leg")
    ap.add_argument("--limit", type=int, default=None, help="cap tokens for a cautious first pass")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--no-progress", action="store_true")
    ap.add_argument("--no-season-backfill", action="store_true",
                    help="do not fill pm_markets.season from price timestamps")
    args = ap.parse_args(argv[1:])

    families = ([f.strip() for f in args.families.split(",")] if args.families
                else STAT_FAMILIES + ["sixth_man", "most_improved"])
    bad = [f for f in families if f not in FAMILY_PATTERNS]
    if bad:
        raise SystemExit(f"unknown families: {bad}; valid: {sorted(FAMILY_PATTERNS)}")
    seasons = {int(s) for s in args.seasons.split(",")} if args.seasons else None
    contexts = {c.strip() for c in args.contexts.split(",")}

    summary = run(families, seasons, contexts, args.yes_only, args.limit,
                  not args.no_cache, not args.no_progress,
                  season_backfill=not args.no_season_backfill)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
