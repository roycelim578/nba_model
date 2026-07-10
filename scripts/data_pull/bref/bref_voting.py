"""Basketball-Reference award voting history scraper.

Produces ``stg_award_voting`` rows (one per season/award/bref_id) from bref's
consolidated per-season awards pages at ``/awards/awards_<ENDING_YEAR>.html``.

CRITICAL SEASON CONVENTION
--------------------------
bref labels each season by its ENDING year. The page ``awards_2025.html`` is
the 2024-25 season. We store ``season`` as the STARTING year, so that page is
stored as ``season = 2024``. The rule is, exactly:

    stored_season = page_label_year - 1

A one-year misalignment here is invisible to eyeballing but silently corrupts
the model's walk-forward CV (train on <=T, test T+1). We therefore (a) only
ever convert in one place, ``page_label_to_season``, and (b) assert the
invariant on every row before writing.

Run:  uv run python -m scripts.data_pull.bref.bref_voting
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

from bs4 import BeautifulSoup

try:  # shared helper owned by the nba_api chat; fall back to local stopgap
    from scripts.common.db import connect, upsert, utc_now
except ImportError:  # pragma: no cover - import path depends on run context
    from db import connect, upsert, utc_now  # type: ignore

try:
    from scripts.data_pull.bref.bref_common import fetch_html, uncomment_tables
except ImportError:  # pragma: no cover
    from bref_common import fetch_html, uncomment_tables  # type: ignore

log = logging.getLogger("bref_voting")

# ----------------------------------------------------------------------------
# Award configuration
# ----------------------------------------------------------------------------
#
# award          : canonical string stored in stg_award_voting.award. The
#                  project (PROJECT.md, schema.sql) uses "ROTY"; bref's table id
#                  is "roy". We store the project's canonical name and map to the
#                  bref table id separately.
# table_id       : the HTML id of the per-award table inside the awards page
#                  (bref uses "div_<id>" wrapper / "<id>" table id).
# first_page_year: the EARLIEST page-label (ending) year that has this award's
#                  voting table. DPOY did not exist before the 1982-83 season,
#                  so its first page is awards_1983 (stored season=1982). MVP and
#                  ROY go back further; we floor the global pull at the 1980-81
#                  season (page 1981) per the brief, and let downstream choose a
#                  narrower usable window.
#
# Below-floor (award, season) pairs are EXPECTED-ABSENT: we do not fetch a
# per-award table for them, do not count them as failures, and never retry.

GLOBAL_FIRST_PAGE_YEAR = 1981  # 1980-81 season; brief's "1980+" floor

AWARDS: dict[str, dict] = {
    "MVP":  {"table_id": "mvp",  "first_page_year": GLOBAL_FIRST_PAGE_YEAR},
    "DPOY": {"table_id": "dpoy", "first_page_year": 1983},  # 1982-83 inception
    "ROTY": {"table_id": "roy",  "first_page_year": GLOBAL_FIRST_PAGE_YEAR},
}


def page_label_to_season(page_label_year: int) -> int:
    """Convert a bref page-label (ending) year to a stored STARTING-year season.

    This is the ONLY place the conversion happens. awards_2025 -> 2024.
    """
    return page_label_year - 1


# ----------------------------------------------------------------------------
# Parsing
# ----------------------------------------------------------------------------

# bref player links look like /players/j/jokicni01.html -> slug "jokicni01".
_BREF_SLUG_RE = re.compile(r"/players/[a-z]/([a-z0-9.]+)\.html", re.IGNORECASE)


@dataclass
class VotingRow:
    season: int            # STARTING year
    award: str             # canonical: MVP / DPOY / ROTY
    bref_id: str
    player_name_raw: str   # exactly as printed on bref, for audit
    first_place_votes: int | None
    total_points: float | None
    vote_share: float | None
    rank: int | None
    won_flag: int          # 1 for rank==1 else 0
    pulled_at: str


def _cell(row, data_stat: str):
    """Return the text of a cell by its data-stat attribute, or None."""
    el = row.find(["td", "th"], attrs={"data-stat": data_stat})
    if el is None:
        return None
    txt = el.get_text(strip=True)
    return txt if txt != "" else None


def _to_int(val):
    if val is None:
        return None
    try:
        return int(float(val.replace(",", "")))
    except (ValueError, AttributeError):
        return None


def _to_float(val):
    if val is None:
        return None
    try:
        return float(val.replace(",", ""))
    except (ValueError, AttributeError):
        return None


def parse_award_table(html: str, table_id: str, award: str, season: int, pulled_at: str) -> list[VotingRow]:
    """Parse one award's voting table out of an awards page.

    Returns [] if the table is not present (caller decides whether that is
    expected-absent or a genuine miss). bref's per-award table commonly carries
    columns with data-stat: rank, player, first_place (or votes_first),
    points_won, award_share. Column names have drifted across bref redesigns,
    so we probe a few known aliases per field rather than hardcoding one.
    """
    soup = BeautifulSoup(uncomment_tables(html), "html.parser")
    table = soup.find("table", id=table_id)
    if table is None:
        return []

    body = table.find("tbody") or table
    rows: list[VotingRow] = []

    # Known data-stat aliases across bref redesigns.
    fp_keys = ("first_place", "votes_first", "first_place_votes")
    pts_keys = ("points_won", "votes")
    share_keys = ("award_share", "share")

    def first_present(tr, keys):
        for k in keys:
            v = _cell(tr, k)
            if v is not None:
                return v
        return None

    for tr in body.find_all("tr"):
        # Skip header/spacer rows.
        if tr.get("class") and any(c in ("thead", "over_header") for c in tr.get("class")):
            continue

        player_cell = tr.find(["td", "th"], attrs={"data-stat": "player"})
        if player_cell is None:
            continue

        link = player_cell.find("a", href=True)
        if link is None:
            # Row without a player link (e.g. a footer); skip.
            continue
        m = _BREF_SLUG_RE.search(link["href"])
        if not m:
            continue
        bref_id = m.group(1)
        player_name_raw = link.get_text(strip=True)

        rank = _to_int(_cell(tr, "rank")) or _to_int(_cell(tr, "ranker"))
        fpv = _to_int(first_present(tr, fp_keys))
        pts = _to_float(first_present(tr, pts_keys))
        share = _to_float(first_present(tr, share_keys))

        rows.append(
            VotingRow(
                season=season,
                award=award,
                bref_id=bref_id,
                player_name_raw=player_name_raw,
                first_place_votes=fpv,
                total_points=pts,
                vote_share=share,
                rank=rank,
                won_flag=1 if rank == 1 else 0,
                pulled_at=pulled_at,
            )
        )
    return rows


def renormalise_shares(rows: list[VotingRow]) -> list[VotingRow]:
    """Verify (season, award) shares sum to ~1.0; renormalise if not.

    bref provides Share directly. Occasionally it does not sum to exactly 1
    (rounding, or incomplete historical tallies). If the sum is positive and
    materially off 1.0, rescale proportionally. If shares are entirely missing,
    leave as-is (None) and let the caller's sanity check flag the season.
    """
    present = [r for r in rows if r.vote_share is not None]
    if not present:
        return rows
    total = sum(r.vote_share for r in present)
    if total <= 0:
        return rows
    if abs(total - 1.0) > 1e-6:
        for r in present:
            r.vote_share = r.vote_share / total
    return rows


# ----------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------

def seasons_for_award(award: str, last_page_year: int) -> list[int]:
    """Page-label (ending) years to fetch for an award, floor to last_page_year.

    Below-floor years are simply not produced, so they are never fetched.
    """
    floor = AWARDS[award]["first_page_year"]
    return list(range(floor, last_page_year + 1))


def scrape_voting(last_page_year: int, db_path: str, force_refresh: bool = False) -> dict:
    """Scrape all awards across all in-range seasons; upsert stg_award_voting.

    One network fetch per season page (the page carries all awards), reused
    across the three award table parses for that season.
    """
    pulled_at = utc_now()
    conn = connect(db_path)

    summary = {"rows": 0, "season_award_ok": 0, "renormalised": 0, "missing_table": [], "fetch_errors": []}

    # All page-label years that any award needs (union), so we fetch each page once.
    all_years = sorted({y for a in AWARDS for y in seasons_for_award(a, last_page_year)})

    for page_year in all_years:
        season = page_label_to_season(page_year)
        slug = f"awards/awards_{page_year}"
        try:
            html = fetch_html(slug, force_refresh=force_refresh)
        except FileNotFoundError:
            # Whole-season page genuinely absent. Only unexpected below ~1981.
            log.warning("awards page %s not found (page-year %d); skipping", slug, page_year)
            summary["fetch_errors"].append((slug, "404"))
            continue
        except Exception as exc:  # log-and-continue per brief
            log.error("fetch failed for %s: %s", slug, exc)
            summary["fetch_errors"].append((slug, str(exc)))
            continue

        for award, cfg in AWARDS.items():
            # Skip awards that did not exist in this season (expected-absent).
            if page_year < cfg["first_page_year"]:
                continue

            rows = parse_award_table(html, cfg["table_id"], award, season, pulled_at)
            if not rows:
                # Table absent on a page where we EXPECT it -> genuine miss.
                log.warning("no %s table on %s (season=%d)", award, slug, season)
                summary["missing_table"].append((award, season))
                continue

            before = [r.vote_share for r in rows]
            rows = renormalise_shares(rows)
            if [r.vote_share for r in rows] != before:
                summary["renormalised"] += 1

            # Assert the season convention on every row before writing.
            for r in rows:
                assert r.season == page_year - 1, (
                    f"season convention violated: stored {r.season} for page {page_year}"
                )

            n = upsert(
                conn,
                "stg_award_voting",
                [asdict(r) for r in rows],
                conflict_keys=["season", "award", "bref_id"],
            )
            summary["rows"] += n
            summary["season_award_ok"] += 1

    conn.close()
    return summary


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Scrape bref award voting history -> stg_award_voting")
    p.add_argument("--db", default="data/awards.db")
    p.add_argument(
        "--last-page-year",
        type=int,
        default=datetime.now(UTC).year,
        help="Most recent bref page-label (ending) year to pull, e.g. 2026 for 2025-26.",
    )
    p.add_argument("--force-refresh", action="store_true", help="Bypass HTML cache.")
    args = p.parse_args(argv)

    summary = scrape_voting(args.last_page_year, args.db, args.force_refresh)
    log.info(
        "done: %d rows across %d (season,award) groups; %d renormalised; "
        "%d expected-present tables missing; %d fetch errors",
        summary["rows"], summary["season_award_ok"], summary["renormalised"],
        len(summary["missing_table"]), len(summary["fetch_errors"]),
    )
    if summary["missing_table"]:
        log.warning("missing expected tables: %s", summary["missing_table"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
