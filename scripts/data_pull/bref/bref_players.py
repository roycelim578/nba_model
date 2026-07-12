"""Basketball-Reference player reference/bio scraper.

For every player appearing in any voting table (i.e. every distinct bref_id in
stg_award_voting), scrape their bref player page for bio fields and write
stg_bref_players. This is what lets the downstream resolution component create
"bref-only" players who predate reliable nba_api coverage.

Keyed entirely by bref_id. We NEVER allocate player_id (that is resolution's
job) and we NEVER write the canonical players table.

Run:  uv run python -m scripts.data_pull.bref.bref_players
(Run after bref_voting, since it reads the set of bref_ids from stg_award_voting.)
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from dataclasses import asdict, dataclass

from bs4 import BeautifulSoup

try:
    from scripts.common.db import connect, upsert, utc_now
except ImportError:  # pragma: no cover
    from db import connect, upsert, utc_now  # type: ignore

try:
    from scripts.data_pull.bref.bref_common import fetch_html, uncomment_tables
except ImportError:  # pragma: no cover
    from bref_common import fetch_html, uncomment_tables  # type: ignore

log = logging.getLogger("bref_players")


@dataclass
class PlayerRow:
    bref_id: str
    name: str
    position: str | None
    dob: str | None           # ISO date YYYY-MM-DD
    first_season: int | None  # STARTING year of earliest season
    last_season: int | None   # STARTING year of latest season
    pulled_at: str


def player_slug_to_page(bref_id: str) -> str:
    """bref player pages live under /players/<first-letter-of-id>/<id>.html."""
    return f"players/{bref_id[0]}/{bref_id}"


_POS_RE = re.compile(r"Position:\s*([A-Za-z ,and-]+)")
_DOB_ATTR_RE = re.compile(r"data-birth=\"(\d{4}-\d{2}-\d{2})\"")


def _parse_dob(soup: BeautifulSoup) -> str | None:
    # bref exposes birth date on a <span id="necro-birth" data-birth="YYYY-MM-DD">.
    el = soup.find("span", id="necro-birth")
    if el and el.has_attr("data-birth"):
        return el["data-birth"]
    return None


def _parse_position(soup: BeautifulSoup) -> str | None:
    # Position appears in the meta block, e.g. "Position: Point Guard ...".
    meta = soup.find("div", id="meta")
    if not meta:
        return None
    text = meta.get_text(" ", strip=True)
    m = _POS_RE.search(text)
    if not m:
        return None
    pos = m.group(1).strip().rstrip(",").strip()
    # Normalise wordy positions to the abbreviations the schema expects where
    # obvious; otherwise leave the bref text for resolution to map.
    mapping = {
        "Point Guard": "PG", "Shooting Guard": "SG", "Small Forward": "SF",
        "Power Forward": "PF", "Center": "C",
    }
    # Take the first listed position token if multiple.
    first = pos.split(" and ")[0].split(",")[0].strip()
    return mapping.get(first, first)


def _parse_seasons(soup: BeautifulSoup) -> tuple[int | None, int | None]:
    """Read first/last STARTING-year season from the per-game table.

    The per-game table's season column shows bref's "YYYY-YY" season label
    (e.g. "2024-25"); the leading 4-digit year IS the starting year, so no
    minus-one is needed here (unlike the awards page label). We read min/max of
    those leading years across the player's rows.
    """
    soup2 = BeautifulSoup(uncomment_tables(str(soup)), "html.parser")
    table = soup2.find("table", id="per_game") or soup2.find("table", id="per_game_stats")
    if table is None:
        return (None, None)
    years: list[int] = []
    body = table.find("tbody") or table
    for tr in body.find_all("tr"):
        cell = tr.find(["th", "td"], attrs={"data-stat": "year_id"}) or tr.find(
            ["th", "td"], attrs={"data-stat": "season"}
        )
        if cell is None:
            continue
        txt = cell.get_text(strip=True)
        m = re.match(r"(\d{4})", txt)
        if m:
            years.append(int(m.group(1)))
    if not years:
        return (None, None)
    return (min(years), max(years))


def _parse_name(soup: BeautifulSoup, fallback: str) -> str:
    h1 = soup.find("h1")
    if h1:
        name = h1.get_text(strip=True)
        if name:
            return name
    return fallback


def parse_player_page(html: str, bref_id: str, pulled_at: str) -> PlayerRow:
    soup = BeautifulSoup(html, "html.parser")
    first_season, last_season = _parse_seasons(soup)
    return PlayerRow(
        bref_id=bref_id,
        name=_parse_name(soup, fallback=bref_id),
        position=_parse_position(soup),
        dob=_parse_dob(soup),
        first_season=first_season,
        last_season=last_season,
        pulled_at=pulled_at,
    )


def distinct_bref_ids(conn) -> list[str]:
    cur = conn.execute("SELECT DISTINCT bref_id FROM stg_award_voting ORDER BY bref_id")
    return [r["bref_id"] for r in cur]


def scrape_players(db_path: str, force_refresh: bool = False) -> dict:
    pulled_at = utc_now()
    conn = connect(db_path)
    ids = distinct_bref_ids(conn)

    # Optional progress bar; degrade gracefully if tqdm missing.
    try:
        from tqdm import tqdm
        iterator = tqdm(ids, desc="players")
    except ImportError:  # pragma: no cover
        iterator = ids

    summary = {"players": 0, "errors": []}
    for bref_id in iterator:
        slug = player_slug_to_page(bref_id)
        try:
            html = fetch_html(slug, force_refresh=force_refresh)
        except FileNotFoundError:
            log.warning("player page not found: %s", slug)
            summary["errors"].append((bref_id, "404"))
            continue
        except Exception as exc:  # log-and-continue per brief
            log.error("fetch failed for %s: %s", slug, exc)
            summary["errors"].append((bref_id, str(exc)))
            continue

        try:
            row = parse_player_page(html, bref_id, pulled_at)
        except Exception as exc:  # parse failure on one player must not abort run
            log.error("parse failed for %s: %s", bref_id, exc)
            summary["errors"].append((bref_id, f"parse:{exc}"))
            continue

        upsert(conn, "stg_bref_players", [asdict(row)], ["bref_id"])
        summary["players"] += 1

    conn.close()
    return summary


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Scrape bref player bios -> stg_bref_players")
    p.add_argument("--db", default="data/awards.db")
    p.add_argument("--force-refresh", action="store_true")
    args = p.parse_args(argv)

    summary = scrape_players(args.db, args.force_refresh)
    log.info("done: %d players written; %d errors", summary["players"], len(summary["errors"]))
    if summary["errors"]:
        log.warning("errors: %s", summary["errors"][:20])
    return 0


if __name__ == "__main__":
    sys.exit(main())
