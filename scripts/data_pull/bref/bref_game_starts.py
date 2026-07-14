"""Basketball-Reference season games-started scraper -> stg_bref_game_starts.

Pulls one league totals page per season (season S -> bref ending-year page
NBA_{S+1}_totals) and records games (G) and games-started (GS) per bref_id. This
is a POST-HOC, season-end fact used ONLY to define pre-2017 6MOTY training groups,
where the nba_api started flag is untrustworthy (it over-labels every appearance)
and no as-of start signal exists. It is deliberately not as-of: pre-2017 is
training-only, never served, so using season-end role to clean known history is
not a leak (the label is vote share, not role) and introduces no train/serve skew
(the served era, 2017+, stays on the as-of started-ratio gate). Keyed by
(bref_id, season). Never allocates player_id and never writes players.

Traded players appear as several team rows plus a combined row (team 'TOT' or
'NTM'); the combined row carries the season totals and is preferred.

Run:  uv run python -m scripts.data_pull.bref.bref_game_starts
      uv run python -m scripts.data_pull.bref.bref_game_starts --from 1996 --to 2016
"""

from __future__ import annotations

import argparse
import logging
import re
import sys

from bs4 import BeautifulSoup

try:
    from scripts.common.db import connect, upsert, utc_now
except ImportError:  # pragma: no cover
    from db import connect, upsert, utc_now  # type: ignore

try:
    from scripts.data_pull.bref.bref_common import fetch_html, uncomment_tables
except ImportError:  # pragma: no cover
    from bref_common import fetch_html, uncomment_tables  # type: ignore

log = logging.getLogger("bref_game_starts")

_COMBINED_RE = re.compile(r"^(TOT|\dTM|NTM)$")


def _ensure_schema(conn) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS stg_bref_game_starts (
            bref_id   TEXT    NOT NULL,
            season    INTEGER NOT NULL,
            g         INTEGER,
            gs        INTEGER,
            pulled_at TEXT,
            PRIMARY KEY (bref_id, season)
        )
        """
    )
    conn.commit()


def season_slug(season: int) -> str:
    """bref labels a season by its ending year: our starting-year S -> NBA_{S+1}."""
    return f"leagues/NBA_{season + 1}_totals"


def _cell(tr, *stats):
    for s in stats:
        el = tr.find(["td", "th"], attrs={"data-stat": s})
        if el is not None:
            return el
    return None


def _int(el):
    if el is None:
        return None
    txt = el.get_text(strip=True)
    if not txt:
        return None
    try:
        return int(txt)
    except ValueError:
        return None


def parse_totals(html: str, season: int, pulled_at: str) -> list[dict]:
    soup = BeautifulSoup(uncomment_tables(html), "html.parser")
    table = soup.find("table", id="totals_stats")
    if table is None:
        for t in soup.find_all("table"):
            if t.find(["td", "th"], attrs={"data-stat": "gs"}) is not None:
                table = t
                break
    if table is None:
        raise ValueError(f"totals table not found for season {season}")

    body = table.find("tbody") or table
    by_id: dict[str, list[dict]] = {}
    for tr in body.find_all("tr"):
        cls = tr.get("class") or []
        if "thead" in cls:
            continue
        pcell = _cell(tr, "player", "name_display")
        if pcell is None:
            continue
        bref_id = pcell.get("data-append-csv")
        if not bref_id:
            a = pcell.find("a", href=True)
            if a:
                m = re.search(r"/players/[a-z]/([^.]+)\.html", a["href"])
                bref_id = m.group(1) if m else None
        if not bref_id:
            continue
        tcell = _cell(tr, "team_id", "team_name_abbr", "team")
        team = tcell.get_text(strip=True) if tcell is not None else ""
        by_id.setdefault(bref_id, []).append({
            "team": team,
            "g": _int(_cell(tr, "g", "games")),
            "gs": _int(_cell(tr, "gs", "games_started")),
        })

    out: list[dict] = []
    for bref_id, rows in by_id.items():
        combined = [r for r in rows if _COMBINED_RE.match(r["team"])]
        chosen = combined[0] if combined else max(
            rows, key=lambda r: (r["g"] if r["g"] is not None else -1))
        out.append({
            "bref_id": bref_id,
            "season": season,
            "g": chosen["g"],
            "gs": chosen["gs"],
            "pulled_at": pulled_at,
        })
    return out


def scrape(db_path: str, lo: int, hi: int, force_refresh: bool = False) -> dict:
    pulled_at = utc_now()
    conn = connect(db_path)
    _ensure_schema(conn)
    seasons = list(range(lo, hi + 1))
    try:
        from tqdm import tqdm
        iterator = tqdm(seasons, desc="bref totals")
    except ImportError:  # pragma: no cover
        iterator = seasons

    summary = {"seasons": 0, "rows": 0, "errors": []}
    for season in iterator:
        slug = season_slug(season)
        try:
            html = fetch_html(slug, force_refresh=force_refresh)
        except FileNotFoundError:
            log.warning("totals page not found: %s", slug)
            summary["errors"].append((season, "404"))
            continue
        except Exception as exc:  # log-and-continue
            log.error("fetch failed for %s: %s", slug, exc)
            summary["errors"].append((season, str(exc)))
            continue
        try:
            rows = parse_totals(html, season, pulled_at)
        except Exception as exc:
            log.error("parse failed for season %d: %s", season, exc)
            summary["errors"].append((season, f"parse:{exc}"))
            continue
        if rows:
            upsert(conn, "stg_bref_game_starts", rows, ["bref_id", "season"])
            summary["rows"] += len(rows)
            summary["seasons"] += 1
    conn.commit()
    conn.close()
    return summary


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Scrape bref season GS -> stg_bref_game_starts")
    p.add_argument("--db", default="data/awards.db")
    p.add_argument("--from", dest="lo", type=int, default=1996)
    p.add_argument("--to", dest="hi", type=int, default=2016)
    p.add_argument("--force-refresh", action="store_true")
    args = p.parse_args(argv)
    summary = scrape(args.db, args.lo, args.hi, args.force_refresh)
    log.info("done: %d rows across %d seasons; %d errors",
             summary["rows"], summary["seasons"], len(summary["errors"]))
    if summary["errors"]:
        log.warning("errors: %s", summary["errors"][:20])
    return 0


if __name__ == "__main__":
    sys.exit(main())
