"""Basketball-Reference to nba_api identity crosswalk (normalised name).

The `players` table does not bridge bref_id to nba_api_id for the general pool
(bref-origin and nba-origin rows were never merged), so bref-keyed GS cannot
attach to the nba-keyed box pool. This builds that bridge by normalised name:
bref names (re-parsed from the cached totals pages) matched to `stg_nba_players`
names, restricted to nba_api_ids actually present in the pre-2017 box pool to hold
collisions down. Only unambiguous 1:1 matches (exactly one bref_id and one
nba_api_id per normalised name) are written to `stg_bref_nba_crosswalk`; ambiguous
names fail closed and are reported.

This crosswalk is used ONLY to define pre-2017 6MOTY training groups via post-hoc
bref GS. It is training-only, never served, so it introduces no leak or train/serve
skew. Names are normalised by lowercasing, stripping accents and punctuation, and
dropping Jr/Sr/roman-numeral suffixes, which is what bridges 'J.R. Smith' (bref) to
'JR Smith' (nba).

Run:  uv run python -m scripts.data_pull.bref.bref_nba_crosswalk
The tail of the run is the acceptance report; every pre-2017 winner must be covered
and clear GS/G < 0.5 before the filter is allowed to use this.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import unicodedata
from collections import defaultdict

from bs4 import BeautifulSoup

try:
    from scripts.common.db import connect, upsert, utc_now
except ImportError:  # pragma: no cover
    from db import connect, upsert, utc_now  # type: ignore

try:
    from scripts.data_pull.bref.bref_common import fetch_html, uncomment_tables
except ImportError:  # pragma: no cover
    from bref_common import fetch_html, uncomment_tables  # type: ignore

log = logging.getLogger("bref_nba_crosswalk")

SEASONS = list(range(1996, 2017))
_SUFFIX = {"jr", "sr", "ii", "iii", "iv", "v"}


def _ensure_schema(conn) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS stg_bref_nba_crosswalk (
            bref_id    TEXT    NOT NULL PRIMARY KEY,
            nba_api_id INTEGER NOT NULL,
            name_norm  TEXT,
            pulled_at  TEXT
        )
        """
    )
    conn.commit()


def norm_name(name: str | None) -> str:
    if not name:
        return ""
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    s = s.lower().replace(".", "").replace("'", "").replace("`", "")
    s = re.sub(r"[^a-z ]", " ", s)
    toks = s.split()
    while toks and toks[-1] in _SUFFIX:
        toks.pop()
    return " ".join(toks)


def _season_slug(season: int) -> str:
    return f"leagues/NBA_{season + 1}_totals"


def _bref_names() -> dict[str, str]:
    """bref_id -> display name, re-parsed from cached totals pages (no network)."""
    out: dict[str, str] = {}
    for season in SEASONS:
        try:
            html = fetch_html(_season_slug(season))
        except Exception as exc:  # a missing cache page is non-fatal for the crosswalk
            log.warning("totals cache miss for %d: %s", season, exc)
            continue
        soup = BeautifulSoup(uncomment_tables(html), "html.parser")
        table = soup.find("table", id="totals_stats")
        if table is None:
            for t in soup.find_all("table"):
                if t.find(["td", "th"], attrs={"data-stat": "gs"}) is not None:
                    table = t
                    break
        if table is None:
            continue
        body = table.find("tbody") or table
        for tr in body.find_all("tr"):
            if "thead" in (tr.get("class") or []):
                continue
            pcell = (tr.find(["td", "th"], attrs={"data-stat": "player"})
                     or tr.find(["td", "th"], attrs={"data-stat": "name_display"}))
            if pcell is None:
                continue
            bref_id = pcell.get("data-append-csv")
            if not bref_id:
                a = pcell.find("a", href=True)
                m = re.search(r"/players/[a-z]/([^.]+)\.html", a["href"]) if a else None
                bref_id = m.group(1) if m else None
            if bref_id and bref_id not in out:
                out[bref_id] = pcell.get_text(strip=True)
    return out


def _nba_names(conn) -> dict[int, str]:
    """nba_api_id -> name, restricted to ids present in the pre-2017 box pool."""
    rows = conn.execute(
        """
        SELECT np.nba_api_id nid, np.name nm
        FROM stg_nba_players np
        WHERE np.nba_api_id IN (
            SELECT DISTINCT nba_api_id FROM stg_nba_box_asof WHERE season <= 2016
        )
        """
    ).fetchall()
    return {r["nid"]: r["nm"] for r in rows}


def build(conn) -> dict:
    _ensure_schema(conn)
    pulled_at = utc_now()

    bref = _bref_names()
    nba = _nba_names(conn)

    bref_by_norm: dict[str, set] = defaultdict(set)
    for bid, nm in bref.items():
        bref_by_norm[norm_name(nm)].add(bid)
    nba_by_norm: dict[str, set] = defaultdict(set)
    for nid, nm in nba.items():
        nba_by_norm[norm_name(nm)].add(nid)

    rows = []
    ambiguous = []
    for nkey, bids in bref_by_norm.items():
        if not nkey:
            continue
        nids = nba_by_norm.get(nkey)
        if not nids:
            continue
        if len(bids) == 1 and len(nids) == 1:
            rows.append({
                "bref_id": next(iter(bids)),
                "nba_api_id": next(iter(nids)),
                "name_norm": nkey,
                "pulled_at": pulled_at,
            })
        else:
            ambiguous.append((nkey, sorted(bids), sorted(nids)))

    if rows:
        upsert(conn, "stg_bref_nba_crosswalk", rows, ["bref_id"])
    conn.commit()
    return {"matched": len(rows), "ambiguous": ambiguous,
            "bref_total": len(bref), "nba_pool": len(nba)}


def report(conn) -> None:
    xwalk = {r["nba_api_id"]: r["bref_id"] for r in conn.execute(
        "SELECT nba_api_id, bref_id FROM stg_bref_nba_crosswalk")}
    gs = {(r["bref_id"], r["season"]): (r["g"], r["gs"]) for r in conn.execute(
        "SELECT bref_id, season, g, gs FROM stg_bref_game_starts")}

    print(f"\n{'season':6} {'winner':26} {'status'}")
    blockers = []
    for s in SEASONS:
        w = conn.execute(
            """SELECT p.name nm, p.nba_api_id nid FROM award_voting v
               JOIN players p ON p.player_id=v.player_id
               WHERE v.award='6MOTY' AND v.season=? AND v.won_flag=1""", (s,)).fetchone()
        if w is None:
            continue
        nm = (w["nm"] or "?")[:26]
        nid = w["nid"]
        bid = xwalk.get(nid)
        if nid is None or bid is None:
            print(f"{s:6} {nm:26} UNMAPPED  <-- BLOCKER")
            blockers.append((s, nm, "uncrosswalked"))
            continue
        g, gsv = gs.get((bid, s), (None, None))
        if not g or gsv is None:
            print(f"{s:6} {nm:26} no-GS  <-- BLOCKER")
            blockers.append((s, nm, "no-gs"))
            continue
        ratio = gsv / g
        flag = "OK" if ratio < 0.5 else "STARTER <-- BLOCKER"
        print(f"{s:6} {nm:26} {flag} gs/g={ratio:.2f} ({gsv}/{g})")
        if ratio >= 0.5:
            blockers.append((s, nm, f"ratio={ratio:.2f}"))

    for s in (2008, 2012):
        pool = [r["nid"] for r in conn.execute(
            "SELECT DISTINCT nba_api_id nid FROM stg_nba_box_asof WHERE season=? AND gp_played_asof>=20 AND pra_std IS NOT NULL", (s,))]
        cov = sum(1 for n in pool if n in xwalk)
        print(f"pool coverage {s}: {cov}/{len(pool)} = {100.0*cov/len(pool):.1f}%")

    print()
    if blockers:
        print(f"BLOCKERS ({len(blockers)}):")
        for s, nm, why in blockers:
            print(f"   {s} {nm} ({why})")
    else:
        print("no blockers: every pre-2017 winner crosswalks and clears GS/G < 0.5.")


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Build bref->nba name crosswalk + acceptance report.")
    ap.add_argument("--db", default="data/awards.db")
    args = ap.parse_args(argv)
    conn = connect(args.db)
    summary = build(conn)
    log.info("crosswalk: %d matched (bref_total=%d, nba_pool=%d), %d ambiguous names",
             summary["matched"], summary["bref_total"], summary["nba_pool"],
             len(summary["ambiguous"]))
    if summary["ambiguous"]:
        log.info("ambiguous sample: %s", summary["ambiguous"][:8])
    report(conn)
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
