"""Additive award-voting promoter (read-only against players).

Promotes staged award-voting rows for one award from stg_award_voting into
canonical award_voting, resolving bref_id -> player_id against the EXISTING
players table WITHOUT writing to it. This mirrors exactly what resolve_identity
did for the v1 awards: it resolves names to player_ids in memory and populates
only award_voting, leaving the players identity substrate byte-identical. It
never calls resolve() / build_players, never UPDATEs players, never allocates
ids.

Resolution order per staged bref_id:
  1. a manual (bref_id -> player_id) pin passed via --manual (for disambiguating
     namesakes the automatic step cannot);
  2. an existing players.bref_id link, if one happens to be present;
  3. an unambiguous normalised-name match against players.name (exactly one row).
Anything unmatched, ambiguous, or colliding is routed to a printed review list
and NOT promoted; resolve it (usually via --manual) and re-run. The upsert is
idempotent on (season, award, player_id), so a re-run completes only the newly
resolvable rows and re-writes the rest identically.

Dry-run by default (writes nothing, prints the plan and the review list); pass
--apply to perform the award_voting upsert. players is never written in either
mode.

Run:  uv run python -m scripts.data_pull.identity.promote_award --award 6MOTY
      uv run python -m scripts.data_pull.identity.promote_award --award 6MOTY \
          --manual dunlemi02=2399 hardati02=203501 --apply
"""

from __future__ import annotations

import argparse
import logging
import sys

from scripts.common.db import connect, upsert
from scripts.data_pull.identity.resolve_identity import normalise_name

log = logging.getLogger("promote_award")


def _rows(conn, sql, params=()):
    return [dict(r) for r in conn.execute(sql, params).fetchall()]


def load_links(conn):
    return {r["bref_id"]: r["player_id"]
            for r in conn.execute(
                "SELECT bref_id, player_id FROM players WHERE bref_id IS NOT NULL")}


def load_name_index(conn):
    idx: dict[str, list[int]] = {}
    for r in conn.execute("SELECT player_id, name FROM players"):
        idx.setdefault(normalise_name(r["name"]), []).append(r["player_id"])
    return idx


def valid_player_ids(conn):
    return {r["player_id"] for r in conn.execute("SELECT player_id FROM players")}


def resolve(conn, award, manual=None):
    """Resolve each distinct staged bref_id for the award to a player_id.

    Returns (resolved, review). resolved maps bref_id -> player_id; review is the
    not-promoted list with a reason per entry. players is only read.
    """
    manual = manual or {}
    links = load_links(conn)
    name_idx = load_name_index(conn)
    valid = valid_player_ids(conn)
    staged = _rows(
        conn,
        "SELECT DISTINCT bref_id, player_name_raw FROM stg_award_voting WHERE award = ?",
        (award,),
    )

    resolved: dict[str, int] = {}
    review: list[tuple[str, str, str]] = []

    for s in staged:
        bref_id, raw = s["bref_id"], s["player_name_raw"]
        if bref_id in manual:
            pid = manual[bref_id]
            if pid not in valid:
                review.append((bref_id, raw, f"manual player_id {pid} not in players"))
            else:
                resolved[bref_id] = pid
            continue
        if bref_id in links:
            resolved[bref_id] = links[bref_id]
            continue
        cands = name_idx.get(normalise_name(raw or ""), [])
        if len(cands) == 1:
            resolved[bref_id] = cands[0]
        elif len(cands) == 0:
            review.append((bref_id, raw, "no normalised-name match; pin via --manual"))
        else:
            review.append((bref_id, raw, f"ambiguous ({len(cands)} namesakes); pin via --manual"))

    return resolved, review


def build_award_rows(conn, award, resolved):
    staged = _rows(conn, "SELECT * FROM stg_award_voting WHERE award = ?", (award,))
    out = []
    for s in staged:
        pid = resolved.get(s["bref_id"])
        if pid is None:
            continue
        out.append({
            "season": s["season"], "award": s["award"], "player_id": pid,
            "first_place_votes": s["first_place_votes"], "total_points": s["total_points"],
            "vote_share": s["vote_share"], "rank": s["rank"], "won_flag": s["won_flag"],
        })
    return out


def promote(db_path, award, apply, manual=None):
    conn = connect(db_path)
    try:
        resolved, review = resolve(conn, award, manual)
        rows = build_award_rows(conn, award, resolved)
        print(f"=== promote {award} (apply={apply}) ===")
        print(f"distinct staged bref_ids resolved: {len(resolved)}")
        print(f"award_voting rows to upsert: {len(rows)}")
        print(f"review (NOT promoted): {len(review)}")
        for bref_id, raw, reason in review:
            print(f"    REVIEW bref_id={bref_id!r} raw={raw!r}: {reason}")
        if not apply:
            print("\nDRY RUN: nothing written (players never written in any mode).")
            return
        n = upsert(conn, "award_voting", rows, ["season", "award", "player_id"])
        conn.commit()
        print(f"\nAPPLIED: upserted {n} award_voting rows. players untouched.")
    finally:
        conn.close()


def _parse_manual(pairs):
    out = {}
    for p in pairs or []:
        k, _, v = p.partition("=")
        out[k.strip()] = int(v.strip())
    return out


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Additive award-voting promoter (read-only players).")
    p.add_argument("--db", default="data/awards.db")
    p.add_argument("--award", default="6MOTY")
    p.add_argument("--manual", nargs="*", default=[],
                   help="bref_id=player_id pins for ambiguous/unmatched voters.")
    p.add_argument("--apply", action="store_true")
    args = p.parse_args(argv)
    promote(args.db, args.award, args.apply, _parse_manual(args.manual))
    return 0


if __name__ == "__main__":
    sys.exit(main())
