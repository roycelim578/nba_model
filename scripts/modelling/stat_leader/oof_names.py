"""Stat-leader arm: out-of-field leader diagnostic (who did the field miss, when).

The REB P(lead) deficit versus the leaderboard is entirely the snapshots where the
realised season leader was outside the top-30 banked field, so the model scored
him ~0. This names those leaders per stat per season and shows how long they sat
out of the field and when they entered, to settle whether it is a handful of
late-emerging players (benign, we catch them once they surface) or a systematic
field-width problem, and whether it is really REB-specific or just this sample.

Read-only. Prints, per stat, each season whose realised leader was ever
out-of-field: the leader's name, the count and phase span of snapshots he was
out, and the season-fraction at which he first entered the top-30.

Run:
  uv run python3 -m scripts.modelling.stat_leader.oof_names --stat all --eval-min 2008 --eval-max 2023
"""

from __future__ import annotations

import argparse
import logging
import sys

try:
    from scripts.common.db import connect
    from scripts.modelling.stat_leader import mc as MC
except ImportError:  # pragma: no cover
    from db import connect  # type: ignore
    import mc as MC  # type: ignore

log = logging.getLogger("stat_leader.oof_names")

STAT_FLOOR = {"pts": 1997, "reb": 1997, "ast": 2013}


def _name_map(conn):
    """{nba_api_id: name} from the players table, tolerant of column naming."""
    try:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(players)")]
    except Exception:  # noqa: BLE001
        return {}
    idc = next((c for c in ("nba_api_id", "nba_id", "player_nba_id") if c in cols), None)
    nmc = next((c for c in ("full_name", "display_name", "player_name", "name") if c in cols), None)
    if not idc or not nmc:
        return {}
    return {r[0]: r[1] for r in conn.execute(f"SELECT {idc}, {nmc} FROM players "
                                             f"WHERE {idc} IS NOT NULL")}


def _phase(i, n):
    if n <= 1:
        return "mid"
    f = i / (n - 1)
    return "early" if f < 1 / 3 else ("mid" if f < 2 / 3 else "late")


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Out-of-field leader diagnostic.")
    p.add_argument("--db", default="data/awards.db")
    p.add_argument("--stat", default="all", choices=["reb", "pts", "ast", "all"])
    p.add_argument("--eval-min", type=int, default=2008)
    p.add_argument("--eval-max", type=int, default=2023)
    p.add_argument("--fit-lookback", type=int, default=10)
    args = p.parse_args(argv)
    stats = ["reb", "pts", "ast"] if args.stat == "all" else [args.stat]

    conn = connect(args.db)
    names = _name_map(conn)
    for st in stats:
        print("\n" + "=" * 74)
        print(f"stat={st}  realised leaders the top-{MC.FIELD_N} field missed, and when they entered")
        print("-" * 74)
        rows = []
        for s in range(args.eval_min, args.eval_max + 1):
            if s < STAT_FLOOR[st]:
                continue
            B = MC.load_all(conn, s, args.fit_lookback)
            eff = MC.realised_eff(B["finals"], B["ftg"], s, st)
            if not eff:
                continue
            leader = max(eff, key=eff.get)
            snaps = sorted(B["ctx"].keys())
            out_phases, entered_at = [], None
            for si, snap in enumerate(snaps):
                field = MC._field_at(B["counts"], B["ctx"].get(snap, {}), s, snap, st, MC.FIELD_N)
                if leader in field:
                    if entered_at is None:
                        entered_at = si / max(1, len(snaps) - 1)
                else:
                    out_phases.append(_phase(si, len(snaps)))
            if out_phases:
                rows.append((s, leader, len(out_phases), out_phases[0], out_phases[-1], entered_at))
        if not rows:
            print("  none: the realised leader was always in-field.")
        else:
            print(f"  {'season':>6} {'leader':>26} {'#out':>5} {'from':>6} {'to':>5} {'entered@':>9}")
            for s, pid, nout, ph0, ph1, ent in rows:
                nm = names.get(pid, f"id={pid}")
                ent_s = f"{ent:.0%}" if ent is not None else "never"
                print(f"  {s:>6} {nm[:26]:>26} {nout:>5} {ph0:>6} {ph1:>5} {ent_s:>9}")
            print(f"  {len(rows)} season(s) with an out-of-field leader; "
                  f"{len({r[1] for r in rows})} distinct player(s).")
        print("=" * 74)
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
