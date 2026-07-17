"""Split rebounds into offensive and defensive components (cumulative-total basis).

reb = oreb + dreb. Total rebounds are cumulative in stat_rate_counts_asof.reb;
defensive rebounds are cumulative in stg_nba_player_advanced_asof.dreb (a season
total, ~90 mid-season, NOT per game). So the split must be done in cumulative
totals: oreb_total = reb_total - dreb_total, then per-game = /gp.

Reconciles the two sources (they come from different pulls):
  - dreb share = dreb_total / reb_total (expect ~0.65-0.78)
  - fraction of rows with dreb_total > reb_total (source/timing mismatch)
Then characterises each component's surge and whether they move independently.

Run:
  uv run python3 -m scripts.modelling.stat_leader.reb_split --seasons 2016-2023
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

try:
    from scripts.common.db import connect
    from scripts.features.stat_leader import nodes as N
except ImportError:  # pragma: no cover
    from db import connect  # type: ignore
    import nodes as N  # type: ignore


def _load_dreb(conn, seasons):
    qs = ",".join("?" * len(seasons))
    out = {}
    for r in conn.execute(
            f'SELECT season, snapshot_date, nba_api_id, dreb '
            f'FROM stg_nba_player_advanced_asof WHERE season IN ({qs})', seasons):
        if r["dreb"] is not None:
            out[(r["season"], r["snapshot_date"], r["nba_api_id"])] = float(r["dreb"])
    return out


def _stage(frac):
    return "early" if frac > 2.0 / 3 else ("mid" if frac > 1.0 / 3 else "late")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/awards.db")
    ap.add_argument("--seasons", default="2016-2023")
    a = ap.parse_args(argv)
    lo, hi = (a.seasons.split("-") + [a.seasons])[:2]
    seasons = list(range(int(lo), int(hi) + 1))

    conn = connect(a.db)
    counts, finals, _, _ = N._load(conn, seasons)
    dreb = _load_dreb(conn, seasons)
    last = {}
    for (s, snap, pid) in counts:
        k = (s, pid)
        if k not in last or snap > last[k]:
            last[k] = snap

    reb_t, dreb_t, gp_of = {}, {}, {}
    for (s, snap, pid), d in counts.items():
        gp = d.get("gp_played_asof") or 0.0
        dr = dreb.get((s, snap, pid))
        if gp >= 1 and dr is not None:
            reb_t[(s, snap, pid)] = d.get("reb") or 0.0
            dreb_t[(s, snap, pid)] = dr
            gp_of[(s, snap, pid)] = gp

    R = np.array([reb_t[k] for k in reb_t])
    D = np.array([dreb_t[k] for k in reb_t])
    G = np.array([gp_of[k] for k in reb_t])
    share = np.clip(D / np.maximum(R, 1e-9), 0, 2)
    print("=" * 60)
    print(f"REB SPLIT reconciliation (cumulative)  seasons {seasons[0]}-{seasons[-1]}")
    print(f"  mean reb/g {np.mean(R/G):.2f}   mean dreb/g {np.mean(D/G):.2f}   "
          f"mean oreb/g {np.mean((R-D)/G):.2f}")
    print(f"  dreb share of total {np.mean(share):.2f}   "
          f"rows with dreb>reb {100*np.mean(D > R):.1f}%")
    if np.mean(D > R) > 0.05 or not (0.55 < np.mean(share) < 0.82):
        print("  WARNING: split still not reconciling; do not use.")
    else:
        print("  OK: split reconciles, oreb/dreb per-game are usable.")

    def surge(comp):
        out = {}
        for (s, snap, pid), gp in gp_of.items():
            fk = (s, pid); fsnap = last[fk]
            if (s, fsnap, pid) not in reb_t:
                continue
            fgp = gp_of[(s, fsnap, pid)]
            rem = fgp - gp
            if gp < 5 or rem < 10:
                continue
            if comp == "dreb":
                bt, ft = dreb_t[(s, snap, pid)], dreb_t[(s, fsnap, pid)]
            else:
                bt = reb_t[(s, snap, pid)] - dreb_t[(s, snap, pid)]
                ft = reb_t[(s, fsnap, pid)] - dreb_t[(s, fsnap, pid)]
            b_pg = bt / gp
            out[(s, snap, pid)] = ((ft - bt) / rem - b_pg, _stage(rem / fgp))
        return out

    so, sd = surge("oreb"), surge("dreb")
    print("\ncomponent surge (remaining per-game minus banked), by stage:")
    for name, srg in (("oreb", so), ("dreb", sd)):
        for stg in ("early", "mid", "late"):
            v = [x for x, st in srg.values() if st == stg]
            if v:
                print(f"  {name} {stg:<5} mean {np.mean(v):+.3f}  sd {np.std(v):.3f}  n={len(v)}")
    common = set(so) & set(sd)
    if len(common) > 30:
        a1 = np.array([so[k][0] for k in common])
        a2 = np.array([sd[k][0] for k in common])
        print(f"\ncorr(oreb surge, dreb surge) = {np.corrcoef(a1, a2)[0,1]:+.2f}  "
              f"(near 0 => independently driven => splitting adds signal)  n={len(common)}")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
