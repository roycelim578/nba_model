"""Stat-leader arm: pool-relative availability probe (differential + residual).

Compares realised remaining games and minutes to the availability projection the
MC actually draws, split by contender rank, to answer two questions:

  DIFFERENTIAL (--avail-hier off): does the raw pool under-project the durable
  elite more than the pack? top5 >> pack => a fix sharpens P(lead); top5 ~ pack
  => common-mode, mostly a P(top3)/totals lever (though nonlinear, so not fully
  cancelling in P(lead)).

  RESIDUAL (--avail-hier on): after the own-history recentre, how much miss is
  left? ratios near 1.0 => avail-hier closed it; still >1 => minutes/games want
  more than the current recentre.

  ratio_g = realised remaining games   / projected games
  ratio_m = realised remaining minutes / projected minutes

Draws via MC._draw_availability (the same pool draw the engine uses) and, with
--avail-hier, applies avail_hier.recentre before measuring. Realised remainder
comes from the finals row (final totals minus banked), all from load_all.
Read-only.

Run:
  uv run python3 -m scripts.modelling.stat_leader.avail_probe --stat all
  uv run python3 -m scripts.modelling.stat_leader.avail_probe --stat all --avail-hier
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict

import numpy as np

try:
    from scripts.common.db import connect
    from scripts.features.stat_leader import availability as A
    from scripts.features.stat_leader import avail_hier as AH
    from scripts.modelling.stat_leader import mc as MC
except ImportError:  # pragma: no cover
    from db import connect  # type: ignore
    import availability as A  # type: ignore
    import avail_hier as AH  # type: ignore
    import mc as MC  # type: ignore

log = logging.getLogger("stat_leader.avail_probe")

STAT_FLOOR = {"pts": 1997, "reb": 1997, "ast": 2013}
PHASES = ("early", "mid", "late")
K = 400


def _phase(i, n):
    if n <= 1:
        return "mid"
    f = i / (n - 1)
    return "early" if f < 1 / 3 else ("mid" if f < 2 / 3 else "late")


def _bucket(r):
    return "top5" if r < 5 else ("6-15" if r < 15 else "16-30")


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Pool-relative availability probe.")
    p.add_argument("--db", default="data/awards.db")
    p.add_argument("--stat", default="all", choices=["reb", "pts", "ast", "all"])
    p.add_argument("--eval-min", type=int, default=2008)
    p.add_argument("--eval-max", type=int, default=2023)
    p.add_argument("--fit-lookback", type=int, default=10)
    p.add_argument("--avail-hier", action="store_true",
                   help="apply the own-history recentre before measuring (residual mode)")
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args(argv)
    stats = ["reb", "pts", "ast"] if args.stat == "all" else [args.stat]
    rng = np.random.default_rng(args.seed)

    conn = connect(args.db)
    train = list(range(args.eval_min, args.eval_max + 1))
    tcut = A._terciles([r["avail_rate"] for r in A._load(conn, train)])
    pools = A.fit(A._load(conn, train), tcut)
    prior = AH.fit(conn, train) if args.avail_hier else None

    acc = {st: defaultdict(lambda: {"g": [], "m": []}) for st in stats}
    for s in range(args.eval_min, args.eval_max + 1):
        B = MC.load_all(conn, s, args.fit_lookback)
        snaps = sorted(B["ctx"].keys())
        for st in stats:
            if s < STAT_FLOOR[st]:
                continue
            for si, snap in enumerate(snaps):
                ctx_snap = B["ctx"].get(snap, {})
                field = MC._field_at(B["counts"], ctx_snap, s, snap, st, MC.FIELD_N)
                ph = _phase(si, len(snaps))
                for rank, pid in enumerate(field):
                    rc = ctx_snap.get(pid)
                    d = B["counts"].get((s, snap, pid))
                    fd = B["finals"].get((s, pid))
                    if not rc or not d or not fd:
                        continue
                    rf, rm = MC._draw_availability(pools, rc, tcut, K, rng)
                    if rf is None:
                        continue
                    if args.avail_hier and prior is not None:
                        rf, rm = AH.recentre(rf, rm, rc, d, pid, s, prior)
                    gp = d.get("gp_played_asof") or 0.0
                    mn = d.get("min_asof") or 0.0
                    real_g = (fd.get("gp_played_asof") or 0.0) - gp
                    real_min = (fd.get("min_asof") or 0.0) - mn
                    if real_g <= 0:
                        continue
                    proj_g = float(np.mean(rf)) * rc["rem_team"]
                    valid = rm[~np.isnan(rm)]
                    proj_mpg = float(np.mean(valid)) if len(valid) else None
                    real_mpg = real_min / real_g
                    if proj_g > 0:
                        acc[st][(ph, _bucket(rank))]["g"].append(real_g / proj_g)
                    if proj_mpg and proj_mpg > 0:
                        acc[st][(ph, _bucket(rank))]["m"].append(real_mpg / proj_mpg)
    conn.close()

    def med(xs):
        return float(np.median(xs)) if xs else float("nan")

    mode = "RESIDUAL (avail-hier on)" if args.avail_hier else "RAW pool (avail-hier off)"
    for st in stats:
        print("\n" + "=" * 66)
        print(f"stat={st}  realised remainder / projection  [{mode}]")
        print("  ratio>1 => centred low; top5 >> pack => differential (helps P(lead))")
        print("-" * 66)
        print(f"  {'bucket':>7} {'phase':>6} {'games':>7} {'minutes':>8} {'n':>6}")
        for b in ("top5", "6-15", "16-30"):
            for ph in PHASES:
                a = acc[st].get((ph, b))
                if not a or not a["g"]:
                    continue
                print(f"  {b:>7} {ph:>6} {med(a['g']):>7.3f} {med(a['m']):>8.3f} {len(a['g']):>6}")
        print("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())
