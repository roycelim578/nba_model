"""Stat-leader arm: cross-contender co-movement diagnostic (sizes the prize).

The under-confidence in the mid probability bands, shared across all three stats,
is consistent with the field being jointly OVER-DISPERSED: each contender's
remaining total is drawn independently, so the model invents leapfrogs that do
not happen and bleeds win-probability off the true leader. The fix (correlation)
is only worth building if the contenders' realised outputs actually co-move.

This measures how much they co-move, per stat, as an intraclass correlation of
weekly stat-per-minute increments among the top contenders: the fraction of
increment variance explained by a shared within-season date effect (a scarce or
generous league night that lifts everyone together). High ICC => a real shared
environment worth capturing with named drivers (pace, efficiency, opponent);
near-zero ICC => correlation is not the lever and the defect is mu or field
contamination. It also splits the ICC into an early/mid/late profile, since the
argmax is only hedged while remaining season is long.

NOT attempted here: attribution of the co-movement to specific named drivers
(remaining-opponent strength, pace, eFG, playmaking orientation). That needs a
per-snapshot remaining-schedule and opponent-rating panel we have not confirmed
exists; this script sizes the total co-movement so we know whether that build is
worth it. The realised stat rate uses banked increments, so it is model-free.

Run:
  uv run python3 -m scripts.modelling.stat_leader.comove --stat all --eval-min 2008 --eval-max 2023
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict

import numpy as np

try:
    from scripts.common.db import connect
    from scripts.modelling.stat_leader import mc as MC
except ImportError:  # pragma: no cover
    from db import connect  # type: ignore
    import mc as MC  # type: ignore

log = logging.getLogger("stat_leader.comove")

STAT_FLOOR = {"pts": 1997, "reb": 1997, "ast": 2013}
MIN_GAME_MIN = 48.0
TOPK = 20


def _phase_frac(f):
    return "early" if f < 1 / 3 else ("mid" if f < 2 / 3 else "late")


def _icc(groups):
    """Intraclass correlation = between-group variance fraction of pooled residuals."""
    allres = [e for es in groups.values() for e in es]
    if len(allres) < 100:
        return float("nan"), 0
    total = float(np.var(allres))
    if total <= 0:
        return float("nan"), len(allres)
    grand = float(np.mean(allres)); num = den = 0.0
    for es in groups.values():
        if len(es) < 3:
            continue
        num += len(es) * (float(np.mean(es)) - grand) ** 2; den += len(es)
    between = num / den if den > 0 else 0.0
    return between / total, len(allres)


def collect(B, season, stat, topk=TOPK):
    """Return weekly per-minute increments for the top contenders, tagged by date
    and season phase, as (residual vs player mean)."""
    finals = B["finals"]; counts = B["counts"]
    cand = []
    for (s, pid), d in finals.items():
        if s != season:
            continue
        gp = d.get("gp_played_asof") or 0.0
        mn = d.get("min_asof") or 0.0
        if gp < 1 or mn <= 0:
            continue
        cand.append((MC.BANKED[stat](d) / mn, pid))
    top = set(pid for _, pid in sorted(cand, reverse=True)[:topk])

    panel = defaultdict(list)
    for (s, snap, pid) in counts:
        if s != season or pid not in top:
            continue
        d = counts[(s, snap, pid)]
        mn = d.get("min_asof")
        if mn is None:
            continue
        panel[pid].append((snap, MC.BANKED[stat](d), float(mn)))

    snaps_sorted = sorted({snap for (s, snap, _) in counts if s == season})
    idx_of = {snap: i for i, snap in enumerate(snaps_sorted)}
    nsnap = max(len(snaps_sorted) - 1, 1)

    inc = defaultdict(list)
    for pid, rows in panel.items():
        rows.sort()
        for a, b in zip(rows, rows[1:]):
            dmin = b[2] - a[2]; dcnt = b[1] - a[1]
            if dmin >= MIN_GAME_MIN and dcnt >= 0:
                inc[pid].append((b[0], dcnt / dmin))
    out = []
    for pid, seq in inc.items():
        if len(seq) < 3:
            continue
        mu = float(np.mean([r for _, r in seq]))
        for snap, r in seq:
            ph = _phase_frac(idx_of.get(snap, 0) / nsnap)
            out.append((snap, r - mu, ph))
    return out


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Stat-leader cross-contender co-movement (ICC).")
    p.add_argument("--db", default="data/awards.db")
    p.add_argument("--stat", default="all", choices=["reb", "pts", "ast", "all"])
    p.add_argument("--eval-min", type=int, default=2008)
    p.add_argument("--eval-max", type=int, default=2023)
    p.add_argument("--fit-lookback", type=int, default=10)
    p.add_argument("--topk", type=int, default=TOPK)
    args = p.parse_args(argv)

    stats = ["reb", "pts", "ast"] if args.stat == "all" else [args.stat]
    try:
        from scripts.common.config import assert_not_sealed
    except ImportError:
        from config import assert_not_sealed  # type: ignore
    seasons = list(range(args.eval_min, args.eval_max + 1))
    for st in stats:
        for s in seasons:
            assert_not_sealed(MC.STAT_AWARD[st], s)

    rows = {st: [] for st in stats}
    conn = connect(args.db)
    for s in seasons:
        active = [st for st in stats if s >= STAT_FLOOR[st]]
        if not active:
            continue
        try:
            B = MC.load_all(conn, s, args.fit_lookback)
        except Exception as e:
            log.warning("season %d skipped (%s)", s, e); continue
        for st in active:
            for snap, res, ph in collect(B, s, st, args.topk):
                rows[st].append(((s, snap), res, ph))
    conn.close()

    print("\n" + "=" * 72)
    print("cross-contender co-movement (ICC of weekly stat-per-min increments)")
    print("weekly increments attenuate the true per-night ICC, so this is a floor.")
    print("high => shared environment worth capturing; ~0 => correlation not the lever")
    print("=" * 72)
    print(f"{'stat':>5} {'phase':>6} {'ICC':>7} {'n_incr':>8}")
    for st in stats:
        if not rows[st]:
            print(f"{st:>5}  no rows"); continue
        for ph in ("ALL", "early", "mid", "late"):
            groups = defaultdict(list)
            for key, res, p in rows[st]:
                if ph == "ALL" or p == ph:
                    groups[key].append(res)
            icc, n = _icc(groups)
            print(f"{st:>5} {ph:>6} {icc:>7.3f} {n:>8}")
        print("-" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
