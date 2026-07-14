"""Stat-leader arm: terminal composed-coverage diagnostic.

Every per-node coverage-match conditions on realised remaining minutes, by design,
to isolate the volume node from availability. So we have never checked whether the
COMPOSITION, availability compounded with volume over the full remainder, is
calibrated at the terminal eff-value level. Two individually cov80 layers need not
compose to a cov80 total: compounding can over- or under-disperse the product even
when each part is perfect.

This draws each contender's full terminal eff distribution (the MC eff-matrix row)
as-of each snapshot and checks whether the realised terminal eff falls in the
central-80, split by phase and by banked-rank bucket, with mean PIT (predictive CDF
position of the realised value, ~0.5 if centred right) and a median-ratio
(realised / predictive-median, >1 => predictive centred too low, a mu bias).

READ:
  cov80 well ABOVE 0.80  => composition OVER-dispersed (predictive too wide); this
                            is the hypothesis-consistent finding for the mid-band
                            under-confidence, since too-wide contender marginals
                            overlap and flatten the argmax. Fix: deflate composed
                            dispersion (hierarchical own-dispersion node, or
                            terminal calibration against the peer spread).
  cov80 well BELOW 0.80  => composition UNDER-dispersed (predictive too narrow).
  meanPIT >> 0.5         => predictive centred too low (mu understated).
  medianRatio >> 1       => same, seen on the level.

CAVEAT: this is a per-player MARGINAL terminal check. cov80 ~0.80 does NOT clear
the correlation hypothesis, which is a JOINT property a marginal check cannot see;
and fixing terminal marginals may still leave joint over-dispersion. Complementary
to comove/pmass, not a substitute. Read with all knobs off to isolate the
structural composition.

Run:
  uv run python3 -m scripts.modelling.stat_leader.coverage --stat all --eval-min 2008 --eval-max 2023
"""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np

try:
    from scripts.common.db import connect
    from scripts.modelling.stat_leader import mc as MC
except ImportError:  # pragma: no cover
    from db import connect  # type: ignore
    import mc as MC  # type: ignore

log = logging.getLogger("stat_leader.coverage")

STAT_FLOOR = {"pts": 1997, "reb": 1997, "ast": 2013}
PHASES = ("early", "mid", "late")
RANK_BUCKETS = (("top5", 1, 5), ("rank6_30", 6, 30))


def _phase(i, n):
    if n <= 1:
        return "mid"
    f = i / (n - 1)
    return "early" if f < 1 / 3 else ("mid" if f < 2 / 3 else "late")


def _rank_bucket(rank):
    for name, lo, hi in RANK_BUCKETS:
        if lo <= rank <= hi:
            return name
    return None


def collect(B, season, stat, k, field_n):
    er = MC.realised_eff(B["finals"], B["ftg"], season, stat)
    if not er:
        return []
    snaps = sorted(B["ctx"].keys())
    rows = []
    for si, snap in enumerate(snaps):
        ctx_snap = B["ctx"].get(snap, {})
        field = MC._field_at(B["counts"], ctx_snap, season, snap, stat, field_n)
        if not field:
            continue
        eff = MC._eff_matrix(stat, season, snap, field, B["counts"], ctx_snap,
                             B["vpriors"], B["npriors"], B["pools"], B["tcut"],
                             B["pos"], B["firstyr"], k)
        banked = []
        for pid in field:
            d = B["counts"].get((season, snap, pid), {})
            gp = d.get("gp_played_asof") or 0.0
            banked.append((MC.BANKED[stat](d) / gp) if gp else 0.0)
        order = np.argsort(-np.asarray(banked))
        rank_of = {field[order[j]]: j + 1 for j in range(len(field))}
        ph = _phase(si, len(snaps))
        for i, pid in enumerate(field):
            if pid not in er:
                continue
            draws = eff[i, :]
            if not np.any(draws > 0):
                continue
            realised = er[pid]
            q10, q90 = np.quantile(draws, [0.1, 0.9])
            med = float(np.median(draws))
            pit = float(np.mean(draws <= realised))
            rows.append({"phase": ph, "bucket": _rank_bucket(rank_of[pid]),
                         "inside": 1.0 if q10 <= realised <= q90 else 0.0,
                         "pit": pit, "ratio": (realised / med) if med > 0 else float("nan")})
    return rows


def report(stat, rows):
    print("\n" + "=" * 78)
    print(f"stat={stat}  terminal composed-coverage  contender-snapshots={len(rows)}")
    print("  cov80 target 0.80; >0.80 over-dispersed, <0.80 under-dispersed;")
    print("  meanPIT target 0.50; medianRatio target 1.00 (>1 => predictive centred low)")
    print(f"    {'phase':>6} {'bucket':>9} {'n':>6} {'cov80':>7} {'meanPIT':>8} {'medRatio':>9}")
    for ph in PHASES:
        for name, _, _ in RANK_BUCKETS:
            sub = [r for r in rows if r["phase"] == ph and r["bucket"] == name]
            if not sub:
                continue
            cov = np.mean([r["inside"] for r in sub])
            pit = np.mean([r["pit"] for r in sub])
            ratio = np.nanmedian([r["ratio"] for r in sub])
            print(f"    {ph:>6} {name:>9} {len(sub):>6} {cov:>7.3f} {pit:>8.3f} {ratio:>9.3f}")
    allr = rows
    print(f"    {'ALL':>6} {'':>9} {len(allr):>6} "
          f"{np.mean([r['inside'] for r in allr]):>7.3f} "
          f"{np.mean([r['pit'] for r in allr]):>8.3f} "
          f"{np.nanmedian([r['ratio'] for r in allr]):>9.3f}")
    print("=" * 78)


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Stat-leader terminal composed-coverage diagnostic.")
    p.add_argument("--db", default="data/awards.db")
    p.add_argument("--stat", default="all", choices=["reb", "pts", "ast", "all"])
    p.add_argument("--eval-min", type=int, default=2008)
    p.add_argument("--eval-max", type=int, default=2023)
    p.add_argument("--fit-lookback", type=int, default=10)
    p.add_argument("--k", type=int, default=MC.DEFAULT_K)
    p.add_argument("--field-n", type=int, default=MC.FIELD_N)
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

    pooled = {st: [] for st in stats}
    conn = connect(args.db)
    for s in seasons:
        active = [st for st in stats if s >= STAT_FLOOR[st]]
        if not active:
            continue
        try:
            B = MC.load_all(conn, s, args.fit_lookback)
        except Exception as e:
            log.warning("season %d skipped (%s)", s, e); continue
        MC.MPG_K = None; MC.GAMES_K = None; MC.REB_ENV_VAR = 0.0; MC.OWN_PRIOR_K = None
        log.info("season %d", s)
        for st in active:
            pooled[st].extend(collect(B, s, st, args.k, args.field_n))
    conn.close()

    for st in stats:
        if not pooled[st]:
            print(f"\nstat={st}: no rows"); continue
        report(st, pooled[st])
    return 0


if __name__ == "__main__":
    sys.exit(main())
