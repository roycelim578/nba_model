"""Stat-leader arm: shrinkage-weight decomposition diagnostic.

Reports, per stat per within-season phase, the fraction of the MEAN rate
posterior and of the VARIANCE (per-game fano) that comes from the player's own
history versus the position-mpg cohort versus the league, averaged over the
contender field and, separately, over the top five contenders. It tests whether
the rate prior leans too hard on cohort/league and discards player-specific
within-season form, which would centre elite contenders low (the coverage-log
defect) by shrinking their high own rate toward the pack.

Read-only. No retrain. Runs on the current model; nests to whatever fano
structure fit_priors produced, so it lights up the variance own/cohort split
automatically once the hierarchical fano lands.

MEAN decomposition (from volume.rate_posterior). The posterior mean is
    w_within * banked_rate + (1 - w_within) * prior_mean,
    w_within = banked_min / (banked_min + kappa),
and with own_prior on the prior mean is itself
    w_op * own_prior_rate + (1 - w_op) * base,   w_op = own_min / (own_min + own_k),
where base is the cohort rate, backing off to league when the cohort cell was
too thin to fit. So the four terminal weights, summing to one, are
    own_within = w_within
    own_prior  = (1 - w_within) * w_op
    cohort     = (1 - w_within) * (1 - w_op)      if the cohort cell was fit
    league     = (1 - w_within) * (1 - w_op)      otherwise

VARIANCE decomposition (per-game fano). The current engine uses one global fano
per node, so the split is league = 1 by construction. When the hierarchical fano
is present (priors carry ``fano_cohort`` and ``fano_ref_gp`` and the row carries
an own-history dispersion sample count) the same own/cohort/league split is read
off the reliability weight own_games / (own_games + ref_gp).

Run:
  caffeinate -i uv run python3 -m scripts.modelling.stat_leader.shrinkage \
      --stat all --eval-min 2008 --eval-max 2023
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
    from scripts.features.stat_leader import nodes as N
    from scripts.features.stat_leader import volume as V
    from scripts.features.stat_leader import fano_hier as FH
except ImportError:  # pragma: no cover
    from db import connect  # type: ignore
    import mc as MC  # type: ignore
    import nodes as N  # type: ignore
    import volume as V  # type: ignore
    import fano_hier as FH  # type: ignore

log = logging.getLogger("stat_leader.shrinkage")

STAT_FLOOR = {"pts": 1997, "reb": 1997, "ast": 2013}
PHASES = ("early", "mid", "late")
# node key per stat, matching mc.VOL_NODE and the prior_rate_<node> row keys
NODE = MC.VOL_NODE


def _phase(i, n):
    if n <= 1:
        return "mid"
    f = i / (n - 1)
    return "early" if f < 1 / 3 else ("mid" if f < 2 / 3 else "late")


def _mean_weights(vpriors, node, cohort, banked_min, own_rate, own_min, own_k):
    """Return (own_within, own_prior, cohort, league), summing to 1, or None if
    the node has no fit gamma prior (no kappa to decompose against)."""
    g = vpriors.get("gamma", {}).get(node)
    if not g:
        return None
    kappa = g[1]
    cohort_present = cohort in vpriors.get("gamma_cohort", {}).get(node, {})
    w_within = banked_min / (banked_min + kappa) if (banked_min + kappa) > 0 else 0.0
    prior_mass = 1.0 - w_within
    w_op = 0.0
    if own_k is not None and own_rate is not None and own_rate > 0 and own_min > 0:
        w_op = own_min / (own_min + own_k)
    own_prior = prior_mass * w_op
    base = prior_mass * (1.0 - w_op)
    cohort_w = base if cohort_present else 0.0
    league_w = 0.0 if cohort_present else base
    return w_within, own_prior, cohort_w, league_w


def _var_weights(vpriors, node, d):
    """Return (own, cohort, league) for the per-game fano, summing to 1. Current
    global-fano engine -> league = 1 (no fano_hier). Hierarchical fano -> read the
    reliability weight own_games / (own_games + ref_gp) and split the remainder by
    whether the player's mpg x volume cell was fit."""
    hier = vpriors.get("fano_hier")
    if not hier or node not in FH.HIER_NODES:
        return 0.0, 0.0, 1.0
    of = d.get(FH.OWN_FANO_COL[node])
    gp = d.get("gp_played_asof") or 0.0
    vc, vm = V._vol_count(d, node)
    if vc is None or vm <= 0 or gp <= 0:
        return 0.0, 0.0, 1.0
    cell = FH._cell(node, vc / vm, vm / gp, hier["mpg_cuts"], hier["vol_cuts"])
    w = gp / (gp + hier["ref_gp"]) if (of is not None and of > 0) else 0.0
    base = 1.0 - w
    if cell in hier.get("cohort", {}).get(node, {}):
        return w, base, 0.0
    return w, 0.0, base


def collect(B, season, stat, field_n, own_k):
    node = NODE[stat]
    snaps = sorted(B["ctx"].keys())
    rows = []
    for si, snap in enumerate(snaps):
        ctx_snap = B["ctx"].get(snap, {})
        field = MC._field_at(B["counts"], ctx_snap, season, snap, stat, field_n)
        if not field:
            continue
        ph = _phase(si, len(snaps))
        for rank, pid in enumerate(field):
            d = B["counts"].get((season, snap, pid))
            if not d:
                continue
            vc, vm = V._vol_count(d, node)
            if vc is None:
                continue
            cohort = N._cohort(pid, season, d, B["pos"], B["firstyr"], B["vpriors"]["mpg_cuts"])
            mw = _mean_weights(B["vpriors"], node, cohort, vm,
                               d.get(f"prior_rate_{node}"), d.get("prior_min") or 0.0, own_k)
            if mw is None:
                continue
            vw = _var_weights(B["vpriors"], node, d)
            rows.append({"phase": ph, "rank": rank, "mw": mw, "vw": vw})
    return rows


def _avg(rows, key, idx):
    xs = [r[key][idx] for r in rows]
    return float(np.mean(xs)) if xs else float("nan")


def _report(stat, rows, top, own_prior_on):
    def block(sel, label):
        print(f"  bucket={label}")
        print(f"    {'phase':>6} | {'own_wi':>7} {'own_pr':>7} {'cohort':>7} {'league':>7} "
              f"|| {'v_own':>6} {'v_coh':>6} {'v_lg':>6}  {'n':>5}")
        for ph in PHASES:
            r = [x for x in rows if x["phase"] == ph and sel(x)]
            if not r:
                continue
            print(f"    {ph:>6} | {_avg(r,'mw',0):>7.3f} {_avg(r,'mw',1):>7.3f} "
                  f"{_avg(r,'mw',2):>7.3f} {_avg(r,'mw',3):>7.3f} || "
                  f"{_avg(r,'vw',0):>6.3f} {_avg(r,'vw',1):>6.3f} {_avg(r,'vw',2):>6.3f}  {len(r):>5}")

    print("\n" + "=" * 86)
    print(f"stat={stat}  shrinkage-weight decomposition  own_prior={'on' if own_prior_on else 'off'}")
    print("  MEAN own_wi=within-season banked, own_pr=prior-season own; VAR own=own-history dispersion")
    print("  each of {own_wi+own_pr, cohort, league} and {v_own, v_coh, v_lg} sums to 1 per row")
    print("-" * 86)
    block(lambda x: True, "field(top-%d)" % MC.FIELD_N)
    block(lambda x: x["rank"] < top, "top%d" % top)
    print("=" * 86)


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Stat-leader shrinkage-weight decomposition.")
    p.add_argument("--db", default="data/awards.db")
    p.add_argument("--stat", default="all", choices=["reb", "pts", "ast", "all"])
    p.add_argument("--eval-min", type=int, default=2008)
    p.add_argument("--eval-max", type=int, default=2023)
    p.add_argument("--fit-lookback", type=int, default=10)
    p.add_argument("--field-n", type=int, default=MC.FIELD_N)
    p.add_argument("--top", type=int, default=5)
    p.add_argument("--no-own-prior", action="store_true",
                   help="decompose the cohort-only prior (own_prior is locked ON by default)")
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

    own_k = None if args.no_own_prior else V.REF_MIN
    pooled = {st: [] for st in stats}
    conn = connect(args.db)
    for s in seasons:
        active = [st for st in stats if s >= STAT_FLOOR[st]]
        if not active:
            continue
        try:
            B = MC.load_all(conn, s, args.fit_lookback)
        except Exception as e:  # noqa: BLE001
            log.warning("season %d skipped (%s)", s, e)
            continue
        MC.OWN_PRIOR_K = own_k
        for st in active:
            pooled[st].extend(collect(B, s, st, args.field_n, own_k))
    conn.close()
    MC.OWN_PRIOR_K = None

    for st in stats:
        if not pooled[st]:
            print(f"\nstat={st}: no rows")
            continue
        _report(st, pooled[st], args.top, own_k is not None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
