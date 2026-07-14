"""Stat-leader arm: calibration run (cleanup, filter sweep, Beta), parallel.

Sequence, all at the pricing layer, stat-leader-scoped, voter gate untouched:

  1. Frozen baseline. Ship features (own-prior + avail-hier, reb-env OFF), with a
     cleanup guard that drops snapshots whose realised leader was frozen out of
     the context by rem_team<=0 (the COVID resolved-market artefact), so the
     calibration is fitted to the real defect, not the ghost. This is the single
     fixed reference every variant is scored against.
  2. Reachability filter sweep over q in {0.01,0.02,0.05,0.1,0.2}, pooled from one
     eff draw per snapshot, with admit counts by phase.
  3. Beta calibration, fitted walk-forward to log-loss, coefficients continuous in
     the remaining-games fraction, applied then renormalised within each snapshot.
  4. Interaction: q=0.05 filter followed by a Beta fitted on the filtered output.

Diagnostics per award: the probability-by-phase reliability table (like-for-like
against the frozen baseline), continuous signed calibration error split into
tail / middle / top bands, admit counts for the filter, and the headline P(lead)
BSS / P(top3) / who-leads. Plus a 3x3 reliability grid (award x phase) of
predicted-versus-realised curves for baseline, Beta, and filter+Beta.

Parallelism is built in: BLAS pinned to one thread per process at import, a
ProcessPoolExecutor fans the (stat, season) eff-matrix passes across cores, and
the prior cache makes every variant share one fit per season. One command.

Run:
  caffeinate -i uv run python3 -m scripts.modelling.stat_leader.calib_run \
      --eval-min 2008 --eval-max 2023 --workers 6 --fit-workers 3 --out out/calib
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import logging  # noqa: E402
from collections import defaultdict  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402

import numpy as np  # noqa: E402

try:
    from scripts.common.db import connect
    from scripts.modelling.stat_leader import mc as MC
    from scripts.modelling.stat_leader import scorecard as SC
    from scripts.modelling.stat_leader import prior_cache as PC
    from scripts.modelling.stat_leader import calib as CB
except ImportError:  # pragma: no cover
    from db import connect  # type: ignore
    import mc as MC  # type: ignore
    import scorecard as SC  # type: ignore
    import prior_cache as PC  # type: ignore
    import calib as CB  # type: ignore

log = logging.getLogger("stat_leader.calib_run")

STAT_FLOOR = {"pts": 1997, "reb": 1997, "ast": 2013}
QS = [0.01, 0.02, 0.05, 0.1, 0.2]
INTERACTION_Q = 0.05
MIN_PRIOR = 3
PHASES = ("early", "mid", "late")


def _phase(i, n):
    if n <= 1:
        return "mid"
    fr = i / (n - 1)
    return "early" if fr < 1 / 3 else ("mid" if fr < 2 / 3 else "late")


def _ship_globals(B, avail_prior):
    MC.MPG_K = B["mpg_k"]
    MC.GAMES_K = B["games_k"]
    MC.REB_ENV_VAR = 0.0                       # reb-env OFF, decided by the settlement
    MC.OWN_PRIOR_K = MC.V.REF_MIN
    for attr, val in (("AVAIL_HIER", True), ("CORR", False), ("CORR2", False),
                      ("HIER_FANO", False)):
        setattr(MC, attr, val)
    MC._AVAIL_PRIOR = avail_prior


def _fit_one(db, season, lookback, refit):
    conn = connect(db)
    try:
        PC.ensure(conn, season, lookback, refit)
    finally:
        conn.close()
    return season


def _rows_one(db, stat, season, lookback, k, field_n, qs):
    """One (stat, season): regenerate eff matrices, emit baseline + per-q rows,
    with the frozen-leader cleanup guard. Returns (stat, rows, admit)."""
    conn = connect(db)
    try:
        B, ap = PC.load(conn, season, lookback)
        _ship_globals(B, ap)
        eff_real = MC.realised_eff(B["finals"], B["ftg"], season, stat)
        if not eff_real:
            return stat, [], {}
        order = sorted(eff_real, key=eff_real.get, reverse=True)
        leader = order[0]
        top3 = set(order[:3])
        snaps = sorted(B["ctx"].keys())
        n = len(snaps)
        rows = []
        admit = defaultdict(lambda: [0, 0])
        for si, snap in enumerate(snaps):
            ctx_snap = B["ctx"].get(snap, {})
            if leader not in ctx_snap:               # cleanup: leader frozen out (rem_team<=0)
                continue
            field = MC._field_at(B["counts"], ctx_snap, season, snap, stat, field_n)
            if not field:
                continue
            eff = MC._eff_matrix(stat, season, snap, field, B["counts"], ctx_snap,
                                 B["vpriors"], B["npriors"], B["pools"], B["tcut"],
                                 B["pos"], B["firstyr"], k)
            var = CB.reachability_variants(eff, qs)
            ph = _phase(si, n)
            fr = si / max(1, n - 1)
            for name, (pl, p3, nadm) in var.items():
                tag = "base" if name == "base" else f"q{name}"
                if name != "base":
                    a = admit[(tag, ph)]
                    a[0] += nadm
                    a[1] += 1
                for i, pid in enumerate(field):
                    d = B["counts"].get((season, snap, pid), {})
                    gp = d.get("gp_played_asof") or 0.0
                    bpg = (MC.BANKED[stat](d) / gp) if gp else 0.0
                    rows.append({"variant": tag, "season": season, "snap": snap,
                                 "pid": pid, "frac": fr, "phase": ph, "bpg": bpg,
                                 "p_lead": float(pl[i]), "p_top3": float(p3[i]),
                                 "y_lead": 1 if pid == leader else 0,
                                 "y_top3": 1 if pid in top3 else 0})
        return stat, rows, {k2: v for k2, v in admit.items()}
    finally:
        conn.close()


def _strip(rows):
    """Rows for scorecard.summary (needs season, snap, pid, p_lead, p_top3,
    y_lead, y_top3, phase, bpg)."""
    return rows


def beta_walkforward(rows, min_prior=MIN_PRIOR):
    """Fit the stage-conditioned Beta map walk-forward on P(lead), apply and
    renormalise within snapshot. Seasons without enough history pass through."""
    seasons = sorted({r["season"] for r in rows})
    by_season = defaultdict(list)
    for r in rows:
        by_season[r["season"]].append(r)
    out = []
    for s in seasons:
        prior = [r for r in rows if r["season"] < s]
        if len({r["season"] for r in prior}) < min_prior:
            out.extend({**r} for r in by_season[s])
            continue
        w = CB.beta_fit([r["p_lead"] for r in prior],
                        [r["y_lead"] for r in prior],
                        [r["frac"] for r in prior])
        grp = defaultdict(list)
        for r in by_season[s]:
            grp[(r["season"], r["snap"])].append(r)
        for _, rws in grp.items():
            pc = CB.beta_apply([r["p_lead"] for r in rws], [r["frac"] for r in rws], w)
            pc = CB.renorm_snapshot(pc)
            for r, pv in zip(rws, pc):
                out.append({**r, "p_lead": float(pv)})
    return out


def band_line(tag, rows):
    p = np.array([r["p_lead"] for r in rows])
    y = np.array([r["y_lead"] for r in rows])
    b = CB.band_report(p, y)
    parts = "  ".join(f"{k}={v[0]:+.3f}(n{v[1]})" for k, v in b.items())
    print(f"  [{tag}] signed calib error  {parts}")


def phase_bands(tag, rows):
    for ph in PHASES:
        pr = [r for r in rows if r["phase"] == ph]
        if pr:
            band_line(f"{tag}/{ph}", pr)


def render_grid(base_by_stat, beta_by_stat, fb_by_stat, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        log.warning("matplotlib unavailable (%s); skipping reliability grid", e)
        return False
    stats = ["reb", "pts", "ast"]
    fig, axes = plt.subplots(3, 3, figsize=(15, 13), squeeze=False)
    for si, st in enumerate(stats):
        for pj, ph in enumerate(PHASES):
            ax = axes[si][pj]
            ax.plot([0, 1], [0, 1], color="0.6", lw=0.8, ls=":")
            for rowset, lab, style in ((base_by_stat, "baseline", "o-"),
                                       (beta_by_stat, "beta", "s-"),
                                       (fb_by_stat, "q0.05+beta", "^-")):
                rs = [r for r in rowset.get(st, []) if r["phase"] == ph]
                if not rs:
                    continue
                pm, em, ns = CB.reliability_curve([r["p_lead"] for r in rs],
                                                  [r["y_lead"] for r in rs])
                if pm.size:
                    ax.plot(pm, em, style, ms=3, lw=1.2, label=lab)
            ax.set_title(f"{st.upper()}  {ph}", fontsize=9)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_xlabel("predicted P(lead)"); ax.set_ylabel("realised")
            ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return True


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Stat-leader calibration run (parallel).")
    p.add_argument("--db", default="data/awards.db")
    p.add_argument("--eval-min", type=int, default=2008)
    p.add_argument("--eval-max", type=int, default=2023)
    p.add_argument("--fit-lookback", type=int, default=10)
    p.add_argument("--k", type=int, default=MC.DEFAULT_K)
    p.add_argument("--field-n", type=int, default=MC.FIELD_N)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--fit-workers", type=int, default=3)
    p.add_argument("--refit", action="store_true")
    p.add_argument("--out", default="out/calib")
    args = p.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    seasons = list(range(args.eval_min, args.eval_max + 1))
    stats = ["reb", "pts", "ast"]

    try:
        from scripts.common.config import assert_not_sealed
    except ImportError:
        from config import assert_not_sealed  # type: ignore
    for st in stats:
        for s in seasons:
            if s >= STAT_FLOOR[st]:
                assert_not_sealed(MC.STAT_AWARD[st], s)

    log.info("phase 1: prior cache for %d seasons (fit-workers=%d)", len(seasons), args.fit_workers)
    with ProcessPoolExecutor(max_workers=args.fit_workers) as ex:
        for f in as_completed([ex.submit(_fit_one, args.db, s, args.fit_lookback, args.refit)
                               for s in seasons]):
            f.result()

    tasks = [(st, s) for st in stats for s in seasons if s >= STAT_FLOOR[st]]
    log.info("phase 2: eff passes for %d (stat,season) tasks (workers=%d)", len(tasks), args.workers)
    rows_by_variant = defaultdict(lambda: defaultdict(list))   # variant -> stat -> rows
    admit_by_stat = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # stat -> (q,phase)->[sum,n]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_rows_one, args.db, st, s, args.fit_lookback,
                          args.k, args.field_n, QS) for st, s in tasks]
        for f in as_completed(futs):
            st, rows, admit = f.result()
            for r in rows:
                rows_by_variant[r["variant"]][st].append(r)
            for key, v in admit.items():
                a = admit_by_stat[st][key]
                a[0] += v[0]; a[1] += v[1]

    base = rows_by_variant["base"]
    beta = {st: beta_walkforward(base[st]) for st in stats if base.get(st)}
    q05 = rows_by_variant[f"q{INTERACTION_Q}"]
    fb = {st: beta_walkforward(q05[st]) for st in stats if q05.get(st)}

    # ---- headline + reliability + bands for the key variants
    for st in stats:
        if not base.get(st):
            continue
        for tag, rs in (("BASELINE (cleaned, reb-env off)", base[st]),
                        ("BETA (walk-forward)", beta.get(st, [])),
                        (f"q{INTERACTION_Q} FILTER + BETA", fb.get(st, []))):
            if not rs:
                continue
            print("\n" + "#" * 92)
            print(f"# {tag}   stat={st}")
            print("#" * 92)
            SC.summary(st, rs)
            SC.phase_bin_report(st, rs)
            phase_bands(tag, rs)

    # ---- filter sweep turning-point table
    print("\n" + "=" * 84)
    print("FILTER SWEEP: admit counts and calibration by q (find the turning point)")
    print("-" * 84)
    for st in stats:
        if not base.get(st):
            continue
        print(f"\n stat={st}")
        b = base[st]
        bl = SC._bss(np.array([r["p_lead"] for r in b]), np.array([r["y_lead"] for r in b]))
        print(f"  {'variant':>8} {'admit_early':>11} {'admit_mid':>9} {'admit_late':>10} "
              f"{'BSS_clim':>9} {'tailSCE':>8} {'midSCE':>7}")
        for q in ["base"] + QS:
            tag = "base" if q == "base" else f"q{q}"
            rs = rows_by_variant[tag].get(st, [])
            if not rs:
                continue
            pl = np.array([r["p_lead"] for r in rs]); yl = np.array([r["y_lead"] for r in rs])
            bss = SC._bss(pl, yl)
            tsce = CB.signed_calib_error(pl, yl, 0.0, 0.25)[0]
            msce = CB.signed_calib_error(pl, yl, 0.25, 0.70)[0]
            def am(ph):
                a = admit_by_stat[st].get((tag, ph))
                return (a[0] / a[1]) if a and a[1] else float("nan")
            ae, am2, al = am("early"), am("mid"), am("late")
            print(f"  {tag:>8} {ae:>11.1f} {am2:>9.1f} {al:>10.1f} "
                  f"{bss:>9.3f} {tsce:>+8.3f} {msce:>+7.3f}")
    print("=" * 84)

    png = os.path.join(args.out, "reliability_grid.png")
    if render_grid(base, beta, fb, png):
        log.info("wrote %s", png)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
