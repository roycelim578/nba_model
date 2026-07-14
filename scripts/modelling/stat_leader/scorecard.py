"""Stat-leader arm: P(lead) calibration scorecard (v5), the gate for this arm.

Walks the Monte Carlo across non-sealed seasons (each refitting priors on its own
rolling window), applying the MEASURED remaining-minutes shrink from minutes.py
(coverage-matched K per season, no hand-picked scalar). Scores P(lead) with Brier,
climatology BSS, BSS against a leaderboard-anchoring softmax baseline, and a
categorical who-leads log-loss, then prints reliability split by within-season
phase so a mid-band miscalibration can be read as early-season uncertainty versus a
genuine late-season refusal to commit.

--no-shrink runs the untouched engine (MPG_K disabled) for a side-by-side, so we
can see exactly what the measured shrink bought and confirm it was earned.

Run:
  caffeinate -i uv run python3 -m scripts.modelling.stat_leader.scorecard \
      --stat all --eval-min 2008 --eval-max 2023
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from collections import defaultdict

import numpy as np

try:
    from scripts.common.db import connect
    from scripts.modelling.stat_leader import mc as MC
    from scripts.features.stat_leader import minutes as MIN
except ImportError:  # pragma: no cover
    from db import connect  # type: ignore
    import mc as MC  # type: ignore
    import minutes as MIN  # type: ignore

log = logging.getLogger("stat_leader.scorecard")

REL_EDGES = (0.0, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 1.0)
STAT_FLOOR = {"pts": 1997, "reb": 1997, "ast": 2013}
PHASES = ("early", "mid", "late")
EPS = 1e-9


def _phase(i, n):
    if n <= 1:
        return "mid"
    f = i / (n - 1)
    return "early" if f < 1 / 3 else ("mid" if f < 2 / 3 else "late")


def collect(B, season, stat, k, field_n):
    eff_real = MC.realised_eff(B["finals"], B["ftg"], season, stat)
    if not eff_real:
        return []
    order = sorted(eff_real, key=eff_real.get, reverse=True)
    leader = order[0]; top3 = set(order[:3])
    snaps = sorted(B["ctx"].keys())
    rows = []
    for si, snap in enumerate(snaps):
        field, p_lead, p_top3 = MC.snapshot_probs(
            stat, season, snap, B["counts"], B["ctx"], B["vpriors"], B["npriors"],
            B["pools"], B["tcut"], B["pos"], B["firstyr"], k, field_n)
        if not field:
            continue
        ph = _phase(si, len(snaps))
        for i, pid in enumerate(field):
            d = B["counts"].get((season, snap, pid), {})
            gp = d.get("gp_played_asof") or 0.0
            bpg = (MC.BANKED[stat](d) / gp) if gp else 0.0
            rows.append({"season": season, "snap": snap, "pid": pid,
                         "p_lead": float(p_lead[i]), "p_top3": float(p_top3[i]),
                         "y_lead": 1 if pid == leader else 0,
                         "y_top3": 1 if pid in top3 else 0, "phase": ph, "bpg": bpg})
    return rows


def _brier(ps, ys):
    return float(np.mean((ps - ys) ** 2)) if len(ps) else float("nan")


def _bss(ps, ys, ref_brier=None):
    ys = np.asarray(ys, float); ps = np.asarray(ps, float)
    if ref_brier is None:
        pbar = ys.mean(); ref_brier = pbar * (1 - pbar)
    return (1.0 - _brier(ps, ys) / ref_brier) if ref_brier > 0 else float("nan")


def _softmax(x, T):
    z = np.asarray(x, float) / max(T, EPS); z -= z.max(); e = np.exp(z)
    return e / max(e.sum(), EPS)


def _groups(rows):
    g = defaultdict(list)
    for r in rows:
        g[(r["season"], r["snap"])].append(r)
    return g


def _baseline_by_key(rows, T):
    out = {}
    for _, rws in _groups(rows).items():
        p = _softmax([r["bpg"] for r in rws], T)
        for r, pv in zip(rws, p):
            out[(r["season"], r["snap"], r["pid"])] = float(pv)
    return out


def _fit_temperature(rows):
    gs = _groups(rows)
    best_T, best_b = 1.0, float("inf")
    for T in (0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 40.0):
        ps, ys = [], []
        for _, rws in gs.items():
            p = _softmax([r["bpg"] for r in rws], T)
            ps += list(p); ys += [r["y_lead"] for r in rws]
        b = _brier(np.array(ps), np.array(ys))
        if b < best_b:
            best_b, best_T = b, T
    return best_T, best_b


def _logloss(rows, prob_of):
    lls, miss = [], 0
    for _, rws in _groups(rows).items():
        lead = [r for r in rws if r["y_lead"] == 1]
        if not lead:
            miss += 1; continue
        lls.append(-math.log(max(prob_of(lead[0]), EPS)))
    return (float(np.mean(lls)) if lls else float("nan")), miss


def summary(stat, rows):
    arr = np.array([(r["p_lead"], r["p_top3"], r["y_lead"], r["y_top3"]) for r in rows], float)
    pL, p3, yL, y3 = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    T, b_base = _fit_temperature(rows)
    base_key = _baseline_by_key(rows, T)
    ll_m, miss = _logloss(rows, lambda r: r["p_lead"])
    ll_b, _ = _logloss(rows, lambda r: base_key[(r["season"], r["snap"], r["pid"])])
    print(f"  P(lead)  Brier={_brier(pL, yL):.4f}  BSS_clim={_bss(pL, yL):+.3f}  "
          f"BSS_vs_leaderboard={_bss(pL, yL, b_base):+.3f}  (softmax T={T:g})")
    print(f"  P(top3)  Brier={_brier(p3, y3):.4f}  BSS_clim={_bss(p3, y3):+.3f}")
    print(f"  who-leads logloss  model={ll_m:.3f}  leaderboard={ll_b:.3f}  "
          f"skill={1 - ll_m / ll_b:+.3f}  (leader out-of-field groups={miss})")


def _bin_rows(pL, yL, lo, hi, last):
    m = (pL >= lo) & (pL <= hi) if last else (pL >= lo) & (pL < hi)
    n = int(m.sum())
    return n, (float(pL[m].mean()) if n else float("nan")), (float(yL[m].mean()) if n else float("nan"))


def phase_bin_report(stat, rows):
    by = {ph: np.array([(r["p_lead"], r["y_lead"]) for r in rows if r["phase"] == ph], float)
          for ph in PHASES}
    print(f"  reliability by phase (diagonal within phase = calibrated; off-diagonal LATE = defect)")
    print(f"    {'bin':>11} | {'early n/pred/emp':>22} | {'mid n/pred/emp':>22} | {'late n/pred/emp':>22}")
    for j, (lo, hi) in enumerate(zip(REL_EDGES[:-1], REL_EDGES[1:])):
        last = j == len(REL_EDGES) - 2
        cells = []
        for ph in PHASES:
            a = by[ph]
            if len(a):
                n, mp, me = _bin_rows(a[:, 0], a[:, 1], lo, hi, last)
            else:
                n, mp, me = 0, float("nan"), float("nan")
            pr = f"{mp:.3f}" if n else "-"; em = f"{me:.3f}" if n else "-"
            cells.append(f"{n:>6} {pr:>7} {em:>7}")
        print(f"    {f'{lo:.2f}-{hi:.2f}':>11} | {cells[0]:>22} | {cells[1]:>22} | {cells[2]:>22}")


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Stat-leader P(lead) scorecard v5 (measured minutes shrink).")
    p.add_argument("--db", default="data/awards.db")
    p.add_argument("--stat", default="all", choices=["reb", "pts", "ast", "all"])
    p.add_argument("--eval-min", type=int, default=2008)
    p.add_argument("--eval-max", type=int, default=2023)
    p.add_argument("--fit-lookback", type=int, default=10)
    p.add_argument("--k", type=int, default=MC.DEFAULT_K)
    p.add_argument("--field-n", type=int, default=MC.FIELD_N)
    p.add_argument("--no-shrink", action="store_true", help="disable the measured minutes/games shrink")
    p.add_argument("--no-reb-env", action="store_true", help="disable the shared rebounding-environment factor")
    p.add_argument("--own-prior", action="store_true", help="blend the rate prior mean toward the player's prior-season rate")
    p.add_argument("--avail-hier", action="store_true", help="hierarchical availability recentre (own history for the centre)")
    p.add_argument("--corr2", action="store_true", help="v2 correlation: heterogeneous per-opponent shared shock")
    p.add_argument("--corr", action="store_true", help="named-driver correlation: measured remaining-opponent mu-sharpening")
    p.add_argument("--hier-fano", action="store_true", help="hierarchical per-game fano (own-history -> mpg x volume cohort -> league)")
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
    kvals = defaultdict(list); kgvals = defaultdict(list); envvals = defaultdict(list)
    conn = connect(args.db)
    for s in seasons:
        active = [st for st in stats if s >= STAT_FLOOR[st]]
        if not active:
            continue
        MC.CORR = bool(args.corr)
        MC.AVAIL_HIER = bool(args.avail_hier); MC.CORR2 = bool(args.corr2)
        try:
            B = MC.load_all(conn, s, args.fit_lookback)
        except Exception as e:
            log.warning("season %d skipped (%s)", s, e); continue
        MC.MPG_K = None if args.no_shrink else B["mpg_k"]
        MC.GAMES_K = None if args.no_shrink else B["games_k"]
        MC.REB_ENV_VAR = 0.0 if args.no_reb_env else B["reb_env_var"]
        MC.OWN_PRIOR_K = MC.V.REF_MIN if args.own_prior else None
        MC.HIER_FANO = bool(args.hier_fano)
        log.info("season %d: fit rolling %d-%d, minutesK=%.0f gamesKg=%.0f rebEnvVar=%.4f own_prior=%s%s", s,
                 s - args.fit_lookback, s - 1, B["mpg_k"], B["games_k"], B["reb_env_var"],
                 bool(args.own_prior), " (SHRINK OFF)" if args.no_shrink else "")
        for st in active:
            kvals[st].append(B["mpg_k"]); kgvals[st].append(B["games_k"])
            envvals[st].append(B["reb_env_var"])
            pooled[st].extend(collect(B, s, st, args.k, args.field_n))
    conn.close()
    MC.MPG_K = None; MC.GAMES_K = None; MC.REB_ENV_VAR = 0.0; MC.OWN_PRIOR_K = None; MC.AVAIL_HIER = False; MC.CORR2 = False; MC.CORR = False; MC.HIER_FANO = False

    for st in stats:
        if not pooled[st]:
            print(f"\nstat={st}: no rows"); continue
        n_seas = len({r["season"] for r in pooled[st]})
        kmed = float(np.median(kvals[st])) if kvals[st] else float("nan")
        kgmed = float(np.median(kgvals[st])) if kgvals[st] else float("nan")
        envmed = float(np.median(envvals[st])) if envvals[st] else 0.0
        if args.no_shrink:
            tag = "SHRINK OFF"
        else:
            mtag = "off" if kmed >= MIN.NO_SHRINK else f"K~{kmed:.0f}"
            gtag = "off" if kgmed >= MIN.NO_SHRINK else f"Kg~{kgmed:.0f}"
            tag = f"minutes={mtag} games={gtag}"
        if st == "reb":
            tag += f" reb_env={'off' if args.no_reb_env or envmed <= 0 else f'var~{envmed:.4f}'}"
        if args.own_prior:
            tag += " own_prior=on"
        if args.avail_hier:
            tag += " avail_hier=on"
        if args.corr2:
            tag += " corr2=on"
        if args.corr:
            tag += " corr=on"
        if args.hier_fano:
            tag += " hier_fano=on"
        print("\n" + "=" * 94)
        print(f"stat={st}  seasons={n_seas}  rows={len(pooled[st])}  [{tag}]")
        summary(st, pooled[st])
        print("-" * 94)
        phase_bin_report(st, pooled[st])
        print("=" * 94)
    return 0


if __name__ == "__main__":
    sys.exit(main())
