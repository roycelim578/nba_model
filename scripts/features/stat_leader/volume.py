"""Stat-leader arm: volume/level nodes (the per-minute rate layer).

The shape layer in ``nodes.py`` (allocation split, shotmix split, zone
efficiencies, ast conversion) is all conditional on a volume the player
generates; this module supplies that volume. Three sticky per-minute rate nodes,
fit on training seasons only, each exposing a posterior the Monte Carlo samples:

  usage       used possessions per minute = (used_fga + used_ft_trip + used_tov)
              / minutes. Sets the PTS branch's total used possessions, which the
              allocation Dirichlet then splits into FGA : FT-trip : TOV.
  reb         total rebounds per minute. The rebound-leader market is a total and
              the substrate carries only total rebounds, so this is a single
              node; the offensive/defensive split is deliberately NOT taken. It
              would add proxy error (no true OREB/DREB columns exist, only a
              split proxy) without moving a total-rebound argmax.
  ast_create  potential assists per minute (creation volume). Gated 2013+ (the
              tracking floor); the ast-conversion Beta in nodes.py turns creation
              into assists. Pre-2013 has no creation volume, so AST is the
              thinnest, earliest-decided book.

STRUCTURE (Gamma-Poisson, the count analogue of the Beta rate nodes). A player's
per-minute rate has a Gamma prior whose mean is the shrunk cohort mean and whose
strength is a learned pseudo-minute count; the posterior adds the player's banked
count over banked minutes, so a heavy-minute player self-dominates and a
thin-minute player falls back to cohort then league. The MC draws a rate from the
Gamma posterior, then per remaining game draws a count around rate x game-minutes.

Two learned quantities, both measured on the training pool, walk-forward honest:
  slope   banked(~game 25) -> final per-minute-rate regression slope. slope near
          1 (stable rate, e.g. usage) => small pseudo-minute prior => trust the
          data; low slope (noisy) => larger prior => shrink harder. Fixed at a low
          reference minute count REF_MIN, so it never scales with a player's own
          minute volume (that would invert the shrinkage); a heavy-minute star
          self-dominates via his banked minutes, not via the prior. Sets ONLY the
          rate posterior, which is the pure conjugate estimate of the mean rate.
  fano    per-GAME count overdispersion Var/mean, the extra spread of a night's
          volume beyond Poisson (22 shots one game, 9 the next off the same
          minutes). It governs ONLY the per-game count law (a negative binomial
          when above one), NOT the rate posterior. Measured by coverage-matching
          on the training remaining-count predictive, the same disciplined grid
          search used for the Dirichlet widths in nodes.py: pick the fano whose
          training central-80 count coverage is nearest 0.80. Measuring it from
          weekly increments understates it, because aggregating a week's games
          averages the per-game overdispersion away.

Cohort = position x mpg-tier, imported from nodes.py so the volume and shape
layers share one cohort definition (drift between them would be a quiet bug).
Walk-forward: priors fit on seasons < eval, never touching a held-out season. The
rolling 10-year window is applied by the caller (the MC / scorecard harness),
exactly as for nodes.py; this module fits on whatever pool it is handed.

Run (calibration report; rolling 10-year window is the default):
  uv run python -m scripts.features.stat_leader.volume --eval-min 2022 --eval-max 2023
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict

import numpy as np

try:
    from scripts.common.db import connect
except ImportError:  # pragma: no cover
    from db import connect  # type: ignore

try:
    from scripts.features.stat_leader.nodes import _load, _cohort, MIN_MPG, MIN_COHORT
except ImportError:  # pragma: no cover
    from nodes import _load, _cohort, MIN_MPG, MIN_COHORT  # type: ignore

log = logging.getLogger("stat_leader.volume")

REF_MIN = 300.0            # reference banked minutes at which a slope-1 node is
                           # already mostly trusted; sets the prior pseudo-minutes
                           # and, crucially, is fixed (not the pool mean minutes).
KAPPA_MIN_BOUNDS = (50.0, 3000.0)
MIN_BANKED_MIN = 100.0     # min banked minutes before a snapshot enters fit/eval
MIN_FINAL_MIN = 400.0      # min final minutes for a season to be a calibration label
FANO_GRID = (1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0)
FANO_TARGET_COV = 0.80     # per-game count law widened to hit this training coverage
FANO_FIT_CAP = 5000        # cap training samples per node in the coverage match
FANO_FIT_DRAWS = 150       # predictive draws per sample in the coverage match

# volume node -> substrate count columns summed to form the numerator
VOLUME_NODES = {
    "usage": ["used_fga", "used_ft_trip", "used_tov"],
    "reb": ["reb"],
    "ast_create": ["potential_ast_asof"],
}


def _vol_count(d, node):
    """Return (count, minutes) for a node at a substrate row, or (None, None) if a
    component column is absent (e.g. potential_ast pre-2013) or minutes are zero."""
    vals = [d.get(c) for c in VOLUME_NODES[node]]
    if any(v is None for v in vals):
        return None, None
    mn = d.get("min_asof")
    if not mn or mn <= 0:
        return None, None
    return float(sum(vals)), float(mn)


def _measure_slope(counts, finals):
    """Measure, on the training pool only, per-node banked(~game 25)->final
    per-minute-rate regression slope, which sets the prior strength. Walk-forward
    honest, never hardcoded."""
    by = defaultdict(list)
    for (s, snap, pid), d in counts.items():
        by[(s, pid)].append((snap, d))
    slope_pairs = defaultdict(lambda: ([], []))
    for (s, pid), snaps in by.items():
        fd = finals.get((s, pid))
        if not fd or not fd.get("gp_played_asof"):
            continue
        if (fd["min_asof"] / max(fd["gp_played_asof"], 1)) < MIN_MPG:
            continue
        snaps.sort(key=lambda x: x[0])
        for node in VOLUME_NODES:
            fc, fm = _vol_count(fd, node)
            if fc is None or fm < MIN_FINAL_MIN:
                continue
            mids = [dd for (_, dd) in snaps if (dd.get("gp_played_asof") or 0) >= 25]
            if mids:
                mc, mm = _vol_count(mids[0], node)
                if mc is not None and mm >= MIN_BANKED_MIN:
                    X, Y = slope_pairs[node]
                    X.append(mc / mm); Y.append(fc / fm)
    slope = {}
    for node in VOLUME_NODES:
        X, Y = slope_pairs[node]
        if len(X) >= 30:
            X = np.array(X); Y = np.array(Y); A = np.c_[np.ones(len(X)), X]
            b, *_ = np.linalg.lstsq(A, Y, rcond=None)
            slope[node] = float(min(max(b[1], 0.05), 0.98))
        else:
            slope[node] = 0.6
    return slope


def _mpg_cuts(finals):
    mpgs = [(d["min_asof"] / d["gp_played_asof"]) for (s, pid), d in finals.items()
            if d.get("gp_played_asof") and (d["min_asof"] / d["gp_played_asof"]) >= MIN_MPG]
    return (np.quantile(mpgs, 1 / 3), np.quantile(mpgs, 2 / 3)) if mpgs else (20.0, 28.0)


def fit_priors(counts, finals, pos, firstyr):
    slope = _measure_slope(counts, finals)
    mpg_cuts = _mpg_cuts(finals)

    # minute-weighted cohort and league mean per-minute rates from finals, with a
    # player count per cell for the min-count backoff.
    league = defaultdict(lambda: [0.0, 0.0, 0])              # node -> [sum cnt, sum min, n]
    cohort = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0, 0]))
    for (s, pid), d in finals.items():
        if not d.get("gp_played_asof") or (d["min_asof"] / d["gp_played_asof"]) < MIN_MPG:
            continue
        c = _cohort(pid, s, d, pos, firstyr, mpg_cuts)
        for node in VOLUME_NODES:
            cnt, mn = _vol_count(d, node)
            if cnt is None or mn < MIN_FINAL_MIN:
                continue
            L = league[node]; L[0] += cnt; L[1] += mn; L[2] += 1
            C = cohort[node][c]; C[0] += cnt; C[1] += mn; C[2] += 1

    def kappa_from_slope(sl):
        k = REF_MIN * (1.0 - sl) / sl
        return float(min(max(k, KAPPA_MIN_BOUNDS[0]), KAPPA_MIN_BOUNDS[1]))

    priors = {"mpg_cuts": mpg_cuts, "gamma": {}, "gamma_cohort": {},
              "fano": {}, "slope": slope}
    for node in VOLUME_NODES:
        L = league[node]
        if L[2] >= 5 and L[1] > 0:
            priors["gamma"][node] = (L[0] / L[1], kappa_from_slope(slope[node]))
            priors["gamma_cohort"][node] = {
                c: (C[0] / C[1], kappa_from_slope(slope[node]))
                for c, C in cohort[node].items() if C[2] >= MIN_COHORT and C[1] > 0}
    # per-game overdispersion, coverage-matched on the training remaining-count
    # predictive (needs the rate posterior above, so it runs last).
    priors["fano"] = _fit_fano(priors, counts, finals, pos, firstyr)
    return priors


def rate_posterior(priors, node, cohort, banked_count, banked_min,
                   own_rate=None, own_min=0.0, own_k=None):
    """Gamma(shape, rate) posterior for the per-minute volume rate, the pure
    conjugate estimate of the MEAN rate. Sample as rng.gamma(shape, 1.0 / rate).
    Prior mean is the cohort (then league) rate, worth kappa pseudo-minutes of
    exposure, so a heavy-minute player self-dominates. fano does NOT enter here:
    per-game overdispersion is aleatoric and lives in the count law, not in the
    uncertainty about the mean.

    When own_k is given and the player has a reliable prior-season rate (own_rate
    over own_min minutes), the prior MEAN is blended toward his own prior-season
    rate by w = own_min / (own_min + own_k), a far stronger predictor for an
    established player than the position-mpg cohort. own_k None nests exactly to
    the cohort-only prior."""
    m, kappa = priors["gamma_cohort"].get(node, {}).get(
        cohort, priors["gamma"].get(node, (0.0, REF_MIN)))
    if own_k is not None and own_rate is not None and own_rate > 0 and own_min > 0:
        w = own_min / (own_min + own_k)
        m = w * own_rate + (1.0 - w) * m
    a0, b0 = m * kappa, kappa
    return a0 + banked_count, b0 + banked_min


def _draw_count(rng, rate_draws, rem_min, fano):
    """Fold a per-game count law over the remaining minutes. A single negative
    binomial over rem_min with fano F equals the sum of per-game NBs sharing
    p = 1/F (NB is closed under addition at common p), so this matches the MC's
    per-game draw exactly. fano == 1 degenerates to Poisson."""
    mean_cnt = np.maximum(rate_draws * rem_min, 0.0)
    if fano > 1.0 + 1e-9:
        nn = np.maximum(mean_cnt / (fano - 1.0), 1e-6)
        return rng.negative_binomial(nn, 1.0 / fano)
    return rng.poisson(mean_cnt)


def _fit_fano(priors, counts, finals, pos, firstyr):
    """Coverage-match the per-game fano per node on the TRAINING pool. For each
    scored snapshot form the remaining-count posterior-predictive conditioned on
    realised remaining minutes, then pick the fano whose central-80 count coverage
    is nearest FANO_TARGET_COV. This is the grid-search idiom used for the
    Dirichlet widths in nodes.py, targeting the exact metric the node must hit."""
    mpg_cuts = priors["mpg_cuts"]
    rng = np.random.default_rng(23)
    samples = defaultdict(list)
    for (s, snap, pid), d in counts.items():
        fd = finals.get((s, pid))
        if not fd or not fd.get("gp_played_asof") or (fd["min_asof"] / fd["gp_played_asof"]) < MIN_MPG:
            continue
        c = _cohort(pid, s, d, pos, firstyr, mpg_cuts)
        for node in VOLUME_NODES:
            cnt, mn = _vol_count(d, node)
            fc, fm = _vol_count(fd, node)
            if cnt is None or fc is None:
                continue
            rem_cnt = fc - cnt; rem_min = fm - mn
            if mn < MIN_BANKED_MIN or rem_min < MIN_BANKED_MIN or rem_cnt < 0:
                continue
            a, b = rate_posterior(priors, node, c, cnt, mn)
            if a <= 0 or b <= 0:
                continue
            rd = rng.gamma(a, 1.0 / b, size=FANO_FIT_DRAWS)
            samples[node].append((rd, rem_min, rem_cnt))
    fano = {}
    for node in VOLUME_NODES:
        S = samples[node]
        if len(S) < 50:
            fano[node] = 1.0; continue
        if len(S) > FANO_FIT_CAP:
            idx = rng.choice(len(S), size=FANO_FIT_CAP, replace=False)
            S = [S[i] for i in idx]
        best_f, best_gap = 1.0, 9.9
        for F in FANO_GRID:
            hits = tot = 0
            for rd, rem_min, rem_cnt in S:
                cd = _draw_count(rng, rd, rem_min, F)
                lo, hi = np.quantile(cd, [0.1, 0.9])
                hits += 1 if lo <= rem_cnt <= hi else 0; tot += 1
            gap = abs(hits / tot - FANO_TARGET_COV)
            if gap < best_gap:
                best_gap, best_f = gap, F
        fano[node] = float(best_f)
    return fano


def calib(priors, counts, finals, pos, firstyr, eval_seasons):
    """Predictive calibration on the REMAINING season, the quantity the MC forms
    (banked plus a simulated remainder). Scoring the final total instead would let
    the conditioning data overlap the target: late in the season the banked counts
    already are almost the whole final total, so the realised final sits on the
    posterior mean and coverage drifts toward one regardless of width. Targeting
    the remaining season is out-of-sample at every phase and removes that artefact.

    Two views per node per games-played phase, conditioned on realised remaining
    minutes to isolate the volume node from the availability node:
      rate  (PITr/covr)  posterior rate vs realised remaining per-minute rate.
      count (PITc/covc)  posterior-predictive remaining count (draw the rate, then
                         a per-game count law with the node's fano) vs the realised
                         remaining count. This is the MC-faithful view and the only
                         one that exercises fano."""
    mpg_cuts = priors["mpg_cuts"]
    rng = np.random.default_rng(11)

    def phase(gp):
        return "early" if gp < 20 else ("mid" if gp < 45 else "late")

    acc = defaultdict(lambda: defaultdict(lambda: {
        "pit_r": [], "cov_r": [], "pit_c": [], "cov_c": []}))
    for (s, snap, pid), d in counts.items():
        if s not in eval_seasons:
            continue
        fd = finals.get((s, pid))
        if not fd or not fd.get("gp_played_asof") or (fd["min_asof"] / fd["gp_played_asof"]) < MIN_MPG:
            continue
        c = _cohort(pid, s, d, pos, firstyr, mpg_cuts)
        ph = phase(d.get("gp_played_asof") or 0)
        for node in VOLUME_NODES:
            cnt, mn = _vol_count(d, node)
            fc, fm = _vol_count(fd, node)
            if cnt is None or fc is None:
                continue
            rem_cnt = fc - cnt; rem_min = fm - mn
            if mn < MIN_BANKED_MIN or rem_min < MIN_BANKED_MIN or rem_cnt < 0:
                continue
            a, b = rate_posterior(priors, node, c, cnt, mn)
            if a <= 0 or b <= 0:
                continue
            rate_draws = rng.gamma(a, 1.0 / b, size=400)
            realised_rate = rem_cnt / rem_min
            fano = priors.get("fano", {}).get(node, 1.0)
            cnt_draws = _draw_count(rng, rate_draws, rem_min, fano)
            for bucket in (ph, "ALL"):
                A = acc[node][bucket]
                A["pit_r"].append(float(np.mean(rate_draws <= realised_rate)))
                lo, hi = np.quantile(rate_draws, [0.1, 0.9])
                A["cov_r"].append(1 if lo <= realised_rate <= hi else 0)
                A["pit_c"].append(float(np.mean(cnt_draws <= rem_cnt)))
                lo, hi = np.quantile(cnt_draws, [0.1, 0.9])
                A["cov_c"].append(1 if lo <= rem_cnt <= hi else 0)
    return acc


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Stat-leader volume/level nodes.")
    p.add_argument("--db", default="data/awards.db")
    p.add_argument("--fit-min", type=int, default=2012)
    p.add_argument("--fit-max", type=int, default=2021)
    p.add_argument("--eval-min", type=int, default=2022)
    p.add_argument("--eval-max", type=int, default=2023)
    args = p.parse_args(argv)
    conn = connect(args.db)
    fit_seasons = list(range(args.fit_min, args.fit_max + 1))
    eval_seasons = set(range(args.eval_min, args.eval_max + 1))
    all_seasons = fit_seasons + sorted(eval_seasons)
    counts, finals, pos, firstyr = _load(conn, all_seasons)
    conn.close()

    fit_finals = {k: v for k, v in finals.items() if k[0] in set(fit_seasons)}
    fit_counts = {k: v for k, v in counts.items() if k[0] in set(fit_seasons)}
    priors = fit_priors(fit_counts, fit_finals, pos, firstyr)
    log.info("fit priors on %d..%d; mpg terciles %.1f/%.1f", args.fit_min, args.fit_max,
             priors["mpg_cuts"][0], priors["mpg_cuts"][1])
    for node in VOLUME_NODES:
        g = priors["gamma"].get(node)
        if not g:
            log.info("  %-11s NO DATA (check substrate column / season floor)", node); continue
        nc = len(priors["gamma_cohort"].get(node, {}))
        log.info("  %-11s league rate=%.4f/min kappa=%.0f slope=%.2f fano=%.2f cohorts=%d",
                 node, g[0], g[1], priors["slope"][node], priors["fano"].get(node, 1.0), nc)

    acc = calib(priors, counts, finals, pos, firstyr, eval_seasons)
    print("\n" + "=" * 82)
    print(f"volume-node calibration  fit={args.fit_min}-{args.fit_max} eval={args.eval_min}-{args.eval_max}")
    print("remaining-season predictive.  well-calibrated: PIT ~0.50, cov80 ~0.80")
    print("  r = rate posterior vs realised remaining rate")
    print("  c = predictive count (rate x remaining minutes, per-game law) vs realised remaining count")
    print("=" * 82)
    print(f"{'node':>11} {'phase':>6} {'n':>6} {'PITr':>7} {'covr':>7} {'PITc':>7} {'covc':>7}")
    for node in VOLUME_NODES:
        any_row = False
        for ph in ("ALL", "early", "mid", "late"):
            a = acc[node][ph]
            if not a["pit_r"]:
                continue
            any_row = True
            print(f"{node:>11} {ph:>6} {len(a['pit_r']):>6} "
                  f"{np.mean(a['pit_r']):>7.3f} {np.mean(a['cov_r']):>7.3f} "
                  f"{np.mean(a['pit_c']):>7.3f} {np.mean(a['cov_c']):>7.3f}")
        if any_row:
            print("-" * 82)
    print("=" * 82)
    return 0


if __name__ == "__main__":
    sys.exit(main())
