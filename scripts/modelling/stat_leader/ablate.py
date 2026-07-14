"""Stat-leader arm: freeze-node ablation and the REB separation diagnostic.

Two read-only diagnostics that share the freezable MC composition below (each
node's stochastic draw is replaced by its posterior mean when its knob is frozen).

VARIANCE ABLATION (default). Freeze one node, measure the fractional collapse in
per-contender eff variance; the fraction a node accounts for localises excess
width. This is the MC form of a variance decomposition, since the P(lead) argmax
is an order statistic with no closed form.

SEPARATION DIAGNOSTIC (--separation). Freeze EVERY node so eff is each contender's
deterministic mean, then ask how often the argmax-of-means alone picks the
realised season leader, split by phase, alongside the full-MC P(lead) argmax
accuracy. Purpose: decide the REB fix. If mean-argmax is accurate, the contender
means are well separated and the residual under-separation is pure leapfrog
variance from drawing contenders independently, which justifies a shared per-game
environment shock (the finite-boards pool all contenders load on). If mean-argmax
is itself weak, the means are compressed and a correlation factor would not help.
A large gap between mean-argmax accuracy and MC accuracy is the leapfrog effect
made visible: the means know the leader, but independent noise spreads the pick.

Run (variance ablation):
  caffeinate -i uv run python3 -m scripts.modelling.stat_leader.ablate \
      --stat all --seasons 2021,2023 --top 15
Run (separation diagnostic):
  caffeinate -i uv run python3 -m scripts.modelling.stat_leader.ablate --separation \
      --stat all --seasons 2015,2017,2019,2021,2023 --top 20
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
except ImportError:  # pragma: no cover
    from db import connect  # type: ignore
    import mc as MC  # type: ignore
    import nodes as N  # type: ignore
    import volume as V  # type: ignore

log = logging.getLogger("stat_leader.ablate")

KNOBS = {
    "reb": ["avail_games", "avail_min", "avail_all", "volume"],
    "ast": ["avail_games", "avail_min", "avail_all", "volume", "conv"],
    "pts": ["avail_games", "avail_min", "avail_all", "volume", "alloc", "shotmix", "eff"],
}
STAT_FLOOR = {"pts": 1997, "reb": 1997, "ast": 2013}
PHASES = ("early", "mid", "late")


def _rate(rng, vpriors, node, cohort, vc, vm, k, frozen):
    a, b = V.rate_posterior(vpriors, node, cohort, vc, vm)
    if a <= 0 or b <= 0:
        return np.zeros(k)
    return np.full(k, a / b) if frozen else rng.gamma(a, 1.0 / b, size=k)


def _count(rng, rate, rem_min, fano, frozen):
    if frozen:
        return np.maximum(rate * rem_min, 0.0)
    return V._draw_count(rng, rate, rem_min, fano)


def _p(rng, npriors, node, cohort, d, mk, at, k, frozen):
    pa, pb = N.beta_posterior(npriors, node, cohort, d.get(mk) or 0.0, d.get(at) or 0.0)
    if pa <= 0 or pb <= 0:
        return np.zeros(k)
    return np.full(k, pa / (pa + pb)) if frozen else rng.beta(pa, pb, size=k)


def _s(rng, npriors, node, cohort, banked, od, k, frozen):
    alpha = N.dirichlet_posterior(npriors, node, cohort, banked, od=od)
    if alpha is None:
        alpha = np.asarray(banked, float) + 1.0
    alpha = np.maximum(alpha, 1e-6)
    if frozen:
        return np.tile(alpha / alpha.sum(), (k, 1))
    return rng.dirichlet(alpha, size=k)


def _rem(stat, rng, d, cohort, vpriors, npriors, rem_min, k, fr):
    volnode = MC.VOL_NODE[stat]
    vc, vm = V._vol_count(d, volnode)
    if vc is None:
        return np.zeros(k)
    fz_vol = "volume" in fr
    rate = _rate(rng, vpriors, volnode, cohort, vc, vm, k, fz_vol)
    vol = _count(rng, rate, rem_min, MC._fano(vpriors, volnode, d), fz_vol)
    if stat == "reb":
        return vol
    if stat == "ast":
        conv = _p(rng, npriors, "ast_conv", cohort, d, "ast", "potential_ast_asof", k, "conv" in fr)
        return vol * conv
    od = npriors.get("node_od", {}).get("fg2", 0.5)
    used = vol.astype(float)
    s_a = _s(rng, npriors, "alloc", cohort,
             [d.get("used_fga") or 0.0, d.get("used_ft_trip") or 0.0, d.get("used_tov") or 0.0],
             od, k, "alloc" in fr)
    fga = used * s_a[:, 0]; ft_trip = used * s_a[:, 1]
    s_m = _s(rng, npriors, "shotmix", cohort,
             [d.get("fg3a") or 0.0, d.get("fg2a_rim") or 0.0, d.get("fg2a_mid") or 0.0],
             od, k, "shotmix" in fr)
    fg3a = fga * s_m[:, 0]; rim_a = fga * s_m[:, 1]; mid_a = fga * s_m[:, 2]
    fz_eff = "eff" in fr
    p3 = _p(rng, npriors, "fg3", cohort, d, "fg3m", "fg3a", k, fz_eff)
    prim = _p(rng, npriors, "fg2_rim", cohort, d, "fg2m_rim", "fg2a_rim", k, fz_eff)
    pmid = _p(rng, npriors, "fg2_mid", cohort, d, "fg2m_mid", "fg2a_mid", k, fz_eff)
    pft = _p(rng, npriors, "ft", cohort, d, "ftm", "fta", k, fz_eff)
    fta = ft_trip / MC.FT_POSS_COEF
    return 2.0 * (rim_a * prim + mid_a * pmid) + 3.0 * (fg3a * p3) + fta * pft


def _eff(stat, season, snap, pid, d, rc, B, k, fr, seed):
    """Return (mean, var) of the eff draws for one contender under freeze set fr."""
    rng = np.random.default_rng(seed)
    gp = d.get("gp_played_asof") or 0.0
    mn = d.get("min_asof") or 0.0
    banked = MC.BANKED[stat](d)
    cohort = N._cohort(pid, season, d, B["pos"], B["firstyr"], B["vpriors"]["mpg_cuts"])
    rf, rm = MC._draw_availability(B["pools"], rc, B["tcut"], k, rng)
    if rf is None:
        return None
    banked_mpg = (mn / gp) if gp else 0.0
    rm = np.where(np.isnan(rm), banked_mpg, rm)
    if "avail_games" in fr or "avail_all" in fr:
        rf = np.full(k, float(rf.mean()))
    if "avail_min" in fr or "avail_all" in fr:
        rm = np.full(k, float(rm.mean()))
    rem_games = rf * rc["rem_team"]
    rem_min = np.maximum(rem_games * rm, 0.0)
    rem_total = _rem(stat, rng, d, cohort, B["vpriors"], B["npriors"], rem_min, k, fr)
    season_games = gp + rem_games
    denom = np.maximum(season_games, MC._qual(rc["ftg"]))
    eff = np.where(denom > 0, (banked + rem_total) / denom, 0.0)
    return float(eff.mean()), float(eff.var())


def _phase(i, n):
    if n <= 1:
        return "mid"
    f = i / (n - 1)
    return "early" if f < 1 / 3 else ("mid" if f < 2 / 3 else "late")


def run_variance(B, season, stat, k, top):
    acc = defaultdict(list); cv = []
    for snap in sorted(B["ctx"].keys()):
        ctx_snap = B["ctx"].get(snap, {})
        for pid in MC._field_at(B["counts"], ctx_snap, season, snap, stat, top):
            d = B["counts"].get((season, snap, pid)); rc = ctx_snap.get(pid)
            if not d or not rc:
                continue
            seed = MC._snap_seed(season, stat, snap) ^ (pid & 0xFFFF)
            base = _eff(stat, season, snap, pid, d, rc, B, k, set(), seed)
            if base is None or base[1] <= 0:
                continue
            m, vbase = base
            if m > 0:
                cv.append(vbase ** 0.5 / m)
            for knob in KNOBS[stat]:
                res = _eff(stat, season, snap, pid, d, rc, B, k, {knob}, seed)
                if res is not None:
                    acc[knob].append((vbase, res[1]))
    return acc, cv


def run_separation(B, season, stat, k_mc, k_fr, top):
    """Per snapshot: does the argmax of deterministic means pick the realised
    leader, and does the full MC agree? Reported by phase."""
    leader = None
    er = MC.realised_eff(B["finals"], B["ftg"], season, stat)
    if er:
        leader = max(er, key=er.get)
    if leader is None:
        return {}
    MC.MPG_K = None; MC.GAMES_K = None; MC.REB_ENV_VAR = 0.0   # separation asks about base means
    snaps = sorted(B["ctx"].keys())
    out = defaultdict(lambda: {"n": 0, "mean_hit": 0, "mc_hit": 0, "gap": [], "tie": 0})
    for si, snap in enumerate(snaps):
        ctx_snap = B["ctx"].get(snap, {})
        field = MC._field_at(B["counts"], ctx_snap, season, snap, stat, top)
        if not field or leader not in field:
            continue
        means = []
        for pid in field:
            d = B["counts"].get((season, snap, pid)); rc = ctx_snap.get(pid)
            seed = MC._snap_seed(season, stat, snap) ^ (pid & 0xFFFF)
            r = _eff(stat, season, snap, pid, d, rc, B, k_fr, set(KNOBS[stat]), seed)
            means.append(r[0] if r else 0.0)
        means = np.array(means)
        order = np.argsort(-means)
        mean_leader = field[order[0]]
        gap = (means[order[0]] - means[order[1]]) / max(means[order[0]], 1e-9) if len(field) > 1 else 1.0
        _, p_lead, _ = MC.snapshot_probs(stat, season, snap, B["counts"], B["ctx"],
                                         B["vpriors"], B["npriors"], B["pools"], B["tcut"],
                                         B["pos"], B["firstyr"], k_mc, top)
        mc_leader = field[int(np.argmax(p_lead))]
        ph = _phase(si, len(snaps))
        c = out[ph]
        c["n"] += 1
        c["mean_hit"] += 1 if mean_leader == leader else 0
        c["mc_hit"] += 1 if mc_leader == leader else 0
        c["gap"].append(gap)
        c["tie"] += 1 if gap < 0.02 else 0
    return out


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Stat-leader ablation / separation diagnostic.")
    p.add_argument("--db", default="data/awards.db")
    p.add_argument("--stat", default="all", choices=["reb", "pts", "ast", "all"])
    p.add_argument("--seasons", default="2021,2023")
    p.add_argument("--fit-lookback", type=int, default=10)
    p.add_argument("--k", type=int, default=4000)
    p.add_argument("--k-frozen", type=int, default=256)
    p.add_argument("--top", type=int, default=15)
    p.add_argument("--separation", action="store_true")
    p.add_argument("--own-prior", action="store_true", help="blend rate prior toward prior-season rate")
    p.add_argument("--avail-hier", action="store_true", help="hierarchical availability recentre")
    p.add_argument("--corr2", action="store_true", help="v2 heterogeneous correlation")
    p.add_argument("--corr", action="store_true", help="named-driver correlation: remaining-opponent mu-sharpening")
    p.add_argument("--hier-fano", action="store_true", help="hierarchical per-game fano")
    args = p.parse_args(argv)

    stats = ["reb", "pts", "ast"] if args.stat == "all" else [args.stat]
    seasons = [int(x) for x in args.seasons.split(",") if x.strip()]
    try:
        from scripts.common.config import assert_not_sealed
    except ImportError:
        from config import assert_not_sealed  # type: ignore
    for st in stats:
        for s in seasons:
            assert_not_sealed(MC.STAT_AWARD[st], s)

    conn = connect(args.db)
    if args.separation:
        agg = {st: defaultdict(lambda: {"n": 0, "mean_hit": 0, "mc_hit": 0, "gap": [], "tie": 0})
               for st in stats}
        for s in seasons:
            log.info("season %d: fit rolling %d-%d and separate", s, s - args.fit_lookback, s - 1)
            MC.AVAIL_HIER = bool(args.avail_hier); MC.CORR2 = bool(args.corr2)
            MC.CORR = bool(args.corr)
            B = MC.load_all(conn, s, args.fit_lookback)
            MC.OWN_PRIOR_K = MC.V.REF_MIN if args.own_prior else None
            MC.HIER_FANO = bool(args.hier_fano)
            for st in stats:
                if s < STAT_FLOOR[st]:
                    continue
                for ph, c in run_separation(B, s, st, args.k, args.k_frozen, args.top).items():
                    a = agg[st][ph]
                    a["n"] += c["n"]; a["mean_hit"] += c["mean_hit"]; a["mc_hit"] += c["mc_hit"]
                    a["gap"] += c["gap"]; a["tie"] += c["tie"]
        conn.close()
        for st in stats:
            print("\n" + "=" * 74)
            print(f"stat={st}  separation diagnostic  seasons={seasons}")
            print("  meanAcc = argmax-of-means picks realised leader; mcAcc = full-MC P(lead) argmax")
            print(f"    {'phase':>6} {'nSnaps':>7} {'meanAcc':>8} {'mcAcc':>7} {'medGap':>7} {'tieRate':>8}")
            for ph in PHASES:
                a = agg[st][ph]
                if a["n"] == 0:
                    continue
                medgap = float(np.median(a["gap"])) if a["gap"] else float("nan")
                print(f"    {ph:>6} {a['n']:>7} {a['mean_hit'] / a['n']:>8.3f} "
                      f"{a['mc_hit'] / a['n']:>7.3f} {medgap:>7.3f} {a['tie'] / a['n']:>8.3f}")
            print("=" * 74)
        return 0

    agg = {st: defaultdict(list) for st in stats}; cvs = {st: [] for st in stats}
    for s in seasons:
        log.info("season %d: fit rolling %d-%d and ablate", s, s - args.fit_lookback, s - 1)
        MC.AVAIL_HIER = bool(args.avail_hier); MC.CORR2 = bool(args.corr2)
        MC.CORR = bool(args.corr)
        B = MC.load_all(conn, s, args.fit_lookback)
        MC.HIER_FANO = bool(args.hier_fano)
        for st in stats:
            if s < STAT_FLOOR[st]:
                continue
            acc, cv = run_variance(B, s, st, args.k, args.top)
            for knob, pairs in acc.items():
                agg[st][knob].extend(pairs)
            cvs[st].extend(cv)
    conn.close()
    for st in stats:
        print("\n" + "=" * 66)
        cvmean = float(np.mean(cvs[st])) if cvs[st] else float("nan")
        print(f"stat={st}  seasons={seasons}  contender-snapshots={len(cvs[st])}  baseline eff CV={cvmean:.3f}")
        print(f"    {'knob':>12} {'meanContrib':>12} {'n':>7}")
        rows = []
        for knob in KNOBS[st]:
            pr = np.array(agg[st][knob], float)
            if len(pr) == 0:
                continue
            contrib = np.clip(1.0 - pr[:, 1] / np.maximum(pr[:, 0], 1e-12), 0.0, 1.0)
            rows.append((knob, float(contrib.mean()), len(pr)))
        for knob, c, n in sorted(rows, key=lambda x: -x[1]):
            print(f"    {knob:>12} {c:>12.3f} {n:>7}")
        print("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())
