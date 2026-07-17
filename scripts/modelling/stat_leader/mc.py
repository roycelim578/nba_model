"""Stat-leader arm: joint Monte Carlo, P(lead) engine.

Composes the calibrated node layers into P(lead) per contender per weekly
snapshot, by simulating each contender's season eff_value and taking the argmax
over the field per replicate.

  eff_value = (banked_total + simulated_remaining) / max(games_played, q),
  q = ceil(0.70 * final team games)  (the soft denominator floor; a sub-qualifier
  player still leads if his floored average is highest, the Myles Turner rule).

Per replicate, per contender, drawn ONCE so within-player correlation is honest:
  1. availability -> a (remaining-games-fraction, remaining-mpg) pair, sampled
     nonparametrically from historical peers in the player's state bin. Remaining
     minutes = frac * remaining_team_games * mpg.
  2. the branch volume/shape nodes -> a remaining count over those minutes, the
     per-game count law being a negative binomial at the volume node's fano. No
     separate pace draw: the per-minute volume rates are pace-loaded, and pace is
     a common-mode shock that largely cancels in a relative argmax.
  3. compose to a season eff_value; argmax over the field -> the replicate's
     leader. P(lead) is the share of replicates each contender leads; P(top3) the
     share it lands in the top three.

Branches:
  reb  volume(reb-per-min) only.
  pts  volume(usage-per-min) -> allocation Dirichlet (FGA:FT-trip:TOV) -> shotmix
       Dirichlet (3PA:rim:mid) -> zone-efficiency and FT Betas -> the points
       identity 2*(rim+mid makes) + 3*(3P makes) + FT makes.
  ast  volume(potential-ast-per-min) -> conversion Beta (ast / potential_ast).

The field is re-selected at each snapshot as the top FIELD_N by banked per-game
stat. Priors are fit on the rolling window the caller passes, never touching a
held-out season (PRA holds out 2024 and 2025; the seal guard enforces it).

Run (sanity, dev season 2023, priors on the rolling 2013-2022 window):
  uv run python3 -m scripts.modelling.stat_leader.mc --stat all --eval-season 2023
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import pickle
import sys
import zlib

import numpy as np

try:
    from scripts.common.db import connect
    from scripts.features.stat_leader import nodes as N
    from scripts.features.stat_leader import volume as V
    from scripts.features.stat_leader import availability as A
    from scripts.features.stat_leader import minutes as MIN
    from scripts.features.stat_leader import avail_hier as AH
    from scripts.modelling.stat_leader import volume_overlay as VO
except ImportError:  # pragma: no cover
    from db import connect  # type: ignore
    import nodes as N  # type: ignore
    import volume as V  # type: ignore
    import availability as A  # type: ignore
    import minutes as MIN  # type: ignore
    import avail_hier as AH  # type: ignore
    import volume_overlay as VO  # type: ignore

log = logging.getLogger("stat_leader.mc")

FIELD_N = 30
DEFAULT_K = 3000
QUAL_FRAC = 0.70
FT_POSS_COEF = 0.44

# Measured availability shrinks (see minutes.py), set by a driver from load_all's
# coverage-matched values. None disables (identical engine). MPG_K acts on the
# minutes leg, GAMES_K on the remaining-games fraction.
MPG_K = None
GAMES_K = None
# Own-history prior (see volume.rate_posterior). None disables (cohort-only prior,
# identical engine); when set to a pseudo-minute strength the rate prior mean is
# blended toward the player's prior-season rate. Driver sets it from V.REF_MIN.
OWN_PRIOR_K = None
# Hierarchical availability recentre (avail_hier.py). Off => identical engine.
AVAIL_HIER = False
_AVAIL_PRIOR = None
# Per-book two-part volume x conversion for STL/BLK (deflections -> steals,
# rim-FGA -> blocks). Read from env so it reaches ProcessPool scorecard workers.
# Unset/0 => the direct single-leg count, byte-identical engine. 1 => two-part
# where the leg has data (2016+ hustle / 2014+ tracking), else direct as fallback.
TWO_PART = {"stl": os.environ.get("TWO_PART_STL", "1") == "1",
            "blk": os.environ.get("TWO_PART_BLK", "0") == "1"}
# Per-book volume mu-overlay (volume_overlay.py). Env-driven for ProcessPool workers.
# Off => no overlay, byte-identical. On => blend the volume-leg mean toward the
# ElasticNet driver prediction and widen by s_hat, from the per-eval-season artefact.
OVERLAY = {"stl": os.environ.get("OVERLAY_STL", "0") == "1",
           "blk": os.environ.get("OVERLAY_BLK", "0") == "1",
           "ast": os.environ.get("OVERLAY_AST", "0") == "1"}
# s_hat width is opt-in: the gamma posterior already carries the banked uncertainty,
# so adding the regression residual double-counts and over-disperses the argmax.
# Default mean-only; OVERLAY_WIDTH=1 restores the width term.
OVERLAY_WIDTH = os.environ.get("OVERLAY_WIDTH", "0") == "1"
_OVERLAY_ART = {}

STAT_AWARD = {"pts": "PTS", "reb": "REB", "ast": "AST", "stl": "STL", "blk": "BLK"}
VOL_NODE = {"reb": "reb", "pts": "usage", "ast": "ast_create", "stl": "stl", "blk": "blk"}


def _banked_reb(d):
    return d.get("reb") or 0.0


def _banked_pts(d):
    return 2.0 * (d.get("fg2m") or 0.0) + 3.0 * (d.get("fg3m") or 0.0) + (d.get("ftm") or 0.0)


def _banked_ast(d):
    return d.get("ast") or 0.0


def _banked_stl(d):
    return d.get("stl") or 0.0


def _banked_blk(d):
    return d.get("blk") or 0.0


BANKED = {"reb": _banked_reb, "pts": _banked_pts, "ast": _banked_ast, "stl": _banked_stl, "blk": _banked_blk}


def _qual(ftg):
    return math.ceil(QUAL_FRAC * ftg)


def _gamma_rate(rng, priors, node, cohort, vc, vm, k, own_rate=None, own_min=0.0):
    a, b = V.rate_posterior(priors, node, cohort, vc, vm,
                            own_rate=own_rate, own_min=own_min, own_k=OWN_PRIOR_K)
    return rng.gamma(a, 1.0 / b, size=k) if (a > 0 and b > 0) else np.zeros(k)


def _beta_draw(rng, npriors, node, cohort, d, mk_col, at_col, k):
    pa, pb = N.beta_posterior(npriors, node, cohort, d.get(mk_col) or 0.0, d.get(at_col) or 0.0)
    return rng.beta(pa, pb, size=k) if (pa > 0 and pb > 0) else np.zeros(k)


def _overlay_rate(rng, book, d, cohort, rate, banked_cnt, banked_min, k):
    """Scale the per-minute volume rate onto the overlay mean v_star and widen by
    s_hat. No-op when the book's overlay is off, the artefact is absent, or the
    per-game context is degenerate; apply() itself falls back to v_banked on a
    missing driver, so the whole path degrades to the base rate."""
    if not OVERLAY.get(book):
        return rate
    art = _OVERLAY_ART.get(book)
    if art is None:
        return rate
    gp = d.get("gp_played_asof") or 0.0
    if gp <= 0 or banked_min <= 0 or banked_cnt <= 0:
        return rate
    mpg = banked_min / gp
    if mpg <= 0:
        return rate
    v_banked = banked_cnt / gp
    drivers = {c: d.get(c) for (_t, c, _r) in art["drv"]}
    v_star, s_hat = VO.apply(art, drivers, mpg, cohort[0], v_banked)
    scaled = rate * (v_star / v_banked)
    if OVERLAY_WIDTH and s_hat and s_hat > 0:
        scaled = scaled + rng.normal(0.0, s_hat / mpg, size=k)
    return np.maximum(scaled, 0.0)


def _dir_draw(rng, npriors, node, cohort, banked, od, k):
    alpha = N.dirichlet_posterior(npriors, node, cohort, banked, od=od)
    if alpha is None:
        alpha = np.asarray(banked, float) + 1.0
    return rng.dirichlet(np.maximum(alpha, 1e-6), size=k)


def _rem_reb(rng, d, cohort, vpriors, npriors, rem_min, k):
    vc, vm = V._vol_count(d, "reb")
    if vc is None:
        return np.zeros(k)
    rate = _gamma_rate(rng, vpriors, "reb", cohort, vc, vm, k,
                       own_rate=d.get("prior_rate_reb"), own_min=d.get("prior_min") or 0.0)
    return V._draw_count(rng, rate, rem_min, vpriors["fano"].get("reb", 1.0))


def _rem_stl_twopart(rng, d, cohort, vpriors, npriors, rem_min, k):
    """Two-part steals: NB deflection volume x steals-per-deflection Beta. Returns
    None when the deflection leg is absent so the caller falls back to direct."""
    vc, vm = V._vol_count(d, "defl")
    if vc is None:
        return None
    rate = _gamma_rate(rng, vpriors, "defl", cohort, vc, vm, k,
                       own_rate=d.get("prior_rate_defl"), own_min=d.get("prior_min") or 0.0)
    rate = _overlay_rate(rng, "stl", d, cohort, rate, vc, vm, k)
    defl_ct = V._draw_count(rng, rate, rem_min, vpriors["fano"].get("defl", 1.0))
    conv = _beta_draw(rng, npriors, "stl_conv", cohort, d, "stl", "defl", k)
    return defl_ct * conv


def _rem_stl(rng, d, cohort, vpriors, npriors, rem_min, k):
    if TWO_PART.get("stl"):
        r = _rem_stl_twopart(rng, d, cohort, vpriors, npriors, rem_min, k)
        if r is not None:
            return r
    vc, vm = V._vol_count(d, "stl")
    if vc is None:
        return np.zeros(k)
    rate = _gamma_rate(rng, vpriors, "stl", cohort, vc, vm, k,
                       own_rate=d.get("prior_rate_stl"), own_min=d.get("prior_min") or 0.0)
    return V._draw_count(rng, rate, rem_min, vpriors["fano"].get("stl", 1.0))


def _rem_blk_twopart(rng, d, cohort, vpriors, npriors, rem_min, k):
    """Two-part blocks: NB rim-FGA volume x blocks-per-rim-FGA Beta. Returns None
    when the rim-FGA leg is absent so the caller falls back to direct."""
    vc, vm = V._vol_count(d, "rim_fga")
    if vc is None:
        return None
    rate = _gamma_rate(rng, vpriors, "rim_fga", cohort, vc, vm, k,
                       own_rate=d.get("prior_rate_rim_fga"), own_min=d.get("prior_min") or 0.0)
    rim_ct = V._draw_count(rng, rate, rem_min, vpriors["fano"].get("rim_fga", 1.0))
    conv = _beta_draw(rng, npriors, "blk_conv", cohort, d, "blk", "rim_fga", k)
    return rim_ct * conv


def _rem_blk(rng, d, cohort, vpriors, npriors, rem_min, k):
    if TWO_PART.get("blk"):
        r = _rem_blk_twopart(rng, d, cohort, vpriors, npriors, rem_min, k)
        if r is not None:
            return r
    vc, vm = V._vol_count(d, "blk")
    if vc is None:
        return np.zeros(k)
    rate = _gamma_rate(rng, vpriors, "blk", cohort, vc, vm, k,
                       own_rate=d.get("prior_rate_blk"), own_min=d.get("prior_min") or 0.0)
    rate = _overlay_rate(rng, "blk", d, cohort, rate, vc, vm, k)
    return V._draw_count(rng, rate, rem_min, vpriors["fano"].get("blk", 1.0))


def _rem_ast(rng, d, cohort, vpriors, npriors, rem_min, k):
    vc, vm = V._vol_count(d, "ast_create")
    if vc is None:
        return np.zeros(k)
    rate = _gamma_rate(rng, vpriors, "ast_create", cohort, vc, vm, k,
                       own_rate=d.get("prior_rate_ast_create"), own_min=d.get("prior_min") or 0.0)
    pot = V._draw_count(rng, rate, rem_min, vpriors["fano"].get("ast_create", 1.0))
    conv = _beta_draw(rng, npriors, "ast_conv", cohort, d, "ast", "potential_ast_asof", k)
    return pot * conv


def _rem_pts(rng, d, cohort, vpriors, npriors, rem_min, k):
    vc, vm = V._vol_count(d, "usage")
    if vc is None:
        return np.zeros(k)
    rate = _gamma_rate(rng, vpriors, "usage", cohort, vc, vm, k,
                       own_rate=d.get("prior_rate_usage"), own_min=d.get("prior_min") or 0.0)
    used = V._draw_count(rng, rate, rem_min, vpriors["fano"].get("usage", 1.0)).astype(float)
    od = npriors.get("node_od", {}).get("fg2", 0.5)
    s_alloc = _dir_draw(rng, npriors, "alloc", cohort,
                        [d.get("used_fga") or 0.0, d.get("used_ft_trip") or 0.0, d.get("used_tov") or 0.0], od, k)
    fga = used * s_alloc[:, 0]
    ft_trip = used * s_alloc[:, 1]
    s_mix = _dir_draw(rng, npriors, "shotmix", cohort,
                      [d.get("fg3a") or 0.0, d.get("fg2a_rim") or 0.0, d.get("fg2a_mid") or 0.0], od, k)
    fg3a = fga * s_mix[:, 0]; rim_a = fga * s_mix[:, 1]; mid_a = fga * s_mix[:, 2]
    p3 = _beta_draw(rng, npriors, "fg3", cohort, d, "fg3m", "fg3a", k)
    p_rim = _beta_draw(rng, npriors, "fg2_rim", cohort, d, "fg2m_rim", "fg2a_rim", k)
    p_mid = _beta_draw(rng, npriors, "fg2_mid", cohort, d, "fg2m_mid", "fg2a_mid", k)
    p_ft = _beta_draw(rng, npriors, "ft", cohort, d, "ftm", "fta", k)
    fta = ft_trip / FT_POSS_COEF
    return 2.0 * (rim_a * p_rim + mid_a * p_mid) + 3.0 * (fg3a * p3) + fta * p_ft


BRANCH = {"reb": _rem_reb, "pts": _rem_pts, "ast": _rem_ast, "stl": _rem_stl, "blk": _rem_blk}


def _load_context(conn, season):
    """Per (snapshot_date, nba_api_id) banked availability state plus final and
    remaining team games, mirroring availability._load's banked-state formulas so
    the state bins match the training pools. No realised labels (would be leakage)."""
    from collections import defaultdict
    fin = {}
    for r in conn.execute(
        "SELECT nba_api_id, MAX(team_games_asof) ftg FROM stg_nba_availability_asof "
        "WHERE season=? GROUP BY nba_api_id", (season,)):
        fin[r["nba_api_id"]] = r["ftg"]
    ctx = defaultdict(dict)
    for r in conn.execute(
        "SELECT a.snapshot_date, a.nba_api_id, a.games_played_asof, a.team_games_asof, "
        "a.current_absence_streak, b.mpg_std, b.mpg_l10, g.week_index "
        "FROM stg_nba_availability_asof a "
        "JOIN snapshot_grid g ON g.season=a.season AND g.snapshot_date=a.snapshot_date "
        "LEFT JOIN stg_nba_box_asof b ON b.nba_api_id=a.nba_api_id AND b.season=a.season "
        "  AND b.snapshot_date=a.snapshot_date "
        "WHERE a.season=? AND g.snapshot_kind IN ('weekly','ratings') AND a.team_games_asof>0",
        (season,)):
        pid = r["nba_api_id"]; ftg = fin.get(pid)
        if not ftg:
            continue
        rem_team = ftg - r["team_games_asof"]
        if rem_team <= 0:
            continue
        mpg_std = r["mpg_std"] or 0.0
        ctx[r["snapshot_date"]][pid] = {
            "avail_rate": r["games_played_asof"] / r["team_games_asof"],
            "form": (r["mpg_l10"] or mpg_std) - mpg_std,
            "absent": 1 if (r["current_absence_streak"] or 0) >= 2 else 0,
            "week_index": r["week_index"] if r["week_index"] is not None else -1,
            "rem_team": rem_team, "ftg": ftg,
        }
    return ctx


def _load_ftg(conn, season):
    """Final team games per player, from the availability totals (robust source for
    the qualifier q; the last snapshot's context has zero remaining and is empty)."""
    return {r["nba_api_id"]: r["ftg"] for r in conn.execute(
        "SELECT nba_api_id, MAX(team_games_asof) ftg FROM stg_nba_availability_asof "
        "WHERE season=? GROUP BY nba_api_id", (season,)) if r["ftg"]}


def _draw_availability(pools, rec, tcut, k, rng):
    b = A._bin(rec, tcut)
    pool = None
    for key in A._backoff_keys(b):
        p = pools.get(key)
        if p and len(p) >= A.MIN_POOL:
            pool = p; break
    if pool is None:
        pool = pools.get((None, None, None), [])
    if not pool:
        return None, None
    idx = rng.integers(0, len(pool), size=k)
    rf = np.array([pool[i][0] for i in idx], dtype=float)
    rm = np.array([pool[i][1] if pool[i][1] is not None else np.nan for i in idx], dtype=float)
    return rf, rm


def _snap_seed(season, stat, snap):
    return zlib.crc32(f"{season}|{stat}|{snap}".encode()) & 0xFFFFFFFF


def _eff_matrix(stat, season, snap, field, counts, ctx_snap, vpriors, npriors,
                pools, tcut, pos, firstyr, k):
    rng = np.random.default_rng(_snap_seed(season, stat, snap))
    branch = BRANCH[stat]
    eff = np.zeros((len(field), k), dtype=float)
    season_tg = max((rc.get("ftg") or 0) for rc in ctx_snap.values()) if ctx_snap else 0
    for i, pid in enumerate(field):
        d = counts[(season, snap, pid)]
        rc = ctx_snap[pid]
        gp = d.get("gp_played_asof") or 0.0
        mn = d.get("min_asof") or 0.0
        banked = BANKED[stat](d)
        cohort = N._cohort(pid, season, d, pos, firstyr, vpriors["mpg_cuts"])
        rf, rm = _draw_availability(pools, rc, tcut, k, rng)
        if rf is None:
            continue
        if AVAIL_HIER and _AVAIL_PRIOR is not None:
            rf, rm = AH.recentre(rf, rm, rc, d, pid, season, _AVAIL_PRIOR)
        banked_mpg = (mn / gp) if gp else 0.0
        rm = np.where(np.isnan(rm), banked_mpg, rm)
        if GAMES_K is not None:
            tg_asof = rc["ftg"] - rc["rem_team"]
            avail = (gp / tg_asof) if tg_asof > 0 else 1.0
            rf = MIN.shrink_frac(rf, avail, tg_asof, GAMES_K)
        if MPG_K is not None:
            rm = MIN.shrink_mpg(rm, banked_mpg, mn, MPG_K)
        rem_games = rf * rc["rem_team"]
        rem_min = np.maximum(rem_games * rm, 0.0)
        rem_total = branch(rng, d, cohort, vpriors, npriors, rem_min, k)
        season_total = banked + rem_total
        season_games = gp + rem_games
        denom = np.maximum(season_games, _qual(season_tg))
        eff[i, :] = np.where(denom > 0, season_total / denom, 0.0)
    return eff


def _field_at(counts, ctx_snap, season, snap, stat, top_n):
    cand = []
    for pid in ctx_snap:
        d = counts.get((season, snap, pid))
        if not d:
            continue
        gp = d.get("gp_played_asof") or 0.0
        if gp < 1:
            continue
        cand.append((BANKED[stat](d) / gp, pid))
    cand.sort(reverse=True)
    return [pid for _, pid in cand[:top_n]]


def snapshot_probs(stat, season, snap, counts, ctx, vpriors, npriors, pools, tcut,
                   pos, firstyr, k=DEFAULT_K, field_n=FIELD_N, extra_field=None):
    """Return (field, p_lead[np], p_top3[np]) for one snapshot, or ([], None, None)."""
    ctx_snap = ctx.get(snap, {})
    field = _field_at(counts, ctx_snap, season, snap, stat, field_n)
    if extra_field:
        have = set(field)
        for pid in extra_field:
            if pid in have or pid not in ctx_snap:
                continue
            d = counts.get((season, snap, pid))
            if not d or (d.get("gp_played_asof") or 0.0) < 1:
                continue
            field.append(pid)
            have.add(pid)
    if not field:
        return [], None, None
    eff = _eff_matrix(stat, season, snap, field, counts, ctx_snap, vpriors, npriors,
                      pools, tcut, pos, firstyr, k)
    n = len(field)
    p_lead = np.bincount(np.argmax(eff, axis=0), minlength=n) / k
    kk = min(3, n)
    top = np.argpartition(-eff, kk - 1, axis=0)[:kk]
    p_top3 = np.bincount(top.ravel(), minlength=n) / k
    return field, p_lead, p_top3


def realised_eff(finals, ftg, season, stat):
    out = {}
    season_tg = max((v for v in ftg.values() if v), default=0)
    for (s, pid), d in finals.items():
        if s != season:
            continue
        f = ftg.get(pid)
        if not f:
            continue
        gp = d.get("gp_played_asof") or 0.0
        if gp < 1:
            continue
        out[pid] = BANKED[stat](d) / max(gp, _qual(season_tg))
    return out


_PRIOR_CACHE_DIR = "models/stat_leader"


def _fit_priors(conn, eval_season, fit_lookback):
    """The expensive, config-independent prior fit: gamma/cohort priors, availability
    pools, minutes/games shrinks and the avail-hier prior. A pure function of
    (eval_season, fit_lookback, data), cacheable and shared across books and runs. The
    avail-hier prior is always fitted here (not gated on AVAIL_HIER) so the cache is
    toggle-independent; _assemble decides whether to install it."""
    fit_lo = eval_season - fit_lookback
    fit_hi = eval_season - 1
    counts, finals, pos, firstyr = N._load(conn, list(range(fit_lo, eval_season + 1)))
    fit_counts = {k: v for k, v in counts.items() if fit_lo <= k[0] <= fit_hi}
    fit_finals = {k: v for k, v in finals.items() if fit_lo <= k[0] <= fit_hi}
    vpriors = V.fit_priors(fit_counts, fit_finals, pos, firstyr)
    npriors = N.fit_priors(fit_counts, fit_finals, pos, firstyr)
    train_recs = A._load(conn, list(range(fit_lo, fit_hi + 1)))
    tcut = A._terciles([r["avail_rate"] for r in train_recs])
    pools = A.fit(train_recs, tcut)
    mpg_k = MIN.fit_shrink_k(conn, train_recs, pools, tcut, list(range(fit_lo, fit_hi + 1)))
    games_k = MIN.fit_shrink_kg(conn, train_recs, pools, tcut, list(range(fit_lo, fit_hi + 1)))
    avail_prior = AH.fit(conn, list(range(fit_lo, fit_hi + 1)))
    return dict(vpriors=vpriors, npriors=npriors, pools=pools, tcut=tcut,
                mpg_k=mpg_k, games_k=games_k, pos=pos, firstyr=firstyr,
                avail_prior=avail_prior, fit_lo=fit_lo, fit_hi=fit_hi)


def _assemble(conn, eval_season, priors):
    """Cheap per-run assembly onto a fitted-prior bundle: a fresh full-range data read,
    the prior-season counts mutation (prior_min / prior_rate_*) the draw path consumes,
    context and ftg. Verbatim load_all tail, so a cached fit is byte-identical to a live
    one. Installs the avail-hier prior into the module global exactly as load_all did,
    only when AVAIL_HIER."""
    fit_lo, fit_hi = priors["fit_lo"], priors["fit_hi"]
    counts, finals, _pos, _firstyr = N._load(conn, list(range(fit_lo, eval_season + 1)))
    global _AVAIL_PRIOR
    if AVAIL_HIER:
        _AVAIL_PRIOR = priors["avail_prior"]
    global _OVERLAY_ART
    _OVERLAY_ART = {}
    for _bk in ("stl", "blk", "ast"):
        if OVERLAY.get(_bk):
            try:
                _OVERLAY_ART[_bk] = VO.load(_bk, eval_season)
            except Exception:
                _OVERLAY_ART[_bk] = None
    for (s, snap, pid), d in counts.items():
        fprev = finals.get((s - 1, pid))
        if not fprev:
            continue
        pm = fprev.get("min_asof") or 0.0
        if pm <= 0:
            continue
        d["prior_min"] = pm
        for node in ("reb", "usage", "ast_create", "stl", "blk", "defl", "rim_fga"):
            cnt, mn = V._vol_count(fprev, node)
            if cnt is not None and mn and mn > 0:
                d[f"prior_rate_{node}"] = cnt / mn
    ctx = _load_context(conn, eval_season)
    ftg = _load_ftg(conn, eval_season)
    return dict(counts=counts, finals=finals, pos=priors["pos"], firstyr=priors["firstyr"],
                vpriors=priors["vpriors"], npriors=priors["npriors"],
                pools=priors["pools"], tcut=priors["tcut"],
                ctx=ctx, ftg=ftg, mpg_k=priors["mpg_k"], games_k=priors["games_k"],
                fit_lo=fit_lo, fit_hi=fit_hi)


def _prior_fingerprint(conn, fit_lo, eval_season):
    r = conn.execute(
        "SELECT COUNT(*) c, COALESCE(MAX(game_date),'') m FROM stg_nba_player_game_logs "
        "WHERE season BETWEEN ? AND ?", (fit_lo, eval_season)).fetchone()
    return f"{r['c']}_{r['m']}"


def _prior_path(conn, eval_season, fit_lookback):
    fit_lo = eval_season - fit_lookback
    fp = _prior_fingerprint(conn, fit_lo, eval_season)
    return os.path.join(_PRIOR_CACHE_DIR, f"priorfit_s{eval_season}_lb{fit_lookback}_{fp}.pkl")


def _cached_priors(conn, eval_season, fit_lookback):
    """Fit-once prior cache: fit and pickle if absent, else load. Fingerprinted on the
    game-log count and max date over the fit range, so a data pull invalidates it;
    STAT_PRIOR_REFIT=1 forces a rebuild. Distinct seasons write distinct files, and the
    temp name is pid-tagged so parallel book workers cannot corrupt one another."""
    os.makedirs(_PRIOR_CACHE_DIR, exist_ok=True)
    path = _prior_path(conn, eval_season, fit_lookback)
    if os.path.exists(path) and not os.environ.get("STAT_PRIOR_REFIT"):
        with open(path, "rb") as fh:
            return pickle.load(fh)
    priors = _fit_priors(conn, eval_season, fit_lookback)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "wb") as fh:
        pickle.dump(priors, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    return priors


def load_all(conn, eval_season, fit_lookback=10):
    """Assemble the scoring bundle B for eval_season. The prior fit is cached to disk
    (fit once, reuse across books and runs); the per-run assembly is always fresh.
    Byte-identical to the pre-cache load_all for any given (season, lookback, data)."""
    priors = _cached_priors(conn, eval_season, fit_lookback)
    return _assemble(conn, eval_season, priors)


def _seal_check(stat, season):
    try:
        from scripts.common.config import assert_not_sealed
    except ImportError:
        from config import assert_not_sealed  # type: ignore
    assert_not_sealed(STAT_AWARD[stat], season)


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Stat-leader Monte Carlo P(lead) engine.")
    p.add_argument("--db", default="data/awards.db")
    p.add_argument("--stat", default="all", choices=["reb", "pts", "ast", "stl", "blk", "all"])
    p.add_argument("--eval-season", type=int, default=2023)
    p.add_argument("--fit-lookback", type=int, default=10)
    p.add_argument("--k", type=int, default=DEFAULT_K)
    p.add_argument("--field-n", type=int, default=FIELD_N)
    args = p.parse_args(argv)

    stats = ["reb", "pts", "ast", "stl", "blk"] if args.stat == "all" else [args.stat]
    for st in stats:
        _seal_check(st, args.eval_season)

    conn = connect(args.db)
    B = load_all(conn, args.eval_season, args.fit_lookback)
    conn.close()
    globals()["MPG_K"] = B["mpg_k"]
    globals()["GAMES_K"] = B["games_k"]
    log.info("measured shrinks: minutes K=%.0f games Kg=%.0f; own_prior_K=%s",
             B["mpg_k"], B["games_k"], OWN_PRIOR_K)

    snaps = sorted({snap for (s, snap, _) in B["counts"] if s == args.eval_season})
    if not snaps:
        log.error("no snapshots for season %d", args.eval_season); return 1
    probe = [snaps[len(snaps) // 5], snaps[len(snaps) // 2], snaps[-2]]

    for st in stats:
        node = VOL_NODE[st]
        log.info("stat=%s eval=%d fit=%d-%d k=%d field=%d fano=%.2f", st, args.eval_season,
                 B["fit_lo"], B["fit_hi"], args.k, args.field_n,
                 B["vpriors"].get("fano", {}).get(node, 1.0))
        real = sorted(realised_eff(B["finals"], B["ftg"], args.eval_season, st).items(),
                      key=lambda kv: kv[1], reverse=True)[:5]
        print("\n" + "=" * 60)
        print(f"MC sanity  stat={st}  eval={args.eval_season}")
        print("realised end-of-season leaders (eff_value):")
        for pid, eff in real:
            print(f"  {pid:>10}  {eff:.3f}")
        print("=" * 60)
        for snap in probe:
            field, p_lead, p_top3 = snapshot_probs(
                st, args.eval_season, snap, B["counts"], B["ctx"], B["vpriors"],
                B["npriors"], B["pools"], B["tcut"], B["pos"], B["firstyr"],
                args.k, args.field_n)
            if not field:
                continue
            order = np.argsort(-p_lead)[:5]
            print(f"\nsnapshot {snap}  field={len(field)}  sum P(lead)={p_lead.sum():.3f}")
            for j in order:
                print(f"  {field[j]:>10}  P(lead)={p_lead[j]:.3f}  P(top3)={p_top3[j]:.3f}")
    print("\n" + "=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
