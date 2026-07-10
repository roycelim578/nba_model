"""
Portfolio-Kelly sizer, v2. Drop-in successor to backtest_sizer.solve_award. Three
changes from v1, each independently switchable so its backtest effect is isolable.

  1. OBJECT CONSISTENCY. Winner-state weights q are the CALIBRATED central estimate
     softmax_of_mean (vote_share_pred), not mean_of_softmax. mean_of_softmax is flatter
     (eta and ensemble noise regress it toward uniform), which suppressed favourites and
     lifted the pack and so shorted names the model prices as underpriced. The ensemble
     dispersion is not thrown away; it is left to the risk layer (forward_edge CVaR and
     the price-dispersion scaler), where uncertainty belongs, not folded into the point
     estimate. central_weights selects the object; default vote_share_pred.

  2. WARM START + TURNOVER IN THE OBJECTIVE. v1 re-solved from scratch each snapshot on
     a non-convex surface, so the target thrashed week to week for non-fundamental
     reasons and the downstream no-trade band (which only catches dust) could not stop
     it. Here the objective charges the round-trip cost of moving off a_prev and the
     first restart warm-starts from a_prev, so the target stays put unless the edge
     change earns the move. Stabilises the RAW target, which is where the thrash lives.

  3. FADE-YOUR-OWN-TOP-K GUARD. Forbid a NO leg on the model's own top-k by
     vote_share_pred. Removes the favourite-short entry error at source; inert once the
     model sharpens.

TAIL / CONCENTRATION RISK is NOT handled here by a cap. It is a smooth multiplicative
scaler in size_scaling.py, applied at the same layer as the forward-vol dispersion
scaler, so every handoff stays a continuous fade. This sizer returns the RAW signed
Kelly allocation; the scaling layer shrinks it. British English.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize


@dataclass
class SizerResultV2:
    allocation: np.ndarray
    raw_allocation: np.ndarray
    exp_log_growth: float
    deployed: float
    player_ids: list
    converged: bool
    restart_spread: float
    turnover_usd: float


def rank_floor_mask(vote_share_pred, award):
    n_by_award = {"DPOY": 20, "MVP": 15, "ROTY": 10}
    n = n_by_award[award]
    v = np.asarray(vote_share_pred, dtype=float)
    mask = np.zeros(v.size, dtype=bool)
    mask[np.argsort(-v)[:n]] = True
    return mask


def _make_objective(q, yes_price_fn, no_price_fn, portfolio, ncand, a_prev, turnover_frac):
    """Negative expected log wealth over the ncand winner states weighted by q, minus a
    certain turnover toll for moving off a_prev. The toll is paid in every state, so it
    enters base wealth. turnover_frac[i] is the round-trip cost fraction on the dollar
    change in name i. Ruin in any q>0 state returns a large positive (infeasible)."""
    q = np.asarray(q, dtype=float)
    a_prev = np.asarray(a_prev, dtype=float)
    turnover_frac = np.asarray(turnover_frac, dtype=float)

    def neg_growth(a):
        deployed = np.abs(a)
        eff = np.empty(ncand)
        for i in range(ncand):
            if a[i] >= 0:
                eff[i] = yes_price_fn[i](deployed[i]) if deployed[i] > 0 else 1.0
            else:
                eff[i] = no_price_fn[i](deployed[i]) if deployed[i] > 0 else 1.0
        eff = np.clip(eff, 1e-4, 0.9999)
        shares = deployed / eff
        outlay = deployed.sum()
        turnover = float(np.sum(turnover_frac * np.abs(a - a_prev)))
        yes_leg = a > 0
        no_leg = a < 0
        base = portfolio - outlay + shares[no_leg].sum() - turnover
        W = np.full(ncand, base)
        W[no_leg] -= shares[no_leg]
        W[yes_leg] += shares[yes_leg]
        m = q > 0
        if np.any(W[m] <= 0):
            return 1e6
        return -np.sum(q[m] * np.log(W[m] / portfolio))

    return neg_growth


def solve_award_v2(samples, cost_curves, portfolio_usd=1000.0, award_budget=None,
                   kelly_fraction=1.0, n_restarts=6, seed=0, award=None,
                   tradeable_mask=None, central_weights="vote_share_pred",
                   a_prev=None, turnover_frac=None, turnover_default=0.0,
                   guard_top_k=0):
    """
    central_weights: "vote_share_pred" (default, calibrated) or "sizing_weights" (the old
      mean_of_softmax, for A/B).
    a_prev: signed current allocation per candidate (dollars). None -> zeros (cold).
    turnover_frac / turnover_default: per-name round-trip cost fraction for the turnover
      toll (0.0 reproduces v1's no-toll behaviour).
    guard_top_k: forbid a NO leg on the top-k names by vote_share_pred (0 -> off).
    kelly_fraction defaults to 1.0 here; the shrink is delegated to size_scaling.
    """
    ncand = len(samples.player_ids)
    if award_budget is None:
        award_budget = portfolio_usd

    if central_weights == "sizing_weights":
        q = np.asarray(samples.sizing_weights, dtype=float)
    else:
        q = np.asarray(samples.vote_share_pred, dtype=float)

    vsp = np.asarray(samples.vote_share_pred, dtype=float)
    if tradeable_mask is None and award is not None:
        tradeable_mask = rank_floor_mask(vsp, award)
    if tradeable_mask is None:
        tradeable_mask = np.ones(ncand, dtype=bool)
    tradeable_mask = np.asarray(tradeable_mask, dtype=bool)

    if a_prev is None:
        a_prev = np.zeros(ncand)
    a_prev = np.asarray(a_prev, dtype=float)
    if turnover_frac is None:
        turnover_frac = np.full(ncand, float(turnover_default))
    turnover_frac = np.asarray(turnover_frac, dtype=float)

    guard_no = np.zeros(ncand, dtype=bool)
    if guard_top_k and guard_top_k > 0:
        guard_no[np.argsort(-vsp)[:guard_top_k]] = True

    yes_fns = [cost_curves[i]["yes"] for i in range(ncand)]
    no_fns = [cost_curves[i]["no"] for i in range(ncand)]
    neg_growth = _make_objective(q, yes_fns, no_fns, portfolio_usd, ncand, a_prev, turnover_frac)

    def budget_con(a):
        return award_budget - np.abs(a).sum()

    cons = [{"type": "ineq", "fun": budget_con}]

    bounds = []
    for i in range(ncand):
        if not tradeable_mask[i]:
            bounds.append((0.0, 0.0))
        elif guard_no[i]:
            bounds.append((0.0, award_budget))
        else:
            bounds.append((-award_budget, award_budget))

    lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
    rng = np.random.default_rng(seed)
    best = None
    vals = []
    for r in range(n_restarts):
        if r == 0:
            x0 = a_prev.copy()
        elif r == 1:
            x0 = np.zeros(ncand)
        else:
            x0 = a_prev + rng.normal(scale=award_budget / (4 * ncand), size=ncand)
        x0 = np.clip(x0, lo, hi)
        res = minimize(neg_growth, x0, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-9})
        if res.success or res.status == 4:
            vals.append(-res.fun)
            if best is None or -res.fun > -best.fun:
                best = res
    if best is None:
        raw = np.zeros(ncand)
        return SizerResultV2(raw.copy(), raw, 0.0, 0.0, samples.player_ids, False, 0.0, 0.0)

    raw = best.x.copy()
    alloc = raw * kelly_fraction
    spread = (max(vals) - min(vals)) if len(vals) > 1 else 0.0
    turnover_usd = float(np.sum(np.abs(alloc - a_prev * kelly_fraction)))
    return SizerResultV2(
        allocation=alloc, raw_allocation=raw, exp_log_growth=float(-best.fun),
        deployed=float(np.abs(alloc).sum()), player_ids=samples.player_ids,
        converged=True, restart_spread=float(spread), turnover_usd=turnover_usd)


def flat_cost_curves(mids_yes):
    curves = []
    for m in mids_yes:
        m = float(np.clip(m, 1e-3, 1 - 1e-3))
        curves.append({"yes": (lambda s, m=m: m), "no": (lambda s, m=m: 1.0 - m)})
    return curves


def sqrt_cost_curves(mids_yes, c=0.5, near_touch=97.0, bowl=0.02):
    curves = []
    for m in mids_yes:
        m = float(np.clip(m, 1e-3, 1 - 1e-3))
        def mk(px):
            return lambda s, px=px: px * (1 + bowl) * (1 + c * np.sqrt(max(s, 0) / near_touch))
        curves.append({"yes": mk(m), "no": mk(1.0 - m)})
    return curves
