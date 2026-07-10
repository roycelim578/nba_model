"""Composed forward-edge risk object (master spec 6.5, rebuilt).

The horizon SELECTOR is dead: ill-posed against a price model with no drift toward our
private fair value, and banking any drift the vol model does have would be circular (it
drifts toward the market base rate, not our pwin). Hold-to-resolution with reactive trimming
on the weekly re-solve is the base case; convergence is captured when it happens, never
forecast.

What survives is a RISK object. Two penalties attach to a position, on two variances:
  outcome variance  the settlement lottery paying zero (wrong about who wins),
  price variance    capital-time spent at adverse levels while early (right but early).
Settlement wealth cannot carry price variance (the payoff depends only on the winner), so
price variance enters sizing as a capital-time penalty, not as a term in the log-wealth.

Construction, per candidate:
  expected edge = pwin (mean_of_softmax) - current_price - entry_cost, drift-free, no drift
    banked (non-circular).
  price-move distribution = SURVIVAL-WEIGHTED MIXTURE over the vol model's own N-horizon
    net-move distributions (not a single N, not a Gaussian variance sum). Weight on the
    k-step distribution is the probability the position is still held at step k, a
    constant-hazard survival curve; the exponential is the myopic special case of the v2
    backward-induction survival-weighted exposure. Log-odds moves are DE-MEANED (drift
    stripped, skew kept), so E[future_price] = current_price.
  outcome and price draws are composed with a coupling correlation coupling_rho:
    0.0 = INDEPENDENCE (conservative default: widens the downside only), positive =
    INTERRELATED (shared news lifts pwin and price together, cancels in pwin - price,
    tightens CVaR). Both marginals preserved, so expected edge is invariant to coupling_rho.

Read off: expected_edge (mean), cvar_downside (mean of the adverse tail beyond the 5th
percentile, positive, asymmetry-aware where symmetric sigma is not), risk_adjusted_edge =
expected_edge / cvar_downside. price_dispersion exposes the price-only spread that keys the
sizer's fill fraction (so outcome variance is not double-counted in the size).
"""

import numpy as np

from scripts.strategy.sizing.soft_outcome import _softmax


def per_draw_pwin(cloud, cand_idx):
    cloud = np.asarray(cloud, dtype=float)
    return _softmax(cloud, axis=1)[:, cand_idx]


def _rank(x):
    return np.argsort(np.argsort(x))


def survival_weights(half_life, n_max):
    ks = np.arange(1, n_max + 1)
    w = 0.5 ** ((ks - 1) / max(half_life, 1e-6))
    return w / w.sum()


CONV_HL_EARLY = 17.0
CONV_HL_LATE = 9.0


def convergence_half_life(frac):
    """Survival half-life for the fill window, linear in frac from CONV_HL_EARLY at the
    season start to CONV_HL_LATE at resolution. Pinned from the forward-vol corpus, which
    measured the distance-to-resolution to halve in ~17 steps early, ~13 mid, ~9 late:
    early positions sit in quiet, illiquid, information-poor markets and are held far longer,
    so they carry price risk over a two-to-three-week window, not a two-day one. This is the
    price-process holding horizon and it supersedes the cost-default as the half-life source."""
    f = min(max(frac, 0.0), 1.0)
    return CONV_HL_EARLY + (CONV_HL_LATE - CONV_HL_EARLY) * f


def half_life_from_cost(expected_edge, round_trip_cost, lo=1.0, hi=30.0):
    """SUPERSEDED as the fill half-life by convergence_half_life (corpus-measured holding
    horizon). Retained only for the worth-holding logic in the gate / no-trade band, never
    as the price-risk window."""
    if round_trip_cost <= 0:
        return hi
    return float(np.clip(max(expected_edge, 1e-6) / round_trip_cost, lo, hi))


def _mixture_moves(vol_model, price, frac, history, weights, n_draws, rng):
    counts = np.maximum(1, np.round(weights * n_draws).astype(int))
    parts = [np.asarray(vol_model.sample_moves(k, price, frac, history, c,
                                               seed=int(rng.integers(0, 2**63 - 1))), dtype=float)
             for k, c in enumerate(counts, start=1)]
    moves = np.concatenate(parts)
    return moves - moves.mean()


def _to_future_price(moves, price):
    lo = np.log(price / (1 - price)) + moves
    return np.clip(1.0 / (1.0 + np.exp(-lo)), 1e-4, 1 - 1e-4)


def _draw_future_and_pwin(cloud, cand_idx, current_price, vol_model, frac, history,
                          half_life, n_max, n_draws, rng, coupling_rho, central_pwin=None):
    weights = survival_weights(half_life, n_max)
    moves = _mixture_moves(vol_model, current_price, frac, history, weights, n_draws, rng)
    pw_pool = per_draw_pwin(cloud, cand_idx)
    if central_pwin is not None:
        pw_pool = np.clip(pw_pool - pw_pool.mean() + float(central_pwin), 1e-4, 1 - 1e-4)
    n = moves.size
    if coupling_rho <= 0.0:
        pw = rng.choice(pw_pool, size=n, replace=True)
    else:
        z = rng.standard_normal(n)
        w = rng.standard_normal(n)
        m = coupling_rho * z + np.sqrt(max(0.0, 1.0 - coupling_rho ** 2)) * w
        pw = np.quantile(pw_pool, (_rank(z) + 0.5) / n)
        moves = np.sort(moves)[_rank(m)]
    return _to_future_price(moves, current_price), pw


def composite_edge_draws(cloud, cand_idx, current_price, entry_cost, vol_model, frac,
                         history, half_life=None, n_max=25, side="yes", n_draws=8000, rng=None,
                         coupling_rho=0.0, central_pwin=None):
    rng = np.random.default_rng() if rng is None else rng
    if half_life is None:
        half_life = convergence_half_life(frac)
    future_yes, pw = _draw_future_and_pwin(cloud, cand_idx, current_price, vol_model, frac,
                                           history, half_life, n_max, n_draws, rng, coupling_rho,
                                           central_pwin=central_pwin)
    if side == "yes":
        return pw - future_yes - entry_cost
    return (1.0 - pw) - (1.0 - future_yes) - entry_cost


def price_dispersion(current_price, vol_model, frac, history, half_life=None, n_max=25,
                     n_draws=8000, rng=None):
    """Std of the survival-mixture future price. The price-only spread that keys the fill
    fraction, keeping outcome variance out of the size scalar."""
    rng = np.random.default_rng() if rng is None else rng
    if half_life is None:
        half_life = convergence_half_life(frac)
    weights = survival_weights(half_life, n_max)
    moves = _mixture_moves(vol_model, current_price, frac, history, weights, n_draws, rng)
    return float(_to_future_price(moves, current_price).std())


RADJ_CVAR_FLOOR = 0.05
RADJ_ABS_CAP = 10.0


def cvar_downside(edges, q_low=0.05):
    edges = np.asarray(edges, dtype=float)
    thr = np.quantile(edges, q_low)
    tail = edges[edges <= thr]
    if tail.size == 0:
        tail = np.array([thr])
    return float(max(1e-6, -tail.mean()))


def sigma_downside(edges, z=1.645):
    edges = np.asarray(edges, dtype=float)
    return float(max(1e-6, -(edges.mean() - z * edges.std())))


def read_off(edges, q_low=0.05):
    edges = np.asarray(edges, dtype=float)
    exp = float(edges.mean())
    ds = cvar_downside(edges, q_low)
    radj = float(np.clip(exp / max(ds, RADJ_CVAR_FLOOR), -RADJ_ABS_CAP, RADJ_ABS_CAP))
    return dict(expected_edge=exp, cvar_downside=ds, risk_adjusted_edge=radj)


def gate(risk_adjusted_edge, hurdle):
    return risk_adjusted_edge > hurdle
