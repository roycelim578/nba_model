"""Soft-outcome consistency for the portfolio-Kelly sizer (master spec 2a / 6).

Two weightings are derived from the SAME K-booster (times eta) score cloud, and they
answer two different questions:

  edge_weights    softmax_of_mean. The reported point win-probability, differenced
                  against price to quote EDGE. This is the better-calibrated point
                  estimate (spec section 3): it resolves the eventual winner sharply
                  instead of staying artificially flat.
  sizing_weights  mean_of_softmax. The frequency with which the cloud actually says
                  each candidate wins. This is the probability the log-Kelly average
                  must weight outcomes by, and it is NOT a free choice: for a
                  hold-to-resolution position wealth depends only on WHO wins, so the
                  log-Kelly expectation collapses analytically to sum_i q_i log W_i
                  with q = mean_of_softmax. Substituting softmax_of_mean here would
                  plug the wrong probability into a fixed formula.

The analytic sum_i q_i log W_i is the Rao-Blackwellised form of resampling a hard
winner from each draw's softmax: identical target, zero Monte-Carlo variance. The
hard argmax-per-draw winner is RETIRED (it turned a small score margin into a full
winner flip, the discretisation artefact behind the early SGA flattening that spec
section 4 disowns). resampled_sizing_weights below exists only to demonstrate the
equivalence in tests; the sizer never calls it.

Edge is quoted on softmax_of_mean but SIZED on mean_of_softmax. For a favourite that
leads in most draws these differ, mean_of_softmax being the flatter (closer to
uniform), so the sized stake is smaller than the quoted edge implies. Conservative,
and in the same safe direction as the price-path independence assumption: it can only
under-size, never manufacture edge. The one case to guard is a candidate whose sizing
weight and edge weight straddle the market price, where the sized side flips against
the quoted side. The magnitude is self-limiting (such a position sits near a sign
change, so Kelly stakes it tiny), but the sizer must take the side implied by the
SIZING weight, never silently the quoted side.
"""

import numpy as np


def _softmax(scores, axis=-1):
    m = np.max(scores, axis=axis, keepdims=True)
    e = np.exp(scores - m)
    return e / np.sum(e, axis=axis, keepdims=True)


def edge_weights(cloud):
    """softmax_of_mean over the draw axis. cloud shape [n_draws, n_candidates]."""
    cloud = np.asarray(cloud, dtype=float)
    return _softmax(cloud.mean(axis=0))


def sizing_weights(cloud):
    """mean_of_softmax over the draw axis. cloud shape [n_draws, n_candidates]."""
    cloud = np.asarray(cloud, dtype=float)
    return _softmax(cloud, axis=1).mean(axis=0)


def resampled_sizing_weights(cloud, draws_per_row=1, rng=None):
    """Noisy unbiased estimator of sizing_weights, sampling a hard winner per draw.

    Present only to show the Rao-Blackwell equivalence with sizing_weights in tests.
    Never used in the sizer, which takes the analytic mean_of_softmax directly.
    """
    cloud = np.asarray(cloud, dtype=float)
    rng = np.random.default_rng() if rng is None else rng
    probs = _softmax(cloud, axis=1)
    n_draws, n_cand = probs.shape
    counts = np.zeros(n_cand)
    for r in range(n_draws):
        picks = rng.choice(n_cand, size=draws_per_row, p=probs[r])
        for w in picks:
            counts[w] += 1
    return counts / counts.sum()


def wealth_if_winner(alloc, eff_price, portfolio):
    """Wealth in each resolved state for a signed allocation, hold-to-resolution.

    alloc      signed dollars per candidate; a_i > 0 backs YES, a_i < 0 backs NO.
    eff_price  effective per-share price on the leg actually taken (YES token for
               a_i > 0, NO token about 1 - yes_mid for a_i < 0), cost inside.
    portfolio  total book dollars.

    Returns W with W[w] the terminal wealth if candidate w wins. Settlement is free,
    so no exit cost enters here; this is the base hold-to-resolution case.
    """
    alloc = np.asarray(alloc, dtype=float)
    eff_price = np.asarray(eff_price, dtype=float)
    n = alloc.size
    idx = np.arange(n)
    staked = np.abs(alloc).sum()
    shares = np.abs(alloc) / eff_price
    is_yes = alloc > 0
    is_no = alloc < 0
    W = np.full(n, portfolio - staked, dtype=float)
    for w in range(n):
        payoff = shares[w] if is_yes[w] else 0.0
        payoff += shares[is_no & (idx != w)].sum()
        W[w] += payoff
    return W


def expected_log_growth(alloc, q_size, eff_price, portfolio):
    """Soft-outcome hold-to-resolution objective: sum_w q_w log(W_w) - log(portfolio).

    q_size is mean_of_softmax. Ruin in any weighted state (W_w <= 0 where q_w > 0)
    returns -inf, forbidding the allocation. This is the object the portfolio-Kelly
    solver maximises for settle legs; the composed forward-edge build (spec 6.5) wraps
    an outer average over price-path draws for legs exited before resolution.
    """
    q = np.asarray(q_size, dtype=float)
    W = wealth_if_winner(alloc, eff_price, portfolio)
    if np.any((q > 0) & (W <= 0)):
        return -np.inf
    return float(np.sum(q * np.log(W)) - np.log(portfolio))
