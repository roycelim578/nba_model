"""jspeak_floor lift, composed into renorm_set's masked-cloud reshape.

HANDOFF_jspeak_floor_complete section 5 left the pricing-path binding open. This is the
lift half of it: given a contender set S already chosen by renorm_set (union mode) and a
masked cloud (out-of-S rows set to -inf), lift the favourite's price toward the first-place
model's point without disturbing S, VS's ranking, or the union tail, and floor the lift so
FP can only add confidence and never hedge the favourite down.

It is deliberately NOT a second reshape path. renorm_set owns the union renormalisation and
the single masked cloud; this module only adds a per-candidate constant to the IN-S rows of
that cloud so its softmax_of_mean becomes the jspeak_floor'd vector, leaving every candidate's
across-column variance unchanged (a constant shift cancels in the variance). renorm_set then
recomputes vote_share_pred and the sizing weights from the one reshaped cloud as it already
does, so the edge and the wealth object cannot drift apart.

Order of operations (confirmed with the strategy chat, late-sharpen temperature dropped):
  union renorm (renorm_set) THEN jspeak lift (here), both over S. fp is renormalised over S
  exactly as VS is, and any name the FP fold did not score falls back to its VS value so the
  lift is a no-op there.

British English. No inline comments.
"""
from __future__ import annotations

import numpy as np


def _js2(p, q):
    """Base-2 Jensen-Shannon divergence of two prob vectors, in [0, 1]."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    sp, sq = p.sum(), q.sum()
    if sp <= 0 or sq <= 0:
        return 1.0
    p = p / sp
    q = q / sq
    m = 0.5 * (p + q)

    def _kl(a, b):
        k = a > 0
        return float(np.sum(a[k] * np.log2(a[k] / b[k])))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def jspeak_floor_over_union(vsp, fp_vec, umask):
    """VS ranking and union tail kept; favourite peak lifted toward FP, floored so FP may
    only add. Both vectors are renormalised over the union. A NaN or missing FP entry is
    replaced by the VS value for that name so it neither lifts nor hedges. Returns a
    full-length vector that is zero off the union and sums to 1 over it."""
    vsp = np.asarray(vsp, dtype=float)
    fp_vec = np.asarray(fp_vec, dtype=float)
    umask = np.asarray(umask, dtype=bool)

    v = np.where(umask, np.nan_to_num(vsp, nan=0.0), 0.0)
    sv = v.sum()
    if sv <= 0:
        p = np.zeros_like(vsp)
        idx = np.flatnonzero(umask)
        if idx.size:
            p[idx] = 1.0 / idx.size
        return p
    vu = v / sv

    f = np.where(np.isfinite(fp_vec), fp_vec, vsp)
    f = np.where(umask, np.nan_to_num(f, nan=0.0), 0.0)
    sf = f.sum()
    fu = (f / sf) if sf > 0 else vu.copy()

    t = int(np.argmax(vu))
    if umask.sum() < 2:
        return vu

    w = 1.0 - _js2(fu, vu)
    peak = max(vu[t], w * fu[t] + (1.0 - w) * vu[t])

    tail = vu.copy()
    tail[t] = 0.0
    s_tail = tail.sum()
    if s_tail > 0:
        tail = tail * (1.0 - peak) / s_tail
    tail[t] = peak
    return tail


def jspeak_reshape_masked(masked_cloud, S, fp_vec):
    """Add the jspeak lift to the in-S rows of a masked cloud.

    masked_cloud   (ncand, K*M), out-of-S rows already set to -inf by renorm_set.
    S              bool contender mask (the union set).
    fp_vec         first-place point per candidate, aligned to rows; NaN where unscored.

    Returns (reshaped_cloud, p_star) where softmax_of_mean(reshaped_cloud) over S equals
    p_star, p_star sums to 1 over S and is zero off it, and in-S column variance is preserved.
    """
    S = np.asarray(S, dtype=bool)
    cl = np.asarray(masked_cloud, dtype=float).copy()
    ncand = cl.shape[0]
    if not S.any():
        return cl, np.zeros(ncand)

    mu_S = cl[S].mean(axis=1)
    e = np.exp(mu_S - mu_S.max())
    vsp_S = np.zeros(ncand)
    vsp_S[S] = e / e.sum()

    p_star = jspeak_floor_over_union(vsp_S, fp_vec, S)
    delta = np.log(np.clip(p_star[S], 1e-300, None)) - mu_S
    cl[S] = cl[S] + delta[:, None]
    return cl, p_star
