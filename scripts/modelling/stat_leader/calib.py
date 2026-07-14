"""Stat-leader arm: calibration primitives (filter, Beta map, diagnostics).

Pure functions, no I/O, no parallelism, so the maths can be checked in isolation.

reachability_variants: from one eff matrix (contenders x draws), compute baseline
P(lead)/P(top3) and the filtered versions for a list of catch-up quantiles in a
single pooled pass. A trailing candidate is dropped when his optimistic finish
(the 1-q quantile of his eff draws) cannot reach the front-runner's pessimistic
finish (the q quantile of the leader's eff draws); P(lead) is then recomputed as
the argmax over the admitted set only, so dropped names get exactly zero and the
admitted vector still sums to one. Smaller q compares more extreme tails and
drops fewer; larger q is stricter. The field narrows on its own as games run out.

beta_fit / beta_apply: Beta calibration as logistic regression in log-odds space.
    logit(p_cal) = a(f)*log(p) + b(f)*log(1-p) + c(f)
with each coefficient linear in the remaining-games fraction f, so the map is a
continuous function of both the probability and the season stage (no buckets).
Two slopes on log(p) and log(1-p) let it bend into an S, pulling the overconfident
left tail down while lifting the underconfident middle up, which a single-slope
map or a temperature cannot. Fitted by IRLS to log-loss on per-candidate binary
outcomes (marginal calibration); the exclusivity constraint is imposed at apply
time by renormalising within the snapshot. Temperature is the degenerate case
where the two slopes are tied and c=0, which is why it failed on this S-shape.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-6


# --------------------------------------------------------------------------- #
# reachability filter (pooled over quantiles)
# --------------------------------------------------------------------------- #

def _plead_ptop3(eff):
    k = eff.shape[1]
    lead = np.bincount(np.argmax(eff, axis=0), minlength=eff.shape[0]) / k
    t3 = np.zeros(eff.shape[0])
    order = np.argsort(-eff, axis=0)[:3, :]
    for r in range(min(3, eff.shape[0])):
        t3 += np.bincount(order[r], minlength=eff.shape[0])
    return lead, t3 / k


def reachability_variants(eff, qs):
    """Return {'base': (plead, ptop3, n)} plus {q: (plead, ptop3, n_admit)} for
    each q, all from the one eff matrix. plead/ptop3 are full-length (dropped
    candidates carry zero)."""
    F, k = eff.shape
    out = {}
    base_l, base_3 = _plead_ptop3(eff)
    out["base"] = (base_l, base_3, F)
    med = np.median(eff, axis=1)
    leader = int(np.argmax(med))
    hi = np.quantile(eff, 1.0 - np.array(qs), axis=1)   # (len(qs), F) optimistic per candidate
    lead_pess = np.quantile(eff[leader], qs)            # (len(qs),) leader pessimistic
    for j, q in enumerate(qs):
        admit = hi[j] >= lead_pess[j]
        admit[leader] = True
        idx = np.where(admit)[0]
        sub = eff[idx]
        l = np.zeros(F)
        t = np.zeros(F)
        sl, st = _plead_ptop3(sub)
        l[idx] = sl
        t[idx] = st
        out[q] = (l, t, int(idx.size))
    return out


# --------------------------------------------------------------------------- #
# Beta calibration (IRLS logistic regression, log-loss)
# --------------------------------------------------------------------------- #

def _design(p, f):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    f = np.asarray(f, float)
    lp = np.log(p)
    lq = np.log(1 - p)
    ones = np.ones_like(p)
    return np.column_stack([lp, lp * f, lq, lq * f, ones, f])   # 6 features


def beta_fit(p, y, f, ridge=1e-3, iters=50):
    """IRLS fit of the stage-conditioned Beta map to log-loss. Returns a 6-vector
    of coefficients (order: logp, logp*f, log(1-p), log(1-p)*f, 1, f)."""
    X = _design(p, f)
    y = np.asarray(y, float)
    w = np.zeros(X.shape[1])
    R = ridge * np.eye(X.shape[1])
    for _ in range(iters):
        eta = np.clip(X @ w, -30, 30)
        mu = 1.0 / (1.0 + np.exp(-eta))
        s = np.clip(mu * (1 - mu), 1e-6, None)
        grad = X.T @ (mu - y) + ridge * w
        H = X.T @ (X * s[:, None]) + R
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            break
        w_new = w - step
        if not np.all(np.isfinite(w_new)):
            break
        if np.max(np.abs(w_new - w)) < 1e-8:
            w = w_new
            break
        w = w_new
    return w


def beta_apply(p, f, w):
    """Apply the fitted map (pre-renormalisation). Renormalise within a snapshot
    afterwards for the exclusive P(lead) book."""
    eta = np.clip(_design(p, f) @ w, -30, 30)
    return 1.0 / (1.0 + np.exp(-eta))


def renorm_snapshot(pcal):
    s = float(np.sum(pcal))
    return pcal / s if s > 0 else pcal


# --------------------------------------------------------------------------- #
# diagnostics
# --------------------------------------------------------------------------- #

def signed_calib_error(p, y, lo, hi):
    """Signed calibration error over predicted band [lo, hi): mean(p) - mean(y).
    Positive = overconfident (predicts more than realises)."""
    p = np.asarray(p, float)
    y = np.asarray(y, float)
    m = (p >= lo) & (p < hi)
    if not m.any():
        return float("nan"), 0
    return float(p[m].mean() - y[m].mean()), int(m.sum())


def band_report(p, y):
    """Tail / middle / top signed calibration errors."""
    return {
        "tail(0-0.25)": signed_calib_error(p, y, 0.0, 0.25),
        "mid(0.25-0.70)": signed_calib_error(p, y, 0.25, 0.70),
        "top(0.70-1.0)": signed_calib_error(p, y, 0.70, 1.0001),
    }


def reliability_curve(p, y, nbins=12):
    """Quantile-binned reliability curve: arrays (pred_mean, emp_rate, n) per
    non-empty bin, for predicted-vs-realised plots."""
    p = np.asarray(p, float)
    y = np.asarray(y, float)
    if p.size == 0:
        return np.array([]), np.array([]), np.array([])
    edges = np.unique(np.quantile(p, np.linspace(0, 1, nbins + 1)))
    if edges.size < 2:
        return np.array([p.mean()]), np.array([y.mean()]), np.array([p.size])
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, edges.size - 2)
    pm, em, ns = [], [], []
    for b in range(edges.size - 1):
        m = idx == b
        if m.any():
            pm.append(p[m].mean())
            em.append(y[m].mean())
            ns.append(int(m.sum()))
    return np.array(pm), np.array(em), np.array(ns)
