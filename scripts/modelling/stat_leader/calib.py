"""Stat-leader arm: P(lead) calibration (Beta map, walk-forward driver, diagnostics).

Pure functions plus one orchestration driver, no I/O and no parallelism, so the
maths can be checked in isolation.

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

calibrate_plead_walkforward: the locked P(lead) calibration procedure. Fit the map
on strictly-earlier seasons, apply per snapshot, renormalise within the snapshot.
This is the single entry point the pricing/backtest layer calls.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

EPS = 1e-6


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


def beta_fit_cov(p, y, f, ridge=1e-3, iters=50):
    """As beta_fit, and additionally the coefficient covariance inv(H) at the
    optimum. H = X^T W X + ridge*I is the ridged log-loss Hessian, W the diagonal
    of Bernoulli variances mu*(1-mu); its inverse is the coefficient uncertainty
    the P(lead) pool is drawn from. The IRLS loop is a deliberate copy of beta_fit
    (kept separate so beta_fit stays byte-frozen), and the seam test asserts the
    returned w equals beta_fit exactly for identical inputs. Returns (w, cov) with
    cov symmetrised."""
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
    eta = np.clip(X @ w, -30, 30)
    mu = 1.0 / (1.0 + np.exp(-eta))
    s = np.clip(mu * (1 - mu), 1e-6, None)
    H = X.T @ (X * s[:, None]) + R
    try:
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(H)
    cov = 0.5 * (cov + cov.T)
    return w, cov


def plead_pool(p, f, w, cov, n_draws=4000, seed=0, jitter=1e-9):
    """Joint calibrated-P(lead) draw for one snapshot, the outcome-variance channel
    the sizer's CVaR leg consumes (the MC analogue of forward_edge.per_draw_pwin).
    One coefficient vector per draw from N(w, cov), the Beta map applied across the
    WHOLE snapshot per draw, renormalised within the snapshot per draw, so
    cross-candidate exclusivity and coupling are preserved (a row is one joint
    draw). Returns an [n_draws, n_cand] array; column i is candidate i's [n_draws]
    pool. The column mean approximates the point calibrated P(lead); renormalisation
    is nonlinear so it is close not exact, and the seam test reports the max abs
    diff."""
    p = np.asarray(p, float)
    f = np.asarray(f, float)
    rng = np.random.default_rng(seed)
    cov_j = np.asarray(cov, float) + jitter * np.eye(np.asarray(cov, float).shape[0])
    coeff = rng.multivariate_normal(np.asarray(w, float), cov_j, size=int(n_draws),
                                    method="cholesky")
    D = _design(p, f)
    eta = np.clip(D @ coeff.T, -30, 30)
    pc = 1.0 / (1.0 + np.exp(-eta))
    col = pc.sum(axis=0, keepdims=True)
    pc = np.where(col > 0, pc / col, pc)
    return pc.T


# --------------------------------------------------------------------------- #
# Locked walk-forward driver
# --------------------------------------------------------------------------- #

def calibrate_plead_walkforward(rows, min_prior=3):
    """Locked P(lead) calibration. Fit the stage-conditioned Beta map walk-forward
    (on seasons strictly before the eval season, needing >= min_prior prior
    seasons), apply it to each snapshot's P(lead) and renormalise within the
    snapshot so the book stays exclusive. Seasons without enough history pass
    through unchanged. Rows need season, snap, frac, p_lead, y_lead; returns new
    row dicts with p_lead replaced by the calibrated value (all other keys copied
    through)."""
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
        w = beta_fit([r["p_lead"] for r in prior],
                     [r["y_lead"] for r in prior],
                     [r["frac"] for r in prior])
        grp = defaultdict(list)
        for r in by_season[s]:
            grp[(r["season"], r["snap"])].append(r)
        for _, rws in grp.items():
            pc = beta_apply([r["p_lead"] for r in rws], [r["frac"] for r in rws], w)
            pc = renorm_snapshot(pc)
            for r, pv in zip(rws, pc):
                out.append({**r, "p_lead": float(pv)})
    return out


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
