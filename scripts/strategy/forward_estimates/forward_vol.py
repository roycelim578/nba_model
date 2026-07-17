"""Forward price-volatility predictor: the packaged callable interface.

This wraps the validated assembly from pm_corpus_unified_final into the interface
the portfolio-Kelly sizer consumes. Two steps:

  fit_forward_vol(db_path, out_path)   -- train ALL cross-sectional artefacts on
      the FULL corpus (no LOO; every event) and pickle them. Run offline /
      periodically, not per trade.

  ForwardVolModel(path).forward_move(N, price, frac_elapsed, history)  -- load the
      frozen artefacts, refit ONLY GARCH on the live market's own `history`, and
      return point vol + 50/80/95 interval edges + a low-confidence flag.

Discipline (from the handoff, section 7):
  - GLM, jump params, coverage surface, empirical arrays and the two fitted fade
    weights are CROSS-SECTIONAL: fit once on the whole corpus, frozen. The wrapper
    does NOT refit them live.
  - GARCH is the only thing refit live, on the CURRENT market's own log-odds steps
    only (never prior-year same-award data).
  - Both fades key on frac_elapsed. Effective horizon Heff = min(N, days-to-
    resolution). Conservative over-coverage in the centre-late knife-edge.

British English. Depends on pm_corpus_unified_final for the validated internals.

Run fit:  uv run python -m scripts.strategy.forward_estimates.forward_vol --fit
Then in the sizer:
    from scripts.strategy.forward_estimates.forward_vol import ForwardVolModel
    m = ForwardVolModel("models/forward_vol/forward_vol.pkl")
    r = m.forward_move(N=14, price=0.32, frac_elapsed=0.6, history=own_logodds_steps)
    # r.point_vol, r.edge50, r.edge80, r.edge95, r.low_confidence
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
import sys
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from scripts.common.db import connect
from scripts.strategy.forward_estimates.pm_corpus import (
    _abin, _assemble, _basis, _fband, _fit_glm, _fit_jump, _garch, _glm_vol,
    _jump_vol, _load, _logit, _ptier, _ql, _sigmoid, _w_frac, _EPS,
)

DB_PATH = "data/awards.db"
OUT_PATH = "models/forward_vol/forward_vol.pkl"
BW_FRAC, BW_Z = 0.12, 0.7  # kernel bandwidths in frac-elapsed and SIGNED logit
HMAX = 30
LOWCONF_CELLS = {("deep_hi", "late"), ("deep_hi", "mid"), ("hi", "late"),
                 ("centre", "late")}  # from validation: thin or heavy-tailed


def _tqdm(it, disable, **kw):
    if disable:
        return it
    try:
        from tqdm import tqdm
        return tqdm(it, **kw)
    except ImportError:
        return it


# ------------------------------------------------------------------ fit step

def fit_forward_vol(db_path=DB_PATH, out_path=OUT_PATH, eps=0.005, l2=1.0,
                    hmax=HMAX, no_progress=False, loader=None, lowconf_cells=None,
                    jump=True):
    """Train all cross-sectional artefacts on the FULL corpus and pickle them."""
    conn = connect(db_path); S = (loader or _load)(conn, eps); conn.close()
    if not S:
        raise SystemExit("no series in corpus")
    train = list(S.items())  # FULL corpus, no held-out event

    glm = _fit_glm(train, eps, l2)
    jp = _fit_jump(train, eps)
    if glm is None:
        raise SystemExit("GLM fit failed (insufficient data)")

    def _asm(gv, jv, gav, frac, al, wf, wp):
        return _assemble(gv, jv, gav, frac, al, wf, wp) if jump else \
            _assemble_nojump(gv, gav, frac, wf)

    # fitted fade weights (minimise assembled QLIKE over training points)
    tpts = []
    for mid, s in _tqdm(train, no_progress, desc="fade-fit points", unit="series"):
        n = len(s["prices"])
        for t in range(10, n - 1, 8):
            pt = s["prices"][t]; al = abs(_logit(pt, eps)); frac = t / (n - 1); ttr = n - 1 - t
            vpath = _garch(s["dstep"][:t], hmax)
            for H in (7, 14, 30):
                if t + H >= n:
                    break
                steps = s["dstep"][t:t + H]
                rv = math.sqrt(np.mean(np.square(steps))) if len(steps) else 0.0
                if rv <= 0:
                    continue
                gv = _glm_vol(glm, al, ttr, frac, H, 1.0 if pt > 0.5 else 0.0); jv = _jump_vol(jp, al, H); gav = _garch_vol_local(vpath, H)
                tpts.append((rv, gv, jv, gav, frac, al))

    def obj(x):
        wf_p = x[:2]; wp_p = x[2:]
        return sum(_ql(rv, _asm(gv, jv, gav, frac, al, wf_p, wp_p))
                   for rv, gv, jv, gav, frac, al in tpts) / max(len(tpts), 1)
    best = None
    for x0 in ([0.0, 2.0, 0.0, 1.0], [-1.0, 3.0, -1.0, 2.0]):
        r = minimize(obj, x0, method="Nelder-Mead", options={"maxiter": 1200, "fatol": 1e-5})
        if best is None or r.fun < best.fun:
            best = r
    wf_p = list(best.x[:2]); wp_p = list(best.x[2:])

    # coverage surface (per tier,frac,H empirical ratio quantiles) + pooled
    cov_ratios = {}; pooled = []
    for mid, s in _tqdm(train, no_progress, desc="coverage surface", unit="series"):
        n = len(s["prices"])
        for t in range(10, n - 1, 8):
            pt = s["prices"][t]; al = abs(_logit(pt, eps)); frac = t / (n - 1); ttr = n - 1 - t
            tier = _ptier(pt); fb = _fband(frac); vpath = _garch(s["dstep"][:t], hmax)
            for H in (7, 14, 30):
                remaining = n - 1 - t
                if remaining < 1:
                    break
                Heff = min(H, remaining)
                steps = s["dstep"][t:t + Heff]
                if len(steps) < 1:
                    continue
                rmove = max(abs(s["lo"][k] - s["lo"][t]) for k in range(t, t + Heff + 1))
                gv = _glm_vol(glm, al, ttr, frac, H, 1.0 if pt > 0.5 else 0.0); jv = _jump_vol(jp, al, H); gav = _garch_vol_local(vpath, H)
                fv = _asm(gv, jv, gav, frac, al, wf_p, wp_p)
                ratio = rmove / max(fv * math.sqrt(Heff), _EPS)
                cov_ratios.setdefault((tier, fb, H), []).append(ratio); pooled.append(ratio)
    # store as arrays for pickling
    cov_ratios = {k: np.asarray(v) for k, v in cov_ratios.items()}
    pooled = np.asarray(pooled) if pooled else np.array([1.0])

    # empirical arrays (survivorship-free move over min(H, remaining life)).
    # Store: net endpoint move (for sampling coherent outcomes) AND the max
    # up/down excursions (for the signed quantile reader).
    emp = {H: {"frac": [], "zlogit": [], "up": [], "down": [], "net": []} for H in (7, 14, 30)}
    for mid, s in _tqdm(train, no_progress, desc="empirical arrays", unit="series"):
        lo2 = s["lo"]; prices2 = s["prices"]; n2 = len(prices2)
        for t in range(0, n2 - 1, 3):
            pt = prices2[t]; frac = t / (n2 - 1); zlogit = _logit(pt, eps)
            for H in (7, 14, 30):
                end = min(t + H, n2 - 1)
                # SIGNED move: the largest excursion in each direction over the
                # window, stored as two signed values so up-tail and down-tail are
                # queried independently per price point. Direction asymmetry
                # (favourite down-skew, longshot up-skew) falls out of the data.
                ups = max((lo2[k] - lo2[t]) for k in range(t, end + 1))
                downs = min((lo2[k] - lo2[t]) for k in range(t, end + 1))
                net = lo2[end] - lo2[t]  # coherent endpoint move for sampling
                emp[H]["net"].append(net)
                emp[H]["frac"].append(frac); emp[H]["zlogit"].append(zlogit)
                emp[H]["up"].append(max(ups, 0.0)); emp[H]["down"].append(min(downs, 0.0))
    for H in emp:
        for k in emp[H]:
            emp[H][k] = np.asarray(emp[H][k])

    lc = (LOWCONF_CELLS if lowconf_cells is None
          else _derive_lowconf(cov_ratios, pooled) if lowconf_cells == "auto"
          else {tuple(c) for c in lowconf_cells})
    artefacts = {
        "glm_beta": glm[0], "glm_resid": glm[1], "jump_params": jp,
        "wf_p": wf_p, "wp_p": wp_p, "cov_ratios": cov_ratios, "pooled": pooled,
        "emp": emp, "eps": eps, "bw_frac": BW_FRAC, "bw_z": BW_Z,
        "lowconf_cells": sorted(lc), "n_events": len({s["event"] for _, s in train}),
    }
    if not jump:
        artefacts["jump_disabled"] = True
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(artefacts, f)
    print(f"fitted on {artefacts['n_events']} events; artefacts -> {out_path}")
    return out_path


def _garch_vol_local(vpath, H):
    return math.sqrt(max(vpath[:H].mean(), _EPS)) if vpath is not None else None


def _assemble_nojump(glm_v, garch_v, frac, wf_p):
    """No-jump assembly: GLM fades to GARCH by frac; GLM fallback where GARCH
    is unavailable. The jump term and its price-fade are dropped entirely."""
    if garch_v is None:
        return glm_v
    wfrac = _w_frac(wf_p, frac)
    return (1 - wfrac) * glm_v + wfrac * garch_v


def _derive_lowconf(cov_ratios, pooled, thin_n=60, heavy_mult=1.35):
    """Stat-derived low-confidence cells from the coverage surface. A (tier,
    fb) cell is flagged if pooled across horizons it is thin (few samples) OR
    heavy-tailed (its 95th-pct coverage ratio exceeds the pooled 95th by
    heavy_mult, so the model under-covers there). Returns a set of (tier,fb)."""
    agg = {}
    for (tier, fb, _H), arr in cov_ratios.items():
        agg.setdefault((tier, fb), []).extend(list(arr))
    p95 = float(np.percentile(pooled, 95)) if len(pooled) else 1.0
    out = set()
    for cell, vals in agg.items():
        v = np.asarray(vals)
        thin = len(v) < thin_n
        heavy = len(v) >= 10 and float(np.percentile(v, 95)) > heavy_mult * p95
        if thin or heavy:
            out.add(cell)
    return out


# ------------------------------------------------------------- predict step

@dataclass
class ForwardVolResult:
    point_vol: float        # per-step forecast vol (log-odds), symmetric scale summary
    alpha: float            # the tail probability queried (e.g. 0.10 -> 10th/90th)
    down_move: float        # signed alpha-quantile of the DOWN move (<= 0, log-odds)
    up_move: float          # signed (1-alpha)-quantile of the UP move (>= 0, log-odds)
    eff_n_down: float       # kernel effective sample behind the down quantile
    eff_n_up: float         # kernel effective sample behind the up quantile
    low_confidence: bool    # regime flagged low-conf OR thin local sample either side
    tier: str
    frac_band: str

    def price_band(self, current_price, eps=0.005):
        """Convenience: map the signed log-odds move quantiles back to PRICE."""
        z = math.log(min(max(current_price, eps), 1 - eps) / (1 - min(max(current_price, eps), 1 - eps)))
        def inv(zz):
            return 1.0 / (1.0 + math.exp(-zz))
        return inv(z + self.down_move), inv(z + self.up_move)


class ForwardVolModel:
    """Loads frozen artefacts; refits only GARCH live on the market's own history.

    Primary output is an ALPHA-QUERYABLE SIGNED move-quantile: for a given tail
    probability alpha, the down-move (alpha quantile of signed moves, <=0) and the
    up-move ((1-alpha) quantile, >=0), from the survivorship-free kernel-weighted
    empirical distribution keyed on (frac_elapsed, SIGNED logit). The up/down
    magnitudes differ per price point by construction (favourite down-skew,
    longshot up-skew fall out of the data), so the sizer/execution get the
    upside/downside asymmetry directly. point_vol is retained as a symmetric scale
    summary but the signed quantiles are what respect the asymmetry.
    """

    def __init__(self, path=OUT_PATH):
        with open(path, "rb") as f:
            self.a = pickle.load(f)

    def _fv(self, gv, jv, gav, frac, al):
        """Assembled point vol honouring the pkl's jump_disabled flag; absent
        flag (voter pkl) -> original jump assembly, so behaviour is unchanged."""
        a = self.a
        if a.get("jump_disabled", False):
            return _assemble_nojump(gv, gav, frac, a["wf_p"])
        return _assemble(gv, jv, gav, frac, al, a["wf_p"], a["wp_p"])

    # --- coverage-surface edge with thin-cell shrinkage (kept for point_vol scale)
    def _edge(self, tier, fb, H, cover):
        a = self.a
        arr = a["cov_ratios"].get((tier, fb, H))
        pooled_q = float(np.percentile(a["pooled"], 100 * cover))
        nc = 0 if arr is None else len(arr)
        if nc < 10:
            return pooled_q
        cell_q = float(np.percentile(arr, 100 * cover))
        FULL = 80
        wcell = min(1.0, max(0.0, (nc - 10) / (FULL - 10)))
        return wcell * cell_q + (1 - wcell) * pooled_q

    def _signed_quantile(self, frac, zlogit, H, side, q):
        """Kernel-weighted quantile of signed moves on one side.
        side='down' -> the `up`/`down` array is the down excursions (<=0), return
        the q-quantile (small q = deep down tail). side='up' -> up excursions,
        return the q-quantile (large q = deep up tail). Returns (value, eff_n)."""
        a = self.a; d = a["emp"].get(H)
        if d is None:
            return None, 0.0
        arr = d["down"] if side == "down" else d["up"]
        w = np.exp(-0.5 * (((d["frac"] - frac) / a["bw_frac"]) ** 2 +
                           ((d["zlogit"] - zlogit) / a["bw_z"]) ** 2))
        eff_n = float((w.sum() ** 2) / max((w ** 2).sum(), _EPS))  # Kish effective N
        order = np.argsort(arr); vals = arr[order]; ws = w[order]
        cw = np.cumsum(ws) / max(ws.sum(), _EPS)
        idx = min(max(int(np.searchsorted(cw, q)), 0), len(vals) - 1)
        return float(vals[idx]), eff_n

    def sample_moves(self, N, price, frac_elapsed, history, n_draws=2000,
                     days_to_resolution=None, seed=None):
        """Draw n_draws coherent signed N-day price moves (log-odds) from the
        kernel-weighted empirical distribution at this (frac, signed-price) point.

        This is the sampler the sizer/execution Monte Carlo needs: draw a move here,
        draw a fair-value pwin from the PL cloud, compute edge, repeat -> the
        forward-edge distribution, off which mean (expected edge) and downside
        quantiles (two-sided, asymmetric risk) are read directly. Draws inherit the
        real shape (skew, kurtosis, frequent-small-plus-occasional-lurch) from the
        data -- no parametric assumption. Returns an array of signed log-odds moves;
        add to logit(price) and invert for price outcomes (see moves_to_prices).

        Thin-sample guard: where the local effective sample is small, a fraction of
        draws is replaced by model-scaled Gaussian draws (using point_vol) so the
        sampler never returns a discrete handful of repeated historical values.
        """
        a = self.a; eps = a["eps"]; rng = np.random.default_rng(seed)
        z = _logit(price, eps); al = abs(z)
        Hgrid = min((7, 14, 30), key=lambda h: abs(h - N))
        Heff = min(N, days_to_resolution) if days_to_resolution is not None else N
        d = a["emp"].get(Hgrid)

        # point vol for the thin-sample fallback scale
        ttr = days_to_resolution if days_to_resolution is not None else N
        gv = _glm_vol((a["glm_beta"], a["glm_resid"]), al, ttr, frac_elapsed, N, 1.0 if price > 0.5 else 0.0)
        jv = _jump_vol(a["jump_params"], al, N)
        vpath = _garch(np.asarray(list(history), dtype=float), HMAX)
        gav = _garch_vol_local(vpath, N)
        fv = self._fv(gv, jv, gav, frac_elapsed, al)
        model_sd = fv * math.sqrt(max(Heff, 1))

        if d is None or len(d["net"]) < 5:
            return rng.normal(0.0, model_sd, n_draws)  # no data -> model-scaled

        net = d["net"]
        w = np.exp(-0.5 * (((d["frac"] - frac_elapsed) / a["bw_frac"]) ** 2 +
                           ((d["zlogit"] - z) / a["bw_z"]) ** 2))
        wsum = w.sum()
        eff_n = float((wsum ** 2) / max((w ** 2).sum(), _EPS))  # Kish effective N
        # fraction of draws to take from the empirical mixture vs model fallback
        MIN_EFF = 15.0
        frac_emp = min(1.0, eff_n / MIN_EFF)
        n_emp = int(round(frac_emp * n_draws)); n_mod = n_draws - n_emp

        draws = np.empty(n_draws)
        if n_emp > 0:
            p = w / wsum
            idx = rng.choice(len(net), size=n_emp, p=p)
            # small kernel jitter so draws are continuous, not a discrete set of
            # repeated historical values; jitter scale = a fraction of local spread
            local_sd = math.sqrt(max(np.average((net - np.average(net, weights=w)) ** 2, weights=w), _EPS))
            jitter = rng.normal(0.0, 0.15 * local_sd, n_emp)
            draws[:n_emp] = net[idx] + jitter
        if n_mod > 0:
            draws[n_emp:] = rng.normal(0.0, model_sd, n_mod)
        rng.shuffle(draws)
        return draws

    @staticmethod
    def moves_to_prices(moves, price, eps=0.005):
        """Map signed log-odds moves to price outcomes for a given entry price."""
        z = math.log(min(max(price, eps), 1 - eps) / (1 - min(max(price, eps), 1 - eps)))
        return 1.0 / (1.0 + np.exp(-(z + np.asarray(moves))))

    def forward_move(self, N, price, frac_elapsed, history, alpha=0.10,
                     days_to_resolution=None):
        """Signed down/up move quantiles at tail prob alpha, plus point vol.

        alpha    tail probability: returns the alpha-quantile DOWN move and the
                 (1-alpha)-quantile UP move (e.g. alpha=0.10 -> 10th & 90th pct).
                 Query any alpha per call; execution can ask for 0.01, sizing 0.20.
        Others as before. Signed moves are in log-odds; use result.price_band(price)
        to map to price space.
        """
        a = self.a; eps = a["eps"]
        z = _logit(price, eps); al = abs(z)
        ttr = days_to_resolution if days_to_resolution is not None else N
        tier = _ptier(price); fb = _fband(frac_elapsed)
        Heff = min(N, days_to_resolution) if days_to_resolution is not None else N
        Hgrid = min((7, 14, 30), key=lambda h: abs(h - N))

        # point vol (symmetric scale summary): the validated assembly
        gv = _glm_vol((a["glm_beta"], a["glm_resid"]), al, ttr, frac_elapsed, N, 1.0 if price > 0.5 else 0.0)
        jv = _jump_vol(a["jump_params"], al, N)
        hist = np.asarray(list(history), dtype=float)
        vpath = _garch(hist, HMAX)
        gav = _garch_vol_local(vpath, N)
        fv = self._fv(gv, jv, gav, frac_elapsed, al)

        # signed alpha-quantiles from the empirical distribution at this price
        dn, en_d = self._signed_quantile(frac_elapsed, z, Hgrid, "down", alpha)
        up, en_u = self._signed_quantile(frac_elapsed, z, Hgrid, "up", 1 - alpha)

        # thin-sample guard: if either side's local effective sample is small, widen
        # conservatively toward the model scale (never return a precise-looking
        # quantile off a handful of neighbours). Also flags low_confidence.
        MIN_EFF = 15.0
        model_edge = self._edge(tier, fb, Hgrid, 1 - alpha) * fv * math.sqrt(max(Heff, 1))
        thin = False
        if dn is None or en_d < MIN_EFF:
            dn = -abs(model_edge); thin = True
        else:
            dn = min(dn, -abs(model_edge) * 0.0)  # keep empirical; sign already <=0
        if up is None or en_u < MIN_EFF:
            up = abs(model_edge); thin = True
        # conservative floor everywhere: interval at least the symmetric model edge
        dn = min(dn, -abs(model_edge) if thin else dn)
        up = max(up, abs(model_edge) if thin else up)

        low_conf = thin or ((tier, fb) in set(tuple(c) for c in a["lowconf_cells"]))
        return ForwardVolResult(point_vol=fv, alpha=alpha, down_move=dn, up_move=up,
                                eff_n_down=en_d, eff_n_up=en_u, low_confidence=low_conf,
                                tier=tier, frac_band=fb)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Forward-vol predictor: fit / smoke-test")
    ap.add_argument("--fit", action="store_true", help="train artefacts on the full corpus")
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("--out", default=OUT_PATH)
    ap.add_argument("--no-progress", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="load artefacts and run one prediction")
    args = ap.parse_args(argv)
    if args.fit:
        fit_forward_vol(args.db, args.out, no_progress=args.no_progress)
    if args.smoke:
        m = ForwardVolModel(args.out)
        rng = np.random.default_rng(0); hist = rng.normal(0, 0.15, 40)
        print("signed alpha-quantile output (alpha=0.10 -> 10th/90th signed moves):")
        for price, frac, lab in [(0.5, 0.85, "centre-late"), (0.9, 0.85, "favourite-late"),
                                  (0.1, 0.85, "longshot-late"), (0.05, 0.3, "longshot-early")]:
            r = m.forward_move(14, price, frac, hist, alpha=0.10)
            pb = r.price_band(price)
            print(f"  {lab:>16}: vol={r.point_vol:.3f} down={r.down_move:+.3f} up={r.up_move:+.3f} "
                  f"-> price[{pb[0]:.3f},{pb[1]:.3f}] effN(d/u)={r.eff_n_down:.0f}/{r.eff_n_up:.0f} "
                  f"lowconf={r.low_confidence}")
    if not args.fit and not args.smoke:
        ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
