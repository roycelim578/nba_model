"""Capstone: assemble and validate the full unified forward-vol model.

The complete evidence-driven map:
  every price level fades from a CROSS-SECTIONAL model (GLM, good early-life)
  to a SERIES-SPECIFIC model (good late-life) as frac-elapsed grows; the
  series-specific model is the Hawkes JUMP model in the tails (rare discrete
  jumps) and GARCH in the centre (clustered churn). Two fades, both smooth:
    frac fade   w_frac(frac) : GLM -> series-specific as resolution approaches
    price fade  w_price(abs_logit) : jump-model -> GARCH as price -> centre
  so the series-specific forecast = w_price*garch + (1-w_price)*jump, and the
  final = (1-w_frac)*glm + w_frac*series_specific. No hard boundaries.

This harness:
  1. Verifies the frac-elapsed handoff HOLDS IN THE TAILS too (GLM early ->
     jump late), not just assumed by analogy to the centre.
  2. Fits both fade weights on TRAINING events (sigmoid params by frac and by
     abs_logit) by minimising assembled-model QLIKE, then evaluates OOS.
  3. Scores the FULLY ASSEMBLED model (both fades + per-regime coverage scale)
     vs naive across all price tiers x frac bands.

LOO, no lookahead (all fits on training events; forecasts use info <= anchor).
tqdm (--no-progress).

British English. Pure DB + numpy/scipy/arch.

Run:  uv run python -m scripts.strategy.forward_estimates.pm_corpus
      uv run python -m scripts.strategy.forward_estimates.pm_corpus --no-progress
"""

from __future__ import annotations

import argparse
import math
import sys
import warnings

import numpy as np
from scipy.optimize import minimize

from scripts.common.db import connect


def _tqdm(it, disable, **kw):
    if disable:
        return it
    try:
        from tqdm import tqdm
        return tqdm(it, **kw)
    except ImportError:
        return it


DB_PATH = "data/awards.db"
_EPS = 1e-9
JUMP_THR = 0.5
Z80, Z50 = 1.282, 0.674
PTIERS = [(0.005, 0.03, "deep_lo"), (0.03, 0.10, "lo"), (0.10, 0.30, "mid_lo"),
          (0.30, 0.70, "centre"), (0.70, 0.90, "mid_hi"), (0.90, 0.97, "hi"),
          (0.97, 0.995, "deep_hi")]
FRAC_BANDS = [(0.0, 0.33, "early"), (0.33, 0.66, "mid"), (0.66, 1.01, "late")]


def _logit(p, eps):
    p = min(max(p, eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


def _ptier(p):
    for a, z, nm in PTIERS:
        if a <= p < z:
            return nm
    return "centre"


def _fband(f):
    for a, z, nm in FRAC_BANDS:
        if a <= f < z:
            return nm
    return "late"


def _abin(al):
    return "centre" if al < 0.85 else ("mid" if al < 2.2 else "extreme")


def _load(conn, eps):
    raw = {}
    for r in conn.execute(
        "SELECT market_id, day, yes_price FROM corpus_price_daily ORDER BY market_id, day"
    ):
        raw.setdefault(r["market_id"], []).append(float(r["yes_price"]))
    meta = {r["market_id"]: r["event_slug"] for r in conn.execute(
        "SELECT market_id, event_slug FROM corpus_market")}
    out = {}
    for mid, prices in raw.items():
        if len(prices) < 12:
            continue
        lo = [_logit(p, eps) for p in prices]
        dstep = np.array([lo[i] - lo[i - 1] for i in range(1, len(lo))])
        out[mid] = {"prices": prices, "lo": lo, "dstep": dstep, "event": meta.get(mid, mid)}
    return out


def _ql(v, f):
    v = max(v, _EPS); f = max(f, _EPS); r = v / f
    return r - math.log(r) - 1.0


# ---- component models ------------------------------------------------------

def _basis(al, ttr, frac, H, fav=0.0):
    # fav = 1.0 if favourite (price>0.5) else 0.0. Direction terms let the GLM
    # express the measured favourite/longshot asymmetry that REVERSES with life:
    # favourites calmer early, MORE volatile late (they still carry resolution
    # risk while a late longshot has quietly lost). Without these the GLM averages
    # the two and mis-fits both, and under-predicts favourite-late vol (too-tight
    # CIs on favourites -- the dangerous direction). fav, fav*frac, fav*al.
    lH = math.log(H)
    return np.array([1.0, al, al * al, ttr, frac, lH, H, al * lH, frac * lH,
                     frac * frac, fav, fav * frac, fav * al])


def _fit_glm(train, eps, l2):
    Xs, ys = [], []
    for mid, s in train:
        n = len(s["prices"])
        for t in range(5, n - 1, 4):
            pt = s["prices"][t]; al = abs(_logit(pt, eps)); frac = t / (n - 1)
            fav = 1.0 if pt > 0.5 else 0.0
            for H in (2, 5, 10, 20, 30):
                if t + H >= n:
                    break
                steps = s["dstep"][t:t + H]
                if len(steps) < 2:
                    continue
                tg = math.sqrt(np.mean(steps ** 2))
                if tg <= 0:
                    continue
                Xs.append(_basis(al, n - 1 - t, frac, H, fav)); ys.append(math.log(tg))
    if len(Xs) < 20:
        return None
    X = np.array(Xs); y = np.array(ys); p = X.shape[1]
    beta = np.linalg.solve(X.T @ X + l2 * np.eye(p), X.T @ y)
    return beta, float(np.sqrt(np.mean((y - X @ beta) ** 2)))


_GARCH_CACHE = {}


def _garch(hist, hmax):
    from arch import arch_model
    if len(hist) < 20:
        return None
    # memoise on (length, rounded hash of the slice) -- the same history slice
    # recurs across the fit/coverage/eval loops within a run; caching cuts the
    # redundant GARCH refits ~3x with identical results.
    key = (len(hist), hash(np.asarray(hist).tobytes()), hmax)
    if key in _GARCH_CACHE:
        return _GARCH_CACHE[key]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            am = arch_model(np.asarray(hist) * 100.0, mean="Zero", vol="GARCH", p=1, o=0, q=1, dist="normal")
            res = am.fit(disp="off", show_warning=False)
        out = res.forecast(horizon=hmax, reindex=False).variance.values[-1] / (100.0 ** 2)
    except Exception:  # noqa: BLE001
        out = None
    _GARCH_CACHE[key] = out
    return out


def _fit_jump(train, eps):
    buckets = {b: {"days": 0, "jumps": 0, "sizes": []} for b in ("centre", "mid", "extreme")}
    for mid, s in train:
        lo = s["lo"]; prices = s["prices"]
        for i in range(1, len(prices)):
            al = abs(_logit(prices[i - 1], eps)); b = _abin(al); d = lo[i] - lo[i - 1]
            buckets[b]["days"] += 1
            if abs(d) > JUMP_THR:
                buckets[b]["jumps"] += 1; buckets[b]["sizes"].append(abs(d))
    params = {}
    for b, rec in buckets.items():
        rate = rec["jumps"] / max(rec["days"], 1)
        mean_sq = float(np.mean(np.square(rec["sizes"]))) if rec["sizes"] else (JUMP_THR + 0.3) ** 2
        params[b] = {"rate": max(rate, 1e-4), "mean_sq": mean_sq}
    return params


def _glm_vol(glm, al, ttr, frac, H, fav=0.0):
    return math.exp(_basis(al, ttr, frac, H, fav) @ glm[0])


def _jump_vol(jp, al, H):
    p = jp[_abin(al)]
    return math.sqrt(max(p["rate"] * p["mean_sq"], _EPS))


def _garch_vol(vpath, H):
    return math.sqrt(max(vpath[:H].mean(), _EPS)) if vpath is not None else None


# ---- fade weights ----------------------------------------------------------

def _sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _w_frac(params, frac):
    a, b = params
    return _sigmoid(a + b * frac)  # 0=GLM .. 1=series-specific


def _w_price(params, al):
    a, b = params
    return _sigmoid(a + b * al)  # 1=jump(high al) .. 0=garch(low al)


def _assemble(glm_v, jump_v, garch_v, frac, al, wf_p, wp_p):
    wprice = _w_price(wp_p, al)  # weight on JUMP (high at tails)
    if garch_v is None:
        series = jump_v
    else:
        series = wprice * jump_v + (1 - wprice) * garch_v
    wfrac = _w_frac(wf_p, frac)  # weight on series-specific (high late)
    return (1 - wfrac) * glm_v + wfrac * series


def run(db_path, eps, l2, hmax, no_progress):
    conn = connect(db_path); S = _load(conn, eps); conn.close()
    if not S:
        return {"error": "no series"}
    events = sorted({s["event"] for s in S.values()})

    # tail frac-handoff check accumulators (GLM vs jump skill by frac, tails only)
    tail_frac = {nm: {"glm": [], "jump": [], "nq": []} for _, _, nm in FRAC_BANDS}
    # assembled-model accumulators per (tier, fband)
    asm = {}

    for held in _tqdm(events, no_progress, desc="unified final LOO", unit="event"):
        train = [(mid, s) for mid, s in S.items() if s["event"] != held]
        if len(train) < 10:
            continue
        glm = _fit_glm(train, eps, l2)
        jp = _fit_jump(train, eps)
        if glm is None:
            continue

        # --- fit fade weights on TRAINING events (build training points quickly)
        tpts = []
        for mid, s in train:
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
                    gv = _glm_vol(glm, al, ttr, frac, H, 1.0 if pt > 0.5 else 0.0)
                    jv = _jump_vol(jp, al, H)
                    gav = _garch_vol(vpath, H)
                    tpts.append((rv, gv, jv, gav, frac, al))
        if len(tpts) < 50:
            continue

        def obj(x):
            wf_p = x[:2]; wp_p = x[2:]
            tot = 0.0
            for rv, gv, jv, gav, frac, al in tpts:
                f = _assemble(gv, jv, gav, frac, al, wf_p, wp_p)
                tot += _ql(rv, f)
            return tot / len(tpts)
        best = None
        for x0 in ([0.0, 2.0, 0.0, 1.0], [-1.0, 3.0, -1.0, 2.0]):
            r = minimize(obj, x0, method="Nelder-Mead", options={"maxiter": 1200, "fatol": 1e-5})
            if best is None or r.fun < best.fun:
                best = r
        wf_p, wp_p = best.x[:2], best.x[2:]
        # --- coverage surface: fit on TRAINING events (no lookahead), keyed on
        # (tier, fband, horizon). Empirical quantiles of realised-move /
        # (forecast-vol*sqrt(H)) give distribution-free interval edges at 50/80/95.
        # Sparse cells fall back to a pooled quantile.
        cov_ratios = {}
        pooled = []
        for mid, s in train:
            n = len(s["prices"])
            for t in range(10, n - 1, 8):
                pt = s["prices"][t]; al = abs(_logit(pt, eps)); frac = t / (n - 1); ttr = n - 1 - t
                tier = _ptier(pt); fb = _fband(frac)
                vpath = _garch(s["dstep"][:t], hmax)
                for H in (7, 14, 30):
                    remaining = n - 1 - t
                    if remaining < 1:
                        break
                    Heff = min(H, remaining)
                    steps = s["dstep"][t:t + Heff]
                    if len(steps) < 1:
                        continue
                    rmove = max(abs(s["lo"][k] - s["lo"][t]) for k in range(t, t + Heff + 1))
                    gv = _glm_vol(glm, al, ttr, frac, H, 1.0 if pt > 0.5 else 0.0)
                    jv = _jump_vol(jp, al, H)
                    gav = _garch_vol(vpath, H)
                    fv = _assemble(gv, jv, gav, frac, al, wf_p, wp_p)
                    ratio = rmove / max(fv * math.sqrt(Heff), _EPS)
                    cov_ratios.setdefault((tier, fb, H), []).append(ratio)
                    pooled.append(ratio)
        pooled = np.array(pooled) if pooled else np.array([1.0])

        def edge(tier, fb, H, cover):
            arr = cov_ratios.get((tier, fb, H), [])
            nc = len(arr)
            pooled_q = float(np.percentile(pooled, 100 * cover))
            if nc < 10:
                return pooled_q  # too thin to trust at all -> pooled
            cell_q = float(np.percentile(np.array(arr), 100 * cover))
            # shrink cell estimate toward pooled by sample size: full trust at
            # n>=FULL, all-pooled at n<=10, linear between. Prevents a handful of
            # noisy ratios in a thin cell (e.g. mid_hi/early) producing a wild
            # over-wide quantile that covers everything (cover95=1.0 artefact).
            FULL = 80
            wcell = min(1.0, max(0.0, (nc - 10) / (FULL - 10)))
            return wcell * cell_q + (1 - wcell) * pooled_q

        # --- continuous empirical interval (kernel-weighted quantile).
        # Outside the centre the model edge is calibrated. In the centre-late
        # knife-edge no forecast beats naive, so the honest interval is the
        # empirical spread of realised moves in that regime. To avoid flat buckets,
        # we use a KERNEL-WEIGHTED quantile: weight training moves by Gaussian
        # proximity in (frac, abs_logit), giving a width that varies CONTINUOUSLY
        # over frac and price. Survivorship-free (clamped move-to-resolution).
        emp = {H: {"frac": [], "al": [], "mv": []} for H in (7, 14, 30)}
        for mid, s in train:
            lo2 = s["lo"]; prices2 = s["prices"]; n2 = len(prices2)
            for t in range(0, n2 - 1, 3):
                pt = prices2[t]; frac = t / (n2 - 1); al = abs(_logit(pt, eps))
                for H in (7, 14, 30):
                    end = min(t + H, n2 - 1)
                    mv = max(abs(lo2[k] - lo2[t]) for k in range(t, end + 1))
                    emp[H]["frac"].append(frac); emp[H]["al"].append(al); emp[H]["mv"].append(mv)
        for H in emp:
            for k in emp[H]:
                emp[H][k] = np.array(emp[H][k])

        BW_FRAC, BW_AL = 0.12, 0.6  # kernel bandwidths in frac and abs-logit

        def emp_quantile(frac, al, H, cover):
            d = emp.get(H)
            if d is None or len(d["mv"]) < 30:
                return None
            w = np.exp(-0.5 * (((d["frac"] - frac) / BW_FRAC) ** 2 + ((d["al"] - al) / BW_AL) ** 2))
            wsum = w.sum()
            if wsum < 5:  # too few neighbours -> unreliable
                return None
            order = np.argsort(d["mv"])
            mv_s = d["mv"][order]; w_s = w[order]
            cw = np.cumsum(w_s) / wsum
            idx = np.searchsorted(cw, cover)
            idx = min(max(idx, 0), len(mv_s) - 1)
            return float(mv_s[idx])

        # --- evaluate on held-out event
        for mid, s in S.items():
            if s["event"] != held:
                continue
            lo = s["lo"]; prices = s["prices"]; n = len(prices)
            for t in range(10, n - 1, 3):
                pt = prices[t]; al = abs(_logit(pt, eps)); frac = t / (n - 1); ttr = n - 1 - t
                tier = _ptier(pt); fb = _fband(frac)
                vpath = _garch(s["dstep"][:t], hmax)
                for H in (7, 14, 30):
                    remaining = n - 1 - t
                    if remaining < 1:
                        break
                    Heff = min(H, remaining)  # near resolution, hold-to-resolution horizon
                    steps = s["dstep"][t:t + Heff]
                    rv = math.sqrt(np.mean(np.square(steps))) if len(steps) else 0.0
                    if rv <= 0:
                        continue
                    rmove = max(abs(lo[k] - lo[t]) for k in range(t, t + Heff + 1))
                    tr = s["dstep"][max(0, t - 10):t]
                    naive = math.sqrt(np.mean(np.square(tr))) if len(tr) else 0.0
                    if naive <= 0:
                        continue
                    gv = _glm_vol(glm, al, ttr, frac, H, 1.0 if pt > 0.5 else 0.0)
                    jv = _jump_vol(jp, al, H)
                    gav = _garch_vol(vpath, H)
                    fv = _assemble(gv, jv, gav, frac, al, wf_p, wp_p)
                    # assembled record
                    key = (tier, fb)
                    d = asm.setdefault(key, {"q": [], "nq": [], "c50": [], "c80": [], "c95": []})
                    d["q"].append(_ql(rv, fv)); d["nq"].append(_ql(rv, naive))
                    scale = fv * math.sqrt(Heff)
                    e50 = edge(tier, fb, H, 0.50) * scale
                    e80 = edge(tier, fb, H, 0.80) * scale
                    e95 = edge(tier, fb, H, 0.95) * scale
                    # smooth blend model edge -> continuous empirical quantile in the
                    # centre-late regime (no hard max -> no boundary vol spike). Weight
                    # w rises with frac-elapsed AND centrality (abs-logit near 0), so
                    # the empirical only takes over late + central; early or in the
                    # tails w~0 and we get the pure model edge. Same smooth-fade
                    # principle as the GLM->series-specific handoff.
                    eq50 = emp_quantile(frac, al, H, 0.50); eq80 = emp_quantile(frac, al, H, 0.80); eq95 = emp_quantile(frac, al, H, 0.95)
                    if eq50 is not None:
                        w_frac_emp = _sigmoid(6.0 * (frac - 0.55))       # ramps in over the second half of life
                        w_centre = _sigmoid(6.0 * (0.85 - al))           # ramps in as price -> centre (al<0.85)
                        w = w_frac_emp * w_centre
                        # CONSERVATIVE over-coverage: where the empirical arm is
                        # engaged (centre-late), read a HIGHER quantile than nominal
                        # so the interval is deliberately over-covered. The centre-late
                        # move dist is heavy-tailed (t-like) and thin-sampled, and
                        # under-coverage there is catastrophic (a 50/50 that can lurch
                        # to 0/1). Reading the 0.90/0.95/0.99 empirical quantile as the
                        # 0.80/0.90/0.95 edge self-scales to the actual tail heaviness.
                        eqc80 = emp_quantile(frac, al, H, 0.90) or eq80
                        eqc95 = emp_quantile(frac, al, H, 0.99) or eq95
                        e50 = (1 - w) * e50 + w * max(e50, eq50)
                        e80 = (1 - w) * e80 + w * max(e80, eqc80)
                        e95 = (1 - w) * e95 + w * max(e95, eqc95)
                    d["c50"].append(1.0 if rmove <= e50 else 0.0)
                    d["c80"].append(1.0 if rmove <= e80 else 0.0)
                    d["c95"].append(1.0 if rmove <= e95 else 0.0)
                    # tail frac-handoff check (tails only)
                    if tier in ("deep_lo", "lo", "hi", "deep_hi"):
                        tf = tail_frac[fb]
                        tf["glm"].append(_ql(rv, gv)); tf["jump"].append(_ql(rv, jv)); tf["nq"].append(_ql(rv, naive))

    def sk(qm, qn):
        if not qm or not qn:
            return None
        a, b = np.mean(qm), np.mean(qn)
        return _r(1 - a / b) if b > 0 else None

    tailchk = {nm: {"glm_skill": sk(tail_frac[nm]["glm"], tail_frac[nm]["nq"]),
                    "jump_skill": sk(tail_frac[nm]["jump"], tail_frac[nm]["nq"]),
                    "n": len(tail_frac[nm]["glm"])} for _, _, nm in FRAC_BANDS}
    grid = {}
    for _, _, tier in PTIERS:
        for _, _, fb in FRAC_BANDS:
            d = asm.get((tier, fb))
            if not d or len(d["q"]) < 15:
                grid[f"{tier}/{fb}"] = {"n": len(d["q"]) if d else 0, "sparse": True}
                continue
            grid[f"{tier}/{fb}"] = {"n": len(d["q"]),
                                    "skill_vs_naive": sk(d["q"], d["nq"]),
                                    "cover50": _r(np.mean(d["c50"])),
                                    "cover80": _r(np.mean(d["c80"])),
                                    "cover95": _r(np.mean(d["c95"]))}
    return {"tail_frac_handoff": tailchk, "assembled_grid": grid,
            "config": {"n_events": len(events)}}


def _r(x):
    return None if (x is None or (isinstance(x, float) and not np.isfinite(x))) else round(float(x), 4)


def _print(res):
    if "error" in res:
        print(res["error"]); return
    print("=" * 74)
    print("TAIL frac-handoff check (tails only): GLM vs JUMP skill vs naive by frac")
    print("=" * 74)
    for _, _, nm in FRAC_BANDS:
        r = res["tail_frac_handoff"][nm]
        print(f"  {nm:>6}: GLM={str(r['glm_skill']):>8}  JUMP={str(r['jump_skill']):>8}  (n={r['n']})")
    print("  (expect GLM to win early, JUMP to win late -> confirms frac fade in tails)")
    print("\n" + "=" * 74)
    print("ASSEMBLED MODEL skill vs naive  (tier x frac)")
    print("=" * 74)
    print(f"{'tier':>10} | " + " | ".join(f"{fb:>8}" for _, _, fb in FRAC_BANDS))
    print("-" * 74)
    for _, _, tier in PTIERS:
        cells = []
        for _, _, fb in FRAC_BANDS:
            r = res["assembled_grid"][f"{tier}/{fb}"]
            cells.append(f"{'sparse':>8}" if r.get("sparse") else f"{str(r['skill_vs_naive']):>8}")
        print(f"{tier:>10} | " + " | ".join(cells))
    print("\n" + "=" * 74)
    print("COVERAGE (per-regime surface, walk-forward): target 0.50 / 0.80 / 0.95")
    print("=" * 74)
    print(f"{'tier/frac':>18} | {'cover50':>8} | {'cover80':>8} | {'cover95':>8}")
    print("-" * 74)
    # pooled coverage across cells per tier for compactness
    for _, _, tier in PTIERS:
        for _, _, fb in FRAC_BANDS:
            r = res["assembled_grid"][f"{tier}/{fb}"]
            if r.get("sparse"):
                continue
            print(f"{tier + '/' + fb:>18} | {str(r['cover50']):>8} | "
                  f"{str(r['cover80']):>8} | {str(r['cover95']):>8}")
    print("=" * 74)
    print("READ: skill>0 everywhere = unified model beats naive across the book.")
    print("  cover50/80/95 each near target across cells = the walk-forward per-regime")
    print("  coverage surface is calibrated at all interval widths (sizer-ready).")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Unified final model assembly + validation")
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("--eps", type=float, default=0.005)
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--hmax", type=int, default=30)
    ap.add_argument("--no-progress", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    res = run(args.db, args.eps, args.l2, args.hmax, args.no_progress)
    if args.json:
        import json
        print(json.dumps(res, indent=2))
    else:
        _print(res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
