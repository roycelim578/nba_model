#!/usr/bin/env python3
"""Anchored idempotent patch: per-book Beta calibration flag + raw CVaR pool.

stat_samples.build_stat_samples currently applies the walk-forward Beta calibration
to every book and draws the CVaR pool from the Beta coefficient covariance. The
scorecard shows Beta helps only REB and AST; it is net-negative for PTS and
demonstrated net-negative for STL and BLK. This patch prices REB and AST through
Beta as before (byte-identical) and prices the other three RAW: vote_share becomes
the renormalised raw P(lead), and the pool becomes a raw-centred Dirichlet so the
CVaR channel stays consistent with the raw point (pool mean ~ point).

RAW_POOL_KEFF is the raw pool's concentration (width) knob; it is the one CVaR-sizing
parameter to calibrate against the Beta-pool dispersion on the dev season before live.
Empty 2024 STL/BLK books and dev PTS make it inert for the current catch-up.

Run from repo root:
  uv run python3 patch_beta_flag.py            # dry run
  uv run python3 patch_beta_flag.py --apply     # write (.bak.betaflag backup)
Idempotent, aborts on drift, no-ops on re-run.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

REL = "scripts/backtest/stat_leader/stat_samples.py"

EDITS = [
    (REL,
     'MIN_FIT_SEASONS = 3   # matches calibrate_plead_walkforward(min_prior=3)',
     'MIN_FIT_SEASONS = 3   # matches calibrate_plead_walkforward(min_prior=3)\n'
     '\n'
     '# Books priced through the walk-forward Beta calibration; the rest ship RAW\n'
     '# renormalised P(lead). Scorecard: Beta helps REB (+0.152) and AST (+0.087) on\n'
     '# BSS-vs-leaderboard, is net-negative for PTS (+0.062 -> +0.049), and net-negative\n'
     '# for STL and BLK. So only REB and AST calibrate; PTS/STL/BLK ship raw.\n'
     'BETA_BOOKS = {"reb", "ast"}\n'
     '# Concentration of the raw (Beta-off) CVaR pool: Dirichlet(RAW_POOL_KEFF * p),\n'
     '# centred at the raw point with higher = tighter. The width knob for those books;\n'
     '# calibrate against the Beta-pool dispersion on the dev season before live sizing.\n'
     'RAW_POOL_KEFF = 40.0'),
    (REL,
     'def _pool_seed(eval_season, book, snap):\n'
     '    return zlib.crc32(f"pool|{eval_season}|{book}|{snap}".encode()) & 0xFFFFFFFF',
     'def _pool_seed(eval_season, book, snap):\n'
     '    return zlib.crc32(f"pool|{eval_season}|{book}|{snap}".encode()) & 0xFFFFFFFF\n'
     '\n'
     '\n'
     'def _raw_pool(p, n_draws, k_eff, seed):\n'
     '    """Raw-centred joint P(lead) pool for Beta-off books: Dirichlet(k_eff * p)\n'
     '    draws, [n_draws, n_cand], centred at p. The no-calibration analogue of\n'
     '    calib.plead_pool; k_eff is the CVaR-width knob (higher = tighter)."""\n'
     '    rng = np.random.default_rng(seed)\n'
     '    a = np.maximum(np.asarray(p, float) * k_eff, 1e-6)\n'
     '    return rng.dirichlet(a, size=n_draws)'),
    (REL,
     '    all_rows = fit_rows + eval_rows\n'
     '    cal = CB.calibrate_plead_walkforward(all_rows, min_prior=MIN_FIT_SEASONS)\n'
     '    cal_by = {(r["snap"], r["pid"]): r["p_lead"] for r in cal if r["season"] == eval_season}\n'
     '\n'
     '    prior = [r for r in all_rows if r["season"] < eval_season]\n'
     '    w, cov = CB.beta_fit_cov([r["p_lead"] for r in prior],\n'
     '                             [r["y_lead"] for r in prior],\n'
     '                             [r["frac"] for r in prior])\n'
     '\n'
     '    out = {}\n'
     '    for snap, m in meta.items():\n'
     '        field = m["field"]\n'
     '        vsp = np.array([cal_by[(snap, pid)] for pid in field], dtype=float)\n'
     '        fvec = np.full(len(field), m["frac"], dtype=float)\n'
     '        pool = CB.plead_pool(m["p_raw"], fvec, w, cov, n_draws=n_draws,\n'
     '                             seed=_pool_seed(eval_season, book, snap))\n'
     '        out[snap] = StatSamples(\n'
     '            date=snap, frac=m["frac"], player_ids=[int(p) for p in field],\n'
     '            vote_share_pred=vsp, sizing_weights=vsp.copy(), pwin_pool=pool,\n'
     '            p_lead_raw=m["p_raw"], listed_ids=[int(p) for p in m["listed"]])',
     '    all_rows = fit_rows + eval_rows\n'
     '    beta_on = book in BETA_BOOKS\n'
     '    cal_by, w, cov = {}, None, None\n'
     '    if beta_on:\n'
     '        cal = CB.calibrate_plead_walkforward(all_rows, min_prior=MIN_FIT_SEASONS)\n'
     '        cal_by = {(r["snap"], r["pid"]): r["p_lead"] for r in cal if r["season"] == eval_season}\n'
     '        prior = [r for r in all_rows if r["season"] < eval_season]\n'
     '        w, cov = CB.beta_fit_cov([r["p_lead"] for r in prior],\n'
     '                                 [r["y_lead"] for r in prior],\n'
     '                                 [r["frac"] for r in prior])\n'
     '\n'
     '    out = {}\n'
     '    for snap, m in meta.items():\n'
     '        field = m["field"]\n'
     '        raw = np.asarray(m["p_raw"], float)\n'
     '        if beta_on:\n'
     '            vsp = np.array([cal_by[(snap, pid)] for pid in field], dtype=float)\n'
     '            fvec = np.full(len(field), m["frac"], dtype=float)\n'
     '            pool = CB.plead_pool(raw, fvec, w, cov, n_draws=n_draws,\n'
     '                                 seed=_pool_seed(eval_season, book, snap))\n'
     '        else:\n'
     '            tot = raw.sum()\n'
     '            vsp = raw / tot if tot > 0 else raw\n'
     '            pool = _raw_pool(vsp, n_draws, RAW_POOL_KEFF,\n'
     '                             seed=_pool_seed(eval_season, book, snap))\n'
     '        out[snap] = StatSamples(\n'
     '            date=snap, frac=m["frac"], player_ids=[int(p) for p in field],\n'
     '            vote_share_pred=vsp, sizing_weights=vsp.copy(), pwin_pool=pool,\n'
     '            p_lead_raw=m["p_raw"], listed_ids=[int(p) for p in m["listed"]])'),
]


def _classify(text, old, new):
    if new in text:
        return "done"
    n = text.count(old)
    if n == 1:
        return "apply"
    return "drift" if n == 0 else "ambiguous"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--root", default=".")
    args = ap.parse_args(argv)
    root = os.path.abspath(args.root)
    cache, plan, bad = {}, [], False
    for rel, old, new in EDITS:
        path = os.path.join(root, rel)
        if not os.path.exists(path):
            print(f"  MISSING FILE  {rel}"); bad = True; continue
        if path not in cache:
            cache[path] = open(path, encoding="utf-8").read()
        st = _classify(cache[path], old, new)
        tag = {"apply": "APPLY", "done": "skip (done)",
               "drift": "DRIFT (anchor missing)", "ambiguous": "DRIFT (not unique)"}[st]
        print(f"  {tag:<24} {rel}")
        if st in ("drift", "ambiguous"):
            bad = True
        plan.append((path, old, new, st))
    if bad:
        print("\nABORTED: anchor drift or missing file. Nothing written.")
        return 2
    todo = [p for p in plan if p[3] == "apply"]
    if not todo:
        print("\nAll edits already applied. No-op.")
        return 0
    if not args.apply:
        print(f"\nDry run OK, {len(todo)} edit(s) ready. Re-run with --apply.")
        return 0
    new_text = dict(cache)
    for path, old, new, _ in todo:
        new_text[path] = new_text[path].replace(old, new, 1)
    for path in {p[0] for p in todo}:
        bak = f"{path}.bak.betaflag"
        if not os.path.exists(bak):
            shutil.copy2(path, bak)
        open(path, "w", encoding="utf-8").write(new_text[path])
        print(f"  wrote  {os.path.relpath(path, root)}")
    print(f"\nApplied {len(todo)} edit(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
