#!/usr/bin/env python3
"""Anchored idempotent patch: two-part STL/BLK MC branches (base, no overlay).

Adds the two-part generative path for steals and blocks, gated per book by a
TWO_PART flag defaulting OFF. With the flag off the MC is byte-identical to the
current direct single-leg count; with it on, a book uses
    steals  = NB(deflection volume) * Beta(steals per deflection)
    blocks  = NB(rim-FGA volume)    * Beta(blocks per rim-FGA)
exactly mirroring the AST create-times-convert branch, and FALLS BACK to the direct
count for any row whose deflection / rim-FGA leg is absent (hustle pre-2016, tracking
pre-2014, or an unmatched row). _rem_stl / _rem_blk become dispatchers: try two-part
when the flag is on, else (or on a None fallback) the unchanged direct code.

Also extends the _assemble own-history loop (hardcoded node tuple, not VOLUME_NODES)
so prior_rate_defl / prior_rate_rim_fga are built for the two-part volume legs; with
the flag off these are computed-but-unused, so still byte-identical.

NO overlay here by design: this is the two-part BASE, to be proven on the P(lead)
scorecard against the direct model (flip a book on, re-score) before any mu overlay
is wired. Requires the data-layer patch (defl/rim_fga merge, volume nodes, conversion
Betas) already applied.

Run from repo root:
  uv run python3 patch_twopart_branch.py            # dry run
  uv run python3 patch_twopart_branch.py --apply     # write (.bak.twopartbranch backup)
Idempotent, aborts on drift, no-ops on re-run.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

MC = "scripts/modelling/stat_leader/mc.py"

EDITS = [
    # 1. TWO_PART flag beside the other engine toggles
    (MC,
     '# Hierarchical availability recentre (avail_hier.py). Off => identical engine.\n'
     'AVAIL_HIER = False\n'
     '_AVAIL_PRIOR = None',
     '# Hierarchical availability recentre (avail_hier.py). Off => identical engine.\n'
     'AVAIL_HIER = False\n'
     '_AVAIL_PRIOR = None\n'
     '# Per-book two-part volume x conversion for STL/BLK (deflections -> steals,\n'
     '# rim-FGA -> blocks). Read from env so it reaches ProcessPool scorecard workers.\n'
     '# Unset/0 => the direct single-leg count, byte-identical engine. 1 => two-part\n'
     '# where the leg has data (2016+ hustle / 2014+ tracking), else direct as fallback.\n'
     'TWO_PART = {"stl": os.environ.get("TWO_PART_STL", "0") == "1",\n'
     '            "blk": os.environ.get("TWO_PART_BLK", "0") == "1"}'),
    # 2. _rem_stl -> two-part branch + dispatcher
    (MC,
     'def _rem_stl(rng, d, cohort, vpriors, npriors, rem_min, k):\n'
     '    vc, vm = V._vol_count(d, "stl")\n'
     '    if vc is None:\n'
     '        return np.zeros(k)\n'
     '    rate = _gamma_rate(rng, vpriors, "stl", cohort, vc, vm, k,\n'
     '                       own_rate=d.get("prior_rate_stl"), own_min=d.get("prior_min") or 0.0)\n'
     '    return V._draw_count(rng, rate, rem_min, vpriors["fano"].get("stl", 1.0))',
     'def _rem_stl_twopart(rng, d, cohort, vpriors, npriors, rem_min, k):\n'
     '    """Two-part steals: NB deflection volume x steals-per-deflection Beta. Returns\n'
     '    None when the deflection leg is absent so the caller falls back to direct."""\n'
     '    vc, vm = V._vol_count(d, "defl")\n'
     '    if vc is None:\n'
     '        return None\n'
     '    rate = _gamma_rate(rng, vpriors, "defl", cohort, vc, vm, k,\n'
     '                       own_rate=d.get("prior_rate_defl"), own_min=d.get("prior_min") or 0.0)\n'
     '    defl_ct = V._draw_count(rng, rate, rem_min, vpriors["fano"].get("defl", 1.0))\n'
     '    conv = _beta_draw(rng, npriors, "stl_conv", cohort, d, "stl", "defl", k)\n'
     '    return defl_ct * conv\n'
     '\n'
     '\n'
     'def _rem_stl(rng, d, cohort, vpriors, npriors, rem_min, k):\n'
     '    if TWO_PART.get("stl"):\n'
     '        r = _rem_stl_twopart(rng, d, cohort, vpriors, npriors, rem_min, k)\n'
     '        if r is not None:\n'
     '            return r\n'
     '    vc, vm = V._vol_count(d, "stl")\n'
     '    if vc is None:\n'
     '        return np.zeros(k)\n'
     '    rate = _gamma_rate(rng, vpriors, "stl", cohort, vc, vm, k,\n'
     '                       own_rate=d.get("prior_rate_stl"), own_min=d.get("prior_min") or 0.0)\n'
     '    return V._draw_count(rng, rate, rem_min, vpriors["fano"].get("stl", 1.0))'),
    # 3. _rem_blk -> two-part branch + dispatcher
    (MC,
     'def _rem_blk(rng, d, cohort, vpriors, npriors, rem_min, k):\n'
     '    vc, vm = V._vol_count(d, "blk")\n'
     '    if vc is None:\n'
     '        return np.zeros(k)\n'
     '    rate = _gamma_rate(rng, vpriors, "blk", cohort, vc, vm, k,\n'
     '                       own_rate=d.get("prior_rate_blk"), own_min=d.get("prior_min") or 0.0)\n'
     '    return V._draw_count(rng, rate, rem_min, vpriors["fano"].get("blk", 1.0))',
     'def _rem_blk_twopart(rng, d, cohort, vpriors, npriors, rem_min, k):\n'
     '    """Two-part blocks: NB rim-FGA volume x blocks-per-rim-FGA Beta. Returns None\n'
     '    when the rim-FGA leg is absent so the caller falls back to direct."""\n'
     '    vc, vm = V._vol_count(d, "rim_fga")\n'
     '    if vc is None:\n'
     '        return None\n'
     '    rate = _gamma_rate(rng, vpriors, "rim_fga", cohort, vc, vm, k,\n'
     '                       own_rate=d.get("prior_rate_rim_fga"), own_min=d.get("prior_min") or 0.0)\n'
     '    rim_ct = V._draw_count(rng, rate, rem_min, vpriors["fano"].get("rim_fga", 1.0))\n'
     '    conv = _beta_draw(rng, npriors, "blk_conv", cohort, d, "blk", "rim_fga", k)\n'
     '    return rim_ct * conv\n'
     '\n'
     '\n'
     'def _rem_blk(rng, d, cohort, vpriors, npriors, rem_min, k):\n'
     '    if TWO_PART.get("blk"):\n'
     '        r = _rem_blk_twopart(rng, d, cohort, vpriors, npriors, rem_min, k)\n'
     '        if r is not None:\n'
     '            return r\n'
     '    vc, vm = V._vol_count(d, "blk")\n'
     '    if vc is None:\n'
     '        return np.zeros(k)\n'
     '    rate = _gamma_rate(rng, vpriors, "blk", cohort, vc, vm, k,\n'
     '                       own_rate=d.get("prior_rate_blk"), own_min=d.get("prior_min") or 0.0)\n'
     '    return V._draw_count(rng, rate, rem_min, vpriors["fano"].get("blk", 1.0))'),
    # 4. own-history prior for the two-part volume legs
    (MC,
     '        for node in ("reb", "usage", "ast_create", "stl", "blk"):',
     '        for node in ("reb", "usage", "ast_create", "stl", "blk", "defl", "rim_fga"):'),
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
        bak = f"{path}.bak.twopartbranch"
        if not os.path.exists(bak):
            shutil.copy2(path, bak)
        open(path, "w", encoding="utf-8").write(new_text[path])
        print(f"  wrote  {os.path.relpath(path, root)}")
    print(f"\nApplied {len(todo)} edit(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
