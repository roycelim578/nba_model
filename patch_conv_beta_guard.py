#!/usr/bin/env python3
"""Anchored idempotent fix: guard beta_posterior against b <= 0.

beta_posterior returns b = b0 + (at_eff - mk_eff), which assumes banked makes never
exceed banked attempts. True for every original Beta node by construction (fg3m<=fg3a,
ftm<=fta, ast<=potential_ast), but NOT for the new cross-source conversion nodes:
stl_conv = steals(box) / deflections(tracking) and blk_conv = blocks(box) /
rim_fga(tracking). Where the tracking leg undercounts, banked makes exceed banked
attempts, at_eff - mk_eff goes below -b0, and b <= 0 crashes rng.beta (and would
zero the conversion in mc._beta_draw's pb>0 guard, silently under-projecting).

Floor the effective (attempts - makes) at zero, capping the implied conversion at
1.0. No-op for the original nodes (at >= mk there), so PRA and the direct STL/BLK
legs stay byte-identical; only the two new nodes are affected.

Run from repo root:
  uv run python3 patch_conv_beta_guard.py            # dry run
  uv run python3 patch_conv_beta_guard.py --apply     # write (.bak.convguard backup)
Idempotent, aborts on drift, no-ops on re-run.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

REL = "scripts/features/stat_leader/nodes.py"

EDITS = [
    (REL,
     '    return a0 + mk_eff, b0 + (at_eff - mk_eff)',
     '    return a0 + mk_eff, b0 + max(at_eff - mk_eff, 0.0)'),
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
        bak = f"{path}.bak.convguard"
        if not os.path.exists(bak):
            shutil.copy2(path, bak)
        open(path, "w", encoding="utf-8").write(new_text[path])
        print(f"  wrote  {os.path.relpath(path, root)}")
    print(f"\nApplied {len(todo)} edit(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
