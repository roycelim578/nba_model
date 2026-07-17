#!/usr/bin/env python3
"""Anchored idempotent patch: bake STL two-part on by default (ship config).

The two-part STL model beat direct on the P(lead) scorecard (+0.289 vs +0.284
BSS-vs-leaderboard, +0.396 vs +0.369 who-leads), so it is the shipped model. Flip
the TWO_PART_STL default from "0" to "1" so the stat arm uses it without an env flag;
the env override still works (TWO_PART_STL=0 to force direct). BLK stays direct
(two-part lost) and the overlay stays off (dead). Voter arm is untouched.

Run from repo root:
  uv run python3 patch_ship_stl_twopart.py            # dry run
  uv run python3 patch_ship_stl_twopart.py --apply     # write (.bak.shipstl backup)
Idempotent, aborts on drift, no-ops on re-run.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

MC = "scripts/modelling/stat_leader/mc.py"

EDITS = [
    (MC,
     'TWO_PART = {"stl": os.environ.get("TWO_PART_STL", "0") == "1",\n'
     '            "blk": os.environ.get("TWO_PART_BLK", "0") == "1"}',
     'TWO_PART = {"stl": os.environ.get("TWO_PART_STL", "1") == "1",\n'
     '            "blk": os.environ.get("TWO_PART_BLK", "0") == "1"}'),
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
        bak = f"{path}.bak.shipstl"
        if not os.path.exists(bak):
            shutil.copy2(path, bak)
        open(path, "w", encoding="utf-8").write(new_text[path])
        print(f"  wrote  {os.path.relpath(path, root)}")
    print(f"\nApplied {len(todo)} edit(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
