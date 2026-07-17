#!/usr/bin/env python3
"""Anchored idempotent patch: make the overlay s_hat width opt-in (default mean-only).

The overlay hurt P(lead) for STL and BLK. The mean shift and the s_hat width are two
separate channels, and the width double-counts uncertainty: the gamma posterior already
carries the banked uncertainty, so adding the regression residual on top over-disperses
the argmax and flattens the leader. This gates the width behind OVERLAY_WIDTH (default
off) so OVERLAY_STL=1 alone becomes a mean-only test, isolating the mean shift (which
carries the drift signal) from the width (the suspected culprit). OVERLAY_WIDTH=1
restores the previous behaviour. Inert unless an OVERLAY_* book flag is on.

Requires patch_overlay.py already applied.

Run from repo root:
  uv run python3 patch_overlay_width.py            # dry run
  uv run python3 patch_overlay_width.py --apply     # write (.bak.overlaywidth backup)
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
     'OVERLAY = {"stl": os.environ.get("OVERLAY_STL", "0") == "1",\n'
     '           "blk": os.environ.get("OVERLAY_BLK", "0") == "1",\n'
     '           "ast": os.environ.get("OVERLAY_AST", "0") == "1"}\n'
     '_OVERLAY_ART = {}',
     'OVERLAY = {"stl": os.environ.get("OVERLAY_STL", "0") == "1",\n'
     '           "blk": os.environ.get("OVERLAY_BLK", "0") == "1",\n'
     '           "ast": os.environ.get("OVERLAY_AST", "0") == "1"}\n'
     '# s_hat width is opt-in: the gamma posterior already carries the banked uncertainty,\n'
     '# so adding the regression residual double-counts and over-disperses the argmax.\n'
     '# Default mean-only; OVERLAY_WIDTH=1 restores the width term.\n'
     'OVERLAY_WIDTH = os.environ.get("OVERLAY_WIDTH", "0") == "1"\n'
     '_OVERLAY_ART = {}'),
    (MC,
     '    scaled = rate * (v_star / v_banked)\n'
     '    if s_hat and s_hat > 0:\n'
     '        scaled = scaled + rng.normal(0.0, s_hat / mpg, size=k)\n'
     '    return np.maximum(scaled, 0.0)',
     '    scaled = rate * (v_star / v_banked)\n'
     '    if OVERLAY_WIDTH and s_hat and s_hat > 0:\n'
     '        scaled = scaled + rng.normal(0.0, s_hat / mpg, size=k)\n'
     '    return np.maximum(scaled, 0.0)'),
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
        bak = f"{path}.bak.overlaywidth"
        if not os.path.exists(bak):
            shutil.copy2(path, bak)
        open(path, "w", encoding="utf-8").write(new_text[path])
        print(f"  wrote  {os.path.relpath(path, root)}")
    print(f"\nApplied {len(todo)} edit(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
