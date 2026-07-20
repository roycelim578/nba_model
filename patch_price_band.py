#!/usr/bin/env python3
"""Anchored idempotent patcher: tradeable price band [PX_MIN, PX_MAX].

notrade_region.py (size_positions_region): a name whose market price sits outside the band
(default 0.05 to 0.95) is not opened into and existing positions are not added to; holds and
trims are unaffected, so a winner that rides above 0.95 is kept, we simply never buy in there.
This fades rather than joins longshot bias (we do not buy sub-5% YES, and we do not pay 95%+
to fade a near-dead name), and it removes the sub-1c price-series artefacts for free. It does
not touch carry, radj, or sizing. Assumes patch_gate_slack.py is applied (the _EDGE_NOISE
anchor was added by it). British English.
"""
from __future__ import annotations
import argparse, difflib, os, shutil, sys

ROOT = os.path.expanduser("~/Desktop/QuantDev_Project/nba_model")
NTR = os.path.join(ROOT, "scripts/strategy/trade_regions/notrade_region.py")

EDITS = [
    (NTR,
     '    _EDGE_NOISE = float(os.environ.get("EDGE_NOISE", "0.02"))\n',
     '    _EDGE_NOISE = float(os.environ.get("EDGE_NOISE", "0.02"))\n'
     '    _PX_MIN = float(os.environ.get("PX_MIN", "0.05"))\n'
     '    _PX_MAX = float(os.environ.get("PX_MAX", "0.95"))\n'),
    (NTR,
     "        pr = float(np.clip(px[i], 1e-4, 1.0 - 1e-4))\n",
     "        pr = float(np.clip(px[i], 1e-4, 1.0 - 1e-4))\n"
     "        if (pr < _PX_MIN or pr > _PX_MAX) and abs(scaled) > abs(cur[i]):\n"
     "            scaled = cur[i]  # outside tradeable price band: hold, never open or add\n"),
]


def run(apply):
    cache = {}
    for path, _, _ in EDITS:
        if path not in cache:
            cache[path] = open(path).read() if os.path.exists(path) else ""
    ok = True
    for path, old, new in EDITS:
        src = cache[path]
        if src == "":
            print(f"[MISS] {path}", file=sys.stderr); ok = False; continue
        if new in src and old not in src:
            print(f"[skip] {os.path.relpath(path, ROOT)}: edit already applied"); continue
        n = src.count(old)
        if n != 1:
            print(f"[ABORT] {os.path.relpath(path, ROOT)}: anchor count {n} != 1 for:\n    {old[:70]!r}", file=sys.stderr)
            ok = False; continue
        cache[path] = src.replace(old, new, 1)
    if not ok:
        print("\nNo files written. If an anchor was missing, confirm patch_gate_slack.py is applied.", file=sys.stderr)
        sys.exit(1)
    for path in cache:
        orig = open(path).read()
        if cache[path] == orig:
            continue
        if apply:
            shutil.copyfile(path, path + ".bak")
            open(path, "w").write(cache[path])
            print(f"[apply] {os.path.relpath(path, ROOT)} (.bak written)")
        else:
            sys.stdout.writelines(difflib.unified_diff(
                orig.splitlines(True), cache[path].splitlines(True),
                fromfile=os.path.relpath(path, ROOT), tofile="(patched)"))
            print()
    print("OK" if apply else "dry-run OK (pass --apply to write)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    run(ap.parse_args().apply)
