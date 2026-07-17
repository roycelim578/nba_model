#!/usr/bin/env python3
"""Anchored idempotent patch: two-part STL/BLK data layer (defl + rim_fga).

Adds the volume and conversion inputs the two-part STL/BLK rebuild needs, WITHOUT
touching the live substrate. Deflections and rim-FGA-faced are already staged
as-of on the same (season, snapshot_date, nba_api_id) key, so nodes._load merges
them into each counts row: defl = defl_std * gp_played_asof (defl_std is per-game,
verified against clean integer totals); rim_fga = def_rim_fga (already cumulative).
Missing source rows (hustle pre-2016, tracking pre-2014, ~1.8% unmatched) leave the
column None, which _vol_count reads as an inactive node so the two-part branch will
fall back to the direct count there.

Then two volume nodes (defl, rim_fga) APPENDED to VOLUME_NODES so fit_priors
auto-coverage-matches their fano, and two conversion Betas (stl_conv = steals per
deflection, blk_conv = blocks per rim-FGA) added to BETA_NODES mirroring ast_conv.

INERT by construction until the mc.py branch reads them:
  - defl/rim_fga are appended LAST in VOLUME_NODES, and _fit_fano advances one shared
    RNG in dict order, so every existing node's fano draws are byte-identical.
  - the Beta prior fit is deterministic (method of moments), so adding stl_conv/blk_conv
    shifts no existing prior.
  - the MC draws named nodes explicitly, never by iterating the dicts, so PRA and the
    direct STL/BLK projections are unchanged. The new priors are fit-but-unused until
    TWO_PART is switched on in the (separate) branch patch.
Requires the merge (edit 1) for the Beta fit to read d["defl"]/d["rim_fga"]; the three
edits are one atomic unit.

Run from repo root:
  uv run python3 patch_twopart_data.py            # dry run
  uv run python3 patch_twopart_data.py --apply     # write (.bak.twopartdata backup)
Idempotent, aborts on drift, no-ops on re-run.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

NODES = "scripts/features/stat_leader/nodes.py"
VOLUME = "scripts/features/stat_leader/volume.py"

EDITS = [
    # 1. merge defl + rim_fga into every counts row inside nodes._load
    (NODES,
     '        counts[(r["season"], r["snapshot_date"], r["nba_api_id"])] = dict(r)\n'
     '    # final-season rate per (season, pid): realised end-of-season, the calibration label\n'
     '    finals = {}',
     '        counts[(r["season"], r["snapshot_date"], r["nba_api_id"])] = dict(r)\n'
     '    for _k in counts:\n'
     '        counts[_k]["defl"] = None\n'
     '        counts[_k]["rim_fga"] = None\n'
     '    for r in conn.execute(f"SELECT season, snapshot_date, nba_api_id, defl_std "\n'
     '                          f"FROM stg_nba_hustle_asof WHERE season IN ({qs})", seasons):\n'
     '        d = counts.get((r["season"], r["snapshot_date"], r["nba_api_id"]))\n'
     '        if d is not None and r["defl_std"] is not None and d.get("gp_played_asof"):\n'
     '            d["defl"] = r["defl_std"] * d["gp_played_asof"]\n'
     '    for r in conn.execute(f"SELECT season, snapshot_date, nba_api_id, def_rim_fga "\n'
     '                          f"FROM stg_nba_player_asof_ext WHERE season IN ({qs})", seasons):\n'
     '        d = counts.get((r["season"], r["snapshot_date"], r["nba_api_id"]))\n'
     '        if d is not None:\n'
     '            d["rim_fga"] = r["def_rim_fga"]\n'
     '    # final-season rate per (season, pid): realised end-of-season, the calibration label\n'
     '    finals = {}'),
    # 2. conversion Betas mirroring ast_conv
    (NODES,
     '    "ast_conv": ("ast", "potential_ast_asof"),\n'
     '}',
     '    "ast_conv": ("ast", "potential_ast_asof"),\n'
     '    "stl_conv": ("stl", "defl"),\n'
     '    "blk_conv": ("blk", "rim_fga"),\n'
     '}'),
    # 3. two volume nodes APPENDED last (preserves the fano RNG order)
    (VOLUME,
     '    "stl": ["stl"],\n'
     '    "blk": ["blk"],\n'
     '}',
     '    "stl": ["stl"],\n'
     '    "blk": ["blk"],\n'
     '    "defl": ["defl"],\n'
     '    "rim_fga": ["rim_fga"],\n'
     '}'),
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
        bak = f"{path}.bak.twopartdata"
        if not os.path.exists(bak):
            shutil.copy2(path, bak)
        open(path, "w", encoding="utf-8").write(new_text[path])
        print(f"  wrote  {os.path.relpath(path, root)}")
    print(f"\nApplied {len(todo)} edit(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
