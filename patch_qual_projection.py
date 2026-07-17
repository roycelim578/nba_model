#!/usr/bin/env python3
"""Anchored idempotent bugfix: season-level qualifier floor in _eff_matrix.

Companion to patch_qual_fix.py. That patch fixed the LABEL path (realised_eff);
this fixes the PROJECTION path (_eff_matrix), which floored each contender's
simulated eff denominator on ceil(0.70 * rc["ftg"]) where rc["ftg"] is the
player's OWN rostered team-games. A late call-up (e.g. Kobi Simmons, 2023 STL:
gp=1) then gets a tiny q, so his eff = season_total / max(season_games, q)
divides by a tiny denominator, inflating BOTH his projected mean and his spread,
and the fat upper tail of that over-wide distribution steals p_lead mass off the
real contenders (observed: p_lead 0.145 on a one-game player).

Fix: floor on ceil(0.70 * season_tg) where season_tg is the max team-games in
the snapshot context (the season's full schedule length; a full-season
contender is always present in the context, so the max recovers the schedule).
This rescales a call-up's projected mean AND SD down by the same denominator
ratio, collapsing his eff distribution toward zero, which kills the tail win as
well as the mean. Full-season contenders are byte-identical (their ftg already
equals the season max). One-line insert plus the denom swap: no signature
change, so direct _eff_matrix callers are unaffected.

Run from repo root:
  uv run python3 patch_qual_projection.py            # dry run
  uv run python3 patch_qual_projection.py --apply     # write (.bak.qualproj)
Idempotent, aborts on drift, no-ops on re-run.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

EDITS: list[tuple[str, str, str]] = [
    ("scripts/modelling/stat_leader/mc.py",
     '    eff = np.zeros((len(field), k), dtype=float)\n'
     '    for i, pid in enumerate(field):',
     '    eff = np.zeros((len(field), k), dtype=float)\n'
     '    season_tg = max((rc.get("ftg") or 0) for rc in ctx_snap.values()) if ctx_snap else 0\n'
     '    for i, pid in enumerate(field):'),
    ("scripts/modelling/stat_leader/mc.py",
     '        denom = np.maximum(season_games, _qual(rc["ftg"]))',
     '        denom = np.maximum(season_games, _qual(season_tg))'),
]


def _classify(text: str, old: str, new: str) -> str:
    if new in text:
        return "done"
    n = text.count(old)
    if n == 1:
        return "apply"
    return "drift" if n == 0 else "ambiguous"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--root", default=".")
    args = ap.parse_args(argv)
    root = os.path.abspath(args.root)

    cache: dict[str, str] = {}
    plan = []
    bad = False
    for rel, old, new in EDITS:
        path = os.path.join(root, rel)
        if not os.path.exists(path):
            print(f"  MISSING FILE  {rel}"); bad = True; continue
        if path not in cache:
            cache[path] = open(path, encoding="utf-8").read()
        st = _classify(cache[path], old, new)
        tag = {"apply": "APPLY", "done": "skip (done)",
               "drift": "DRIFT (anchor missing)",
               "ambiguous": "DRIFT (not unique)"}[st]
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
        bak = f"{path}.bak.qualproj"
        if not os.path.exists(bak):
            shutil.copy2(path, bak)
        open(path, "w", encoding="utf-8").write(new_text[path])
        print(f"  wrote  {os.path.relpath(path, root)}")
    print(f"\nApplied {len(todo)} edit(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
