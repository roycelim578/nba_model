#!/usr/bin/env python3
"""Anchored idempotent bugfix: season-level qualifier floor in realised_eff.

Bug: realised_eff floored the per-game rate on ceil(0.70 * f) where f is the
player's OWN rostered team-games. A late two-way call-up (e.g. Jacob Gilyard,
2022 STL: 1 game, 3 steals, f=1) gets floor ceil(0.70*1)=1, so eff=3.0 wins the
"title" outright. The floor must be the season's full schedule length, not the
player's rostered games, so a sub-season sample cannot lead a per-game category.

Fix: floor on ceil(0.70 * season_team_games) where season_team_games is the
league-max team-games that season (the full schedule; handles shortened seasons).
Full-season contenders (f == season max) are byte-identical to before; only
sub-qualifier flukes change (they get correctly buried). realised_eff drives the
scorecard labels, the panel labels and stat_true_leader settlement, so one edit
corrects all three.

Run from repo root:
  uv run python3 patch_qual_fix.py            # dry run
  uv run python3 patch_qual_fix.py --apply     # write (.bak.qualfix backup)
Idempotent, aborts on drift, no-ops on re-run.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

EDITS: list[tuple[str, str, str]] = [
    ("scripts/modelling/stat_leader/mc.py",
     '    out = {}\n    for (s, pid), d in finals.items():',
     '    out = {}\n'
     '    season_tg = max((v for v in ftg.values() if v), default=0)\n'
     '    for (s, pid), d in finals.items():'),
    ("scripts/modelling/stat_leader/mc.py",
     '        out[pid] = BANKED[stat](d) / max(gp, _qual(f))',
     '        out[pid] = BANKED[stat](d) / max(gp, _qual(season_tg))'),
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
        bak = f"{path}.bak.qualfix"
        if not os.path.exists(bak):
            shutil.copy2(path, bak)
        open(path, "w", encoding="utf-8").write(new_text[path])
        print(f"  wrote  {os.path.relpath(path, root)}")
    print(f"\nApplied {len(todo)} edit(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
