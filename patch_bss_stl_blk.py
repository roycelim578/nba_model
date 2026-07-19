"""Extend stat_bss_persist to the five-book stat arm (add STL and BLK).

Anchored, idempotent, transactional. Dry-run by default; --apply writes a single
.bak and the edited file. Validates every anchor before touching disk: if any anchor
is missing or ambiguous the patch aborts and writes nothing. Re-running after a
successful apply is a no-op (every edit reports "already applied").

Four edits, all in scripts/modelling/stat_leader/stat_bss_persist.py:
  1. BETA_BY_STAT gains stl/blk, both False (STL and BLK ship Beta OFF).
  2. The main() stats list gains stl/blk.
  3. A defensive STAT_FLOOR extension: stl/blk default to 2008 (the raw-gate window)
     only if the imported scorecard STAT_FLOOR does not already pin them, so the
     persister does not KeyError regardless of whether the live scorecard was extended.
  4. Docstring correction (the "PRA only" line is now false).

British English, no em dashes.
"""
from __future__ import annotations

import argparse
import shutil
import sys

TARGET = "scripts/modelling/stat_leader/stat_bss_persist.py"

EDITS = [
    # 1. Beta map: STL and BLK score raw.
    (
        'BETA_BY_STAT = {"reb": True, "pts": False, "ast": True}',
        'BETA_BY_STAT = {"reb": True, "pts": False, "ast": True, "stl": False, "blk": False}',
    ),
    # 2. Books scored in main().
    (
        '    stats = ["reb", "pts", "ast"]',
        '    stats = ["reb", "pts", "ast", "stl", "blk"]',
    ),
    # 3. Defensive STAT_FLOOR extension right after the module logger.
    (
        'log = logging.getLogger("stat_leader.bss_persist")',
        'log = logging.getLogger("stat_leader.bss_persist")\n\n'
        'STAT_FLOOR = {**STAT_FLOOR, "stl": STAT_FLOOR.get("stl", 2008), '
        '"blk": STAT_FLOOR.get("blk", 2008)}',
    ),
    # 4. Docstring: no longer PRA only.
    (
        'Beta matches the shipping flag (on for REB and AST, off for PTS). PRA only; STL and\n'
        'BLK need the STL/BLK-capable scorecard. British English, no em dashes.',
        'Beta matches the shipping flag (on for REB and AST, off for PTS, STL and BLK).\n'
        'Five books; STL and BLK score raw through the STL/BLK-capable MC. British English,\n'
        'no em dashes.',
    ),
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Extend stat_bss_persist to STL/BLK.")
    ap.add_argument("--apply", action="store_true", help="write changes (default: dry-run)")
    args = ap.parse_args(argv)

    try:
        with open(TARGET, encoding="utf-8") as fh:
            text = fh.read()
    except FileNotFoundError:
        print(f"ABORT: {TARGET} not found; run from the repo root.")
        return 2

    plan = []           # (index, action) where action in {"apply", "already"}
    for i, (old, new) in enumerate(EDITS, 1):
        if new in text:
            plan.append((i, "already"))
            continue
        n = text.count(old)
        if n != 1:
            print(f"ABORT: edit {i} anchor count {n} (expected 1). Nothing written.")
            print(f"       anchor: {old[:70]!r}...")
            return 3
        plan.append((i, "apply"))

    to_apply = [i for i, a in plan if a == "apply"]
    already = [i for i, a in plan if a == "already"]

    if already:
        print(f"already applied: edits {already}")
    if not to_apply:
        print("nothing to do (fully applied).")
        return 0
    print(f"will apply: edits {to_apply}")

    if not args.apply:
        print("\ndry-run only. Re-run with --apply to write.")
        return 0

    new_text = text
    for i, (old, new) in enumerate(EDITS, 1):
        if new in new_text:
            continue
        new_text = new_text.replace(old, new, 1)

    shutil.copyfile(TARGET, TARGET + ".bak")
    with open(TARGET, "w", encoding="utf-8") as fh:
        fh.write(new_text)
    print(f"\napplied edits {to_apply}; backup at {TARGET}.bak")
    return 0


if __name__ == "__main__":
    sys.exit(main())
