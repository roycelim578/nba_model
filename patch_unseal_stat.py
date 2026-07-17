"""Unseal the five stat books in config.SEAL_REGISTRY.

Deliberate, one-time action. It removes the default double-seal on PTS, REB, AST,
STL and BLK so the engine will run them on 2024 (their burnt dev season, needed for
the cross-arm correlation and the bss-vs-equal strategy read) and on 2025 (their
one-shot sealed test, which we are consciously spending now). This mirrors how the
voter books went to [] after their one-shot. After this lands there is no further
held-out season for the stat arm, so treat every subsequent 2025 stat number as the
final sealed result and do not re-tune against it.

Patcher discipline: dry-run by default, --apply to write, .bak backup, anchor count
must be exactly 1, aborts on drift, refuses to double-apply. British English.

  uv run python3 patch_unseal_stat.py            # dry-run
  uv run python3 patch_unseal_stat.py --apply     # write
"""
from __future__ import annotations

import argparse
import shutil
import sys

TARGET = "scripts/common/config.py"

ANCHOR = '    "6MOTY": [2024, 2025],\n}'

INSERT = (
    '    "6MOTY": [2024, 2025],\n'
    '    "PTS": [],\n'
    '    "REB": [],\n'
    '    "AST": [],\n'
    '    "STL": [],\n'
    '    "BLK": [],\n'
    '}'
)

ALREADY = '    "PTS": [],\n    "REB": [],'


def main() -> int:
    ap = argparse.ArgumentParser(description="Unseal the five stat books.")
    ap.add_argument("--apply", action="store_true", help="write the change (default dry-run)")
    ap.add_argument("--path", default=TARGET, help="config path relative to repo root")
    args = ap.parse_args()

    with open(args.path, "r", encoding="utf-8") as fh:
        src = fh.read()

    if ALREADY in src:
        print("already applied: stat books present in SEAL_REGISTRY, no change.")
        return 0

    count = src.count(ANCHOR)
    if count != 1:
        print(f"ABORT: anchor found {count} times, expected 1. File has drifted from "
              f"the expected SEAL_REGISTRY layout; not touching it.")
        return 2

    new = src.replace(ANCHOR, INSERT, 1)

    if not args.apply:
        print("DRY-RUN. Would add PTS/REB/AST/STL/BLK = [] to SEAL_REGISTRY.")
        print("--- new SEAL_REGISTRY block ---")
        start = new.index("SEAL_REGISTRY = {")
        end = new.index("}", start) + 1
        print(new[start:end])
        print("--- run again with --apply to write ---")
        return 0

    shutil.copyfile(args.path, args.path + ".bak")
    with open(args.path, "w", encoding="utf-8") as fh:
        fh.write(new)
    print(f"APPLIED. Backup at {args.path}.bak")
    print("Stat books unsealed for all seasons. The 2025 one-shot is now spendable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
