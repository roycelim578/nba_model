"""Delete the dead stat-leader mechanisms, spent diagnostics, the applied forward
patchers, and their out/ artefacts. Dry-run by default; pass --apply to delete.
Git is the rollback (push before running). Skips anything already absent, so it
is safe to re-run. Never touches player_timeseries.py or any live module."""

from __future__ import annotations

import argparse
import os
import shutil

ROOT_DEFAULT = os.path.expanduser("~/Desktop/QuantDev_Project/nba_model")

# dead mechanisms (feature layer)
FILES = [
    "scripts/features/stat_leader/reb_env.py",
    "scripts/features/stat_leader/correlation.py",
    "scripts/features/stat_leader/corr2.py",
    "scripts/features/stat_leader/fano_hier.py",
    # spent diagnostics (modelling layer)
    "scripts/modelling/stat_leader/calib_run.py",
    "scripts/modelling/stat_leader/ablate.py",
    "scripts/modelling/stat_leader/shrinkage.py",
    "scripts/modelling/stat_leader/drift.py",
    "scripts/modelling/stat_leader/avail_probe.py",
    "scripts/modelling/stat_leader/oof_names.py",
    "scripts/modelling/stat_leader/trade_audit.py",
    "scripts/modelling/stat_leader/env_settle.py",
    "scripts/modelling/stat_leader/pmass.py",
    "scripts/modelling/stat_leader/comove.py",
    "scripts/modelling/stat_leader/coverage.py",
    # applied forward patchers (repo-root scratch, if still present)
    "hier_fano_wire.py",
    "corr_wire.py",
    "v2_wire.py",
]

# out/ artefacts from the purged diagnostics
DIRS = [
    "out/v2_batch",
    "out/calib",
    "out/env_settle",
    "out/viz",
    "out/oof",
    "out/trade_audit",
]
LOOSE = [
    "out/env_settle.log",
    "out/calib.log",
]

PYCACHE = [
    "scripts/features/stat_leader/__pycache__",
    "scripts/modelling/stat_leader/__pycache__",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT_DEFAULT)
    ap.add_argument("--apply", action="store_true", help="actually delete (default dry-run)")
    a = ap.parse_args()
    root = a.root
    deleted, skipped = [], []

    def rm_file(rel):
        p = os.path.join(root, rel)
        if os.path.isfile(p):
            if a.apply:
                os.remove(p)
            deleted.append(rel)
        else:
            skipped.append(rel)

    def rm_dir(rel):
        p = os.path.join(root, rel)
        if os.path.isdir(p):
            if a.apply:
                shutil.rmtree(p)
            deleted.append(rel + "/")
        else:
            skipped.append(rel + "/")

    for f in FILES:
        rm_file(f)
    for d in DIRS:
        rm_dir(d)
    for f in LOOSE:
        rm_file(f)
    for d in PYCACHE:
        rm_dir(d)

    mode = "DELETED" if a.apply else "WOULD DELETE (dry-run)"
    print(f"[{mode}] {len(deleted)} item(s):")
    for x in deleted:
        print(f"   - {x}")
    print(f"\nskipped (already absent) {len(skipped)}:")
    for x in skipped:
        print(f"   . {x}")
    if not a.apply:
        print("\nre-run with --apply to delete. git is your rollback.")


if __name__ == "__main__":
    main()
