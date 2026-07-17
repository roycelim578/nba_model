"""Anchored idempotent patcher: add an eventual-leader vs field split to the
drift-bias diagnostic.

The existing pooled rows are the regression anchor and are left byte-identical.
This adds, per node, a second block that splits the same signed bias into the
eventual leader versus the rest of the field, and prints the early-stage
leader-minus-field gap as the single go/no-go number. The leader label is taken
from the SAME path the scorecard uses (mc.realised_eff argmax with the
ceil(0.70*ftg) qualifier), never a second implementation, via NODE_STAT mapping
each node to its stat book (reb->reb identity; usage->pts and potast->ast are the
upstream nodes feeding those composed books).

Usage:
  python3 patch_drift_leader_split.py            # dry run, shows each edit
  python3 patch_drift_leader_split.py --apply     # writes, with .bak backup
"""

from __future__ import annotations

import argparse
import shutil
import sys

TARGET = "diagnose_drift_bias.py"
SENTINEL = "def _report_leader_split("

EDITS = [
    (
        "from scripts.common.db import connect\n"
        "from scripts.features.stat_leader import rates as R\n",
        "from scripts.common.db import connect\n"
        "from scripts.features.stat_leader import rates as R\n"
        "try:\n"
        "    from scripts.features.stat_leader import nodes as N\n"
        "    from scripts.modelling.stat_leader import mc as MC\n"
        "except ImportError:  # pragma: no cover\n"
        "    import nodes as N  # type: ignore\n"
        "    import mc as MC  # type: ignore\n"
        "\n"
        'NODE_STAT = {"usage": "pts", "reb": "reb", "potast": "ast"}\n',
    ),
    (
        "def _season_obs(logs, node, snaps):",
        "def _season_obs(logs, node, snaps, leader_pid):",
    ),
    (
        "                    obs.append((raw_c / mn_c, rem_c / rem_m, rem_m, frac))",
        "                    obs.append((raw_c / mn_c, rem_c / rem_m, rem_m, frac,\n"
        "                                1 if pid == leader_pid else 0))",
    ),
    (
        "def _season_obs_potast(logs, potast, snaps):",
        "def _season_obs_potast(logs, potast, snaps, leader_pid):",
    ),
    (
        "                    obs.append((raw_now / mn_c, rem_c / rem_m, rem_m, frac))",
        "                    obs.append((raw_now / mn_c, rem_c / rem_m, rem_m, frac,\n"
        "                                1 if pid == leader_pid else 0))",
    ),
    (
        "def main(argv=None):\n",
        "def _leaders_by_season(conn, seasons):\n"
        '    """Eventual leader pid per (season, stat), from the same qualifier and\n'
        "    argmax path the scorecard uses (mc.realised_eff), never a second\n"
        '    implementation. Computed once per season, not once per node."""\n'
        "    out = {}\n"
        "    for season in seasons:\n"
        "        _, finals, _, _ = N._load(conn, [season])\n"
        "        ftg = MC._load_ftg(conn, season)\n"
        '        for stat in ("pts", "reb", "ast"):\n'
        "            eff = MC.realised_eff(finals, ftg, season, stat)\n"
        "            if eff:\n"
        "                out[(season, stat)] = max(eff, key=eff.get)\n"
        "    return out\n"
        "\n"
        "\n"
        "def _report_leader_split(node, seasons_obs):\n"
        '    """Signed bias split into the eventual leader versus the rest of the\n'
        "    field, by stage. The early-stage leader-minus-field gap is the number the\n"
        "    go/no-go turns on: pooled bias is close to rank-neutral, so the defect\n"
        "    (leaders cashing above stated probability) requires leaders to be\n"
        '    under-projected MORE than the field, not the field under-projected too."""\n'
        f'    print(f"\\n  leader vs field  (leader = eventual {{NODE_STAT[node].upper()}} leader)")\n'
        "    print(f\"  {'stage':>6} {'grp':>7} {'n':>6} {'signed_bias':>12} {'%negative':>10}\")\n"
        "    gaps = {}\n"
        '    for st in ("early", "mid", "late", "all"):\n'
        '        rows = [o for o in seasons_obs if st == "all" or _stage_of(o[3]) == st]\n'
        "        cell = {}\n"
        '        for grp, want in (("leader", 1), ("field", 0)):\n'
        "            g = [o for o in rows if o[4] == want]\n"
        "            if not g:\n"
        "                cell[grp] = None\n"
        '                print(f"  {st:>6} {grp:>7}      no observations")\n'
        "                continue\n"
        "            w = sum(o[2] for o in g)\n"
        "            bias = sum((o[0] - o[1]) * o[2] for o in g) / w\n"
        "            nneg = sum(1 for o in g if o[0] < o[1])\n"
        "            cell[grp] = bias\n"
        '            print(f"  {st:>6} {grp:>7} {len(g):>6} {bias:>+12.4f} {100*nneg/len(g):>9.1f}%")\n'
        '        if cell.get("leader") is not None and cell.get("field") is not None:\n'
        '            gaps[st] = cell["leader"] - cell["field"]\n'
        '    if "early" in gaps:\n'
        "        print(f\"  --> EARLY leader-minus-field gap = {gaps['early']:+.4f}  \"\n"
        '              f"(more negative = leaders under-projected MORE than field = the defect)")\n'
        "\n"
        "\n"
        "def main(argv=None):\n",
    ),
    (
        "    conn = connect(args.db)\n"
        '    nodes = ["usage", "reb", "potast"] if args.node == "all" else [args.node]\n'
        "    seasons = list(range(args.min_season, args.max_season + 1))\n"
        '    print(f"drift-bias diagnostic  seasons {seasons[0]}-{seasons[-1]}  (current, unmodified projection)")\n'
        "\n"
        "    for node in nodes:\n"
        "        all_obs = []\n"
        "        for season in seasons:\n"
        "            logs = R._load_logs(conn, season)\n"
        "            snaps = R._grid(conn, season)\n"
        "            if not snaps:\n"
        "                continue\n"
        '            if node == "potast":\n'
        "                potast = R._load_potast(conn, season)\n"
        "                all_obs.extend(_season_obs_potast(logs, potast, snaps))\n"
        "            else:\n"
        "                all_obs.extend(_season_obs(logs, node, snaps))\n"
        "        _report(node, all_obs)\n",
        "    conn = connect(args.db)\n"
        '    nodes = ["usage", "reb", "potast"] if args.node == "all" else [args.node]\n'
        "    seasons = list(range(args.min_season, args.max_season + 1))\n"
        '    print(f"drift-bias diagnostic  seasons {seasons[0]}-{seasons[-1]}  (current, unmodified projection)")\n'
        "    leaders = _leaders_by_season(conn, seasons)\n"
        "\n"
        "    for node in nodes:\n"
        "        all_obs = []\n"
        "        for season in seasons:\n"
        "            logs = R._load_logs(conn, season)\n"
        "            snaps = R._grid(conn, season)\n"
        "            if not snaps:\n"
        "                continue\n"
        "            leader_pid = leaders.get((season, NODE_STAT[node]))\n"
        '            if node == "potast":\n'
        "                potast = R._load_potast(conn, season)\n"
        "                all_obs.extend(_season_obs_potast(logs, potast, snaps, leader_pid))\n"
        "            else:\n"
        "                all_obs.extend(_season_obs(logs, node, snaps, leader_pid))\n"
        "        _report(node, all_obs)\n"
        "        _report_leader_split(node, all_obs)\n",
    ),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    with open(TARGET, encoding="utf-8") as fh:
        src = fh.read()

    if SENTINEL in src:
        print("already applied (sentinel present); refusing double-apply")
        return 1

    out = src
    for i, (old, new) in enumerate(EDITS, 1):
        c = out.count(old)
        if c != 1:
            print(f"edit {i}: anchor count == {c}, expected 1; aborting")
            return 2
        out = out.replace(old, new)
        print(f"edit {i}: ok (1 anchor)")

    if not args.apply:
        print("\ndry run only; re-run with --apply to write")
        return 0

    shutil.copyfile(TARGET, TARGET + ".bak")
    with open(TARGET, "w", encoding="utf-8") as fh:
        fh.write(out)
    print(f"\napplied; backup at {TARGET}.bak")
    return 0


if __name__ == "__main__":
    sys.exit(main())
