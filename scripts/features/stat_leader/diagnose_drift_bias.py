"""Diagnose whether within-season drift is a real, systematic effect, separate
from the tail-contamination issue the winsor patch addresses. Reuses the same
as-of/realised-remaining machinery as the s-fit and the winsor tests, but reports
SIGNED bias (projected minus realised) by season stage, not magnitude. Pure noise
around a correctly-centred mean averages to zero; a persistent one-sided bias,
especially a negative one early (the model under-projects, real remaining
production comes in above what was projected), is the concrete trace of the
under-tracked-drift failure mode. This tests the CURRENT unmodified projection
(no winsor, no compression); it exists to settle whether drift is worth building
anything for, not to size a fix.

Also reports the fraction of contender-snapshots with positive vs negative bias:
a systematic direction (e.g. 65/35) argues for real drift; a roughly even split
around a near-zero mean argues the effect is closer to symmetric noise.

Seasons 2024/2025 excluded (2024 stat OOS, 2025 sealed).

  uv run python3 -m scripts.features.stat_leader.diagnose_drift_bias
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
from scripts.common.db import connect
from scripts.features.stat_leader import rates as R
try:
    from scripts.features.stat_leader import nodes as N
    from scripts.modelling.stat_leader import mc as MC
except ImportError:  # pragma: no cover
    import nodes as N  # type: ignore
    import mc as MC  # type: ignore

NODE_STAT = {"usage": "pts", "reb": "reb", "potast": "ast"}

MIN_GP = 15
MIN_REM_MIN = 300.0
CONTENDER_PCTL = 70.0
STAGES = {"early": (0.0, 1 / 3), "mid": (1 / 3, 2 / 3), "late": (2 / 3, 1.0 + 1e-9)}


def _count(g, node):
    if node == "reb":
        return g["rebounds"] or 0.0
    return (g["fga"] or 0.0) + R.FT_POSS_COEF * (g["fta"] or 0.0) + (g["turnovers"] or 0.0)


def _season_obs(logs, node, snaps, leader_pid):
    """(banked as-of rate, realised remaining rate, remaining minutes, frac)."""
    obs = []
    n_snap = len(snaps)
    for pid, games in logs.items():
        games = sorted(games, key=lambda x: x["game_date"])
        tot_c = sum(_count(g, node) for g in games if (g["minutes"] or 0) > 0)
        tot_m = sum((g["minutes"] or 0.0) for g in games if (g["minutes"] or 0) > 0)
        if tot_m <= 0:
            continue
        raw_c = mn_c = 0.0
        gp = gi = 0
        for si, snap in enumerate(snaps):
            while gi < len(games) and games[gi]["game_date"] <= snap:
                g = games[gi]; m = g["minutes"] or 0.0
                if m > 0:
                    raw_c += _count(g, node); mn_c += m; gp += 1
                gi += 1
            if gp >= MIN_GP and mn_c > 0:
                rem_c = tot_c - raw_c; rem_m = tot_m - mn_c
                if rem_m >= MIN_REM_MIN:
                    frac = si / max(1, n_snap - 1)
                    obs.append((raw_c / mn_c, rem_c / rem_m, rem_m, frac,
                                1 if pid == leader_pid else 0))
    return obs


def _season_obs_potast(logs, potast, snaps, leader_pid):
    obs = []
    n_snap = len(snaps)
    for pid, games in logs.items():
        games = sorted(games, key=lambda x: x["game_date"])
        tot_m = sum((g["minutes"] or 0.0) for g in games if (g["minutes"] or 0) > 0)
        vals = potast.get(pid, {})
        if tot_m <= 0 or not vals:
            continue
        tot_c = vals[max(vals)]
        mn_c = 0.0
        gp = gi = 0
        for si, snap in enumerate(snaps):
            while gi < len(games) and games[gi]["game_date"] <= snap:
                m = games[gi]["minutes"] or 0.0
                if m > 0:
                    mn_c += m; gp += 1
                gi += 1
            raw_now = vals.get(snap, 0.0)
            if gp >= MIN_GP and mn_c > 0:
                rem_c = tot_c - raw_now; rem_m = tot_m - mn_c
                if rem_m >= MIN_REM_MIN:
                    frac = si / max(1, n_snap - 1)
                    obs.append((raw_now / mn_c, rem_c / rem_m, rem_m, frac,
                                1 if pid == leader_pid else 0))
    return obs


def _stage_of(frac):
    for st, (lo, hi) in STAGES.items():
        if lo <= frac < hi:
            return st
    return "late"


def _report(node, seasons_obs):
    print(f"\n{'-'*72}\nnode = {node}\n{'-'*72}")
    print(f"  {'stage':>6} {'n':>6} {'signed_bias':>12} {'mean|bias|':>11} "
          f"{'%positive':>10} {'%negative':>10}")
    for st in ("early", "mid", "late", "all"):
        rows = [o for o in seasons_obs if st == "all" or _stage_of(o[3]) == st]
        if not rows:
            print(f"  {st:>6}      no observations"); continue
        w = sum(o[2] for o in rows)
        bias = sum((o[0] - o[1]) * o[2] for o in rows) / w
        mabs = sum(abs(o[0] - o[1]) * o[2] for o in rows) / w
        npos = sum(1 for o in rows if o[0] > o[1])
        nneg = sum(1 for o in rows if o[0] < o[1])
        n = len(rows)
        print(f"  {st:>6} {n:>6} {bias:>+12.4f} {mabs:>11.4f} "
              f"{100*npos/n:>9.1f}% {100*nneg/n:>9.1f}%")
    print("  bias = mean(banked_as_of_rate - realised_remaining_rate), exposure-weighted.")
    print("  negative bias = model's as-of rate sits BELOW what the player actually goes on to")
    print("  do for the rest of the season, i.e. under-projection consistent with untracked upward drift.")


def _leaders_by_season(conn, seasons):
    """Eventual leader pid per (season, stat), from the same qualifier and
    argmax path the scorecard uses (mc.realised_eff), never a second
    implementation. Computed once per season, not once per node."""
    out = {}
    for season in seasons:
        _, finals, _, _ = N._load(conn, [season])
        ftg = MC._load_ftg(conn, season)
        for stat in ("pts", "reb", "ast"):
            eff = MC.realised_eff(finals, ftg, season, stat)
            if eff:
                out[(season, stat)] = max(eff, key=eff.get)
    return out


def _report_leader_split(node, seasons_obs):
    """Signed bias split into the eventual leader versus the rest of the
    field, by stage. The early-stage leader-minus-field gap is the number the
    go/no-go turns on: pooled bias is close to rank-neutral, so the defect
    (leaders cashing above stated probability) requires leaders to be
    under-projected MORE than the field, not the field under-projected too."""
    print(f"\n  leader vs field  (leader = eventual {NODE_STAT[node].upper()} leader)")
    print(f"  {'stage':>6} {'grp':>7} {'n':>6} {'signed_bias':>12} {'%negative':>10}")
    gaps = {}
    for st in ("early", "mid", "late", "all"):
        rows = [o for o in seasons_obs if st == "all" or _stage_of(o[3]) == st]
        cell = {}
        for grp, want in (("leader", 1), ("field", 0)):
            g = [o for o in rows if o[4] == want]
            if not g:
                cell[grp] = None
                print(f"  {st:>6} {grp:>7}      no observations")
                continue
            w = sum(o[2] for o in g)
            bias = sum((o[0] - o[1]) * o[2] for o in g) / w
            nneg = sum(1 for o in g if o[0] < o[1])
            cell[grp] = bias
            print(f"  {st:>6} {grp:>7} {len(g):>6} {bias:>+12.4f} {100*nneg/len(g):>9.1f}%")
        if cell.get("leader") is not None and cell.get("field") is not None:
            gaps[st] = cell["leader"] - cell["field"]
    if "early" in gaps:
        print(f"  --> EARLY leader-minus-field gap = {gaps['early']:+.4f}  "
              f"(more negative = leaders under-projected MORE than field = the defect)")


def main(argv=None):
    p = argparse.ArgumentParser(description="Diagnose whether within-season drift bias is real.")
    p.add_argument("--db", default="data/awards.db")
    p.add_argument("--node", default="all", choices=["all", "usage", "reb", "potast"])
    p.add_argument("--min-season", type=int, default=2008)
    p.add_argument("--max-season", type=int, default=2023)
    args = p.parse_args(argv)

    conn = connect(args.db)
    nodes = ["usage", "reb", "potast"] if args.node == "all" else [args.node]
    seasons = list(range(args.min_season, args.max_season + 1))
    print(f"drift-bias diagnostic  seasons {seasons[0]}-{seasons[-1]}  (current, unmodified projection)")
    leaders = _leaders_by_season(conn, seasons)

    for node in nodes:
        all_obs = []
        for season in seasons:
            logs = R._load_logs(conn, season)
            snaps = R._grid(conn, season)
            if not snaps:
                continue
            leader_pid = leaders.get((season, NODE_STAT[node]))
            if node == "potast":
                potast = R._load_potast(conn, season)
                all_obs.extend(_season_obs_potast(logs, potast, snaps, leader_pid))
            else:
                all_obs.extend(_season_obs(logs, node, snaps, leader_pid))
        _report(node, all_obs)
        _report_leader_split(node, all_obs)

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
