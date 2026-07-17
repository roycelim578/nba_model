#!/usr/bin/env python3
"""Anchored idempotent patcher: catch STL and BLK up to PRA in the stat-leader arm.

Phases (apply in order, one at a time):
  A       modelling layer: rates, volume, mc, scorecard, player_timeseries.
          No shared voter code, no unseal. Safe to apply before any validation.
  wire    integration wiring: stat_pricejoin, stat_samples, registry loop.
          No unseal. Registry rows are inert while 2024 is sealed by default.
  unseal  config.py only: SEAL_REGISTRY STL/BLK -> [2025] and BOOK_WEIGHTS[2024]
          += STL/BLK 1000. This is the ONLY shared-code and only-unseal edit.
          Apply ONLY after the scorecard and panels have been graded.

Run from the repo root (~/Desktop/QuantDev_Project/nba_model):
  uv run python3 patch_stl_blk.py --phase A            # dry run, reports only
  uv run python3 patch_stl_blk.py --phase A --apply     # writes, .bak backups

Idempotent and atomic-per-phase: every anchor must match exactly once, else the
whole phase aborts and nothing is written. A re-run after a successful apply is a
no-op (each edit detects its own completed state). British English.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

# (relative_path, OLD_anchor, NEW_text). OLD must appear exactly once. NEW is the
# post-edit text; its verbatim presence with OLD absent marks the edit done.
EDITS: dict[str, list[tuple[str, str, str]]] = {
    "A": [
        # ---- rates.py: substrate counts ----
        ("scripts/features/stat_leader/rates.py",
         '    "fta", "ftm", "reb", "potential_ast_asof", "ast",',
         '    "fta", "ftm", "reb", "stl", "blk", "potential_ast_asof", "ast",'),
        ("scripts/features/stat_leader/rates.py",
         '        "turnovers, fga, fgm, fg3a, fg3m, fta, ftm "',
         '        "steals, blocks, turnovers, fga, fgm, fg3a, fg3m, fta, ftm "'),
        # winsor parity: STL/BLK mirror reb's per-game clip scaffolding exactly,
        # inert unless VOL_WINSOR_K_STL / _K_BLK are set (off in the baseline, like reb).
        ("scripts/features/stat_leader/rates.py",
         'K_POTAST = _envf("VOL_WINSOR_K_POTAST")',
         'K_POTAST = _envf("VOL_WINSOR_K_POTAST")\n'
         'K_STL = _envf("VOL_WINSOR_K_STL")\n'
         'K_BLK = _envf("VOL_WINSOR_K_BLK")'),
        ("scripts/features/stat_leader/rates.py",
         '    _tot_min = _tot_used = _tot_reb = 0.0',
         '    _tot_min = _tot_used = _tot_reb = _tot_stl = _tot_blk = 0.0'),
        ("scripts/features/stat_leader/rates.py",
         '            _tot_reb += _g["rebounds"] or 0.0',
         '            _tot_reb += _g["rebounds"] or 0.0\n'
         '            _tot_stl += _g["steals"] or 0.0\n'
         '            _tot_blk += _g["blocks"] or 0.0'),
        ("scripts/features/stat_leader/rates.py",
         '    mu0_reb = (_tot_reb / _tot_min) if _tot_min > 0 else 0.0',
         '    mu0_reb = (_tot_reb / _tot_min) if _tot_min > 0 else 0.0\n'
         '    mu0_stl = (_tot_stl / _tot_min) if _tot_min > 0 else 0.0\n'
         '    mu0_blk = (_tot_blk / _tot_min) if _tot_min > 0 else 0.0'),
        ("scripts/features/stat_leader/rates.py",
         '        ref_r = _RunRef(mu0_reb, REF_SEED_MIN, REF_HL_MIN)',
         '        ref_r = _RunRef(mu0_reb, REF_SEED_MIN, REF_HL_MIN)\n'
         '        ref_s = _RunRef(mu0_stl, REF_SEED_MIN, REF_HL_MIN)\n'
         '        ref_b = _RunRef(mu0_blk, REF_SEED_MIN, REF_HL_MIN)'),
        ("scripts/features/stat_leader/rates.py",
         '                    reb_g = g["rebounds"] or 0.0',
         '                    reb_g = g["rebounds"] or 0.0\n'
         '                    stl_g = g["steals"] or 0.0\n'
         '                    blk_g = g["blocks"] or 0.0'),
        ("scripts/features/stat_leader/rates.py",
         '                    f_r = _winsor_factor(reb_g, mn, ref_r, K_REB, c["gp_played_asof"], MIN_GAMES)',
         '                    f_r = _winsor_factor(reb_g, mn, ref_r, K_REB, c["gp_played_asof"], MIN_GAMES)\n'
         '                    f_s = _winsor_factor(stl_g, mn, ref_s, K_STL, c["gp_played_asof"], MIN_GAMES)\n'
         '                    f_b = _winsor_factor(blk_g, mn, ref_b, K_BLK, c["gp_played_asof"], MIN_GAMES)'),
        ("scripts/features/stat_leader/rates.py",
         '                    c["reb"] += reb_g * f_r',
         '                    c["reb"] += reb_g * f_r\n'
         '                    c["stl"] += stl_g * f_s\n'
         '                    c["blk"] += blk_g * f_b'),
        # ---- volume.py: two new single-column rate nodes ----
        ("scripts/features/stat_leader/volume.py",
         '    "ast_create": ["potential_ast_asof"],',
         '    "ast_create": ["potential_ast_asof"],\n'
         '    "stl": ["stl"],\n'
         '    "blk": ["blk"],'),
        # ---- mc.py: dicts, banked, branches, prior loop, cli ----
        ("scripts/modelling/stat_leader/mc.py",
         'STAT_AWARD = {"pts": "PTS", "reb": "REB", "ast": "AST"}',
         'STAT_AWARD = {"pts": "PTS", "reb": "REB", "ast": "AST", "stl": "STL", "blk": "BLK"}'),
        ("scripts/modelling/stat_leader/mc.py",
         'VOL_NODE = {"reb": "reb", "pts": "usage", "ast": "ast_create"}',
         'VOL_NODE = {"reb": "reb", "pts": "usage", "ast": "ast_create", "stl": "stl", "blk": "blk"}'),
        ("scripts/modelling/stat_leader/mc.py",
         'def _banked_ast(d):\n    return d.get("ast") or 0.0',
         'def _banked_ast(d):\n    return d.get("ast") or 0.0\n\n\n'
         'def _banked_stl(d):\n    return d.get("stl") or 0.0\n\n\n'
         'def _banked_blk(d):\n    return d.get("blk") or 0.0'),
        ("scripts/modelling/stat_leader/mc.py",
         'BANKED = {"reb": _banked_reb, "pts": _banked_pts, "ast": _banked_ast}',
         'BANKED = {"reb": _banked_reb, "pts": _banked_pts, "ast": _banked_ast, '
         '"stl": _banked_stl, "blk": _banked_blk}'),
        ("scripts/modelling/stat_leader/mc.py",
         '    return V._draw_count(rng, rate, rem_min, vpriors["fano"].get("reb", 1.0))',
         '    return V._draw_count(rng, rate, rem_min, vpriors["fano"].get("reb", 1.0))\n\n\n'
         'def _rem_stl(rng, d, cohort, vpriors, npriors, rem_min, k):\n'
         '    vc, vm = V._vol_count(d, "stl")\n'
         '    if vc is None:\n'
         '        return np.zeros(k)\n'
         '    rate = _gamma_rate(rng, vpriors, "stl", cohort, vc, vm, k,\n'
         '                       own_rate=d.get("prior_rate_stl"), own_min=d.get("prior_min") or 0.0)\n'
         '    return V._draw_count(rng, rate, rem_min, vpriors["fano"].get("stl", 1.0))\n\n\n'
         'def _rem_blk(rng, d, cohort, vpriors, npriors, rem_min, k):\n'
         '    vc, vm = V._vol_count(d, "blk")\n'
         '    if vc is None:\n'
         '        return np.zeros(k)\n'
         '    rate = _gamma_rate(rng, vpriors, "blk", cohort, vc, vm, k,\n'
         '                       own_rate=d.get("prior_rate_blk"), own_min=d.get("prior_min") or 0.0)\n'
         '    return V._draw_count(rng, rate, rem_min, vpriors["fano"].get("blk", 1.0))'),
        ("scripts/modelling/stat_leader/mc.py",
         'BRANCH = {"reb": _rem_reb, "pts": _rem_pts, "ast": _rem_ast}',
         'BRANCH = {"reb": _rem_reb, "pts": _rem_pts, "ast": _rem_ast, "stl": _rem_stl, "blk": _rem_blk}'),
        ("scripts/modelling/stat_leader/mc.py",
         '        for node in ("reb", "usage", "ast_create"):',
         '        for node in ("reb", "usage", "ast_create", "stl", "blk"):'),
        ("scripts/modelling/stat_leader/mc.py",
         '    p.add_argument("--stat", default="all", choices=["reb", "pts", "ast", "all"])',
         '    p.add_argument("--stat", default="all", choices=["reb", "pts", "ast", "stl", "blk", "all"])'),
        ("scripts/modelling/stat_leader/mc.py",
         '    stats = ["reb", "pts", "ast"] if args.stat == "all" else [args.stat]',
         '    stats = ["reb", "pts", "ast", "stl", "blk"] if args.stat == "all" else [args.stat]'),
        # ---- scorecard.py: floor + cli ----
        ("scripts/modelling/stat_leader/scorecard.py",
         'STAT_FLOOR = {"pts": 1997, "reb": 1997, "ast": 2013}',
         'STAT_FLOOR = {"pts": 1997, "reb": 1997, "ast": 2013, "stl": 1997, "blk": 1997}'),
        ("scripts/modelling/stat_leader/scorecard.py",
         '    p.add_argument("--stat", default="all", choices=["reb", "pts", "ast", "all"])',
         '    p.add_argument("--stat", default="all", choices=["reb", "pts", "ast", "stl", "blk", "all"])'),
        ("scripts/modelling/stat_leader/scorecard.py",
         '    stats = ["reb", "pts", "ast"] if args.stat == "all" else [args.stat]',
         '    stats = ["reb", "pts", "ast", "stl", "blk"] if args.stat == "all" else [args.stat]'),
        # ---- player_timeseries.py: floor + leaderboard temp + cli ----
        ("scripts/modelling/stat_leader/player_timeseries.py",
         'STAT_FLOOR = {"pts": 1997, "reb": 1997, "ast": 2013}',
         'STAT_FLOOR = {"pts": 1997, "reb": 1997, "ast": 2013, "stl": 1997, "blk": 1997}'),
        ("scripts/modelling/stat_leader/player_timeseries.py",
         'LB_T = {"reb": 0.5, "pts": 1.5, "ast": 0.5}',
         'LB_T = {"reb": 0.5, "pts": 1.5, "ast": 0.5, "stl": 0.5, "blk": 0.5}'),
        ("scripts/modelling/stat_leader/player_timeseries.py",
         '    p.add_argument("--stat", default="all", choices=["reb", "pts", "ast", "all"])',
         '    p.add_argument("--stat", default="all", choices=["reb", "pts", "ast", "stl", "blk", "all"])'),
        ("scripts/modelling/stat_leader/player_timeseries.py",
         '    stats = ["reb", "pts", "ast"] if args.stat == "all" else [args.stat]',
         '    stats = ["reb", "pts", "ast", "stl", "blk"] if args.stat == "all" else [args.stat]'),
    ],
    "wire": [
        # ---- stat_pricejoin.py: routing tags + slug-fallback regexes ----
        ("scripts/backtest/stat_leader/stat_pricejoin.py",
         'STAT_BOOK_AWARD = {"pts": "PTS_LEADER", "reb": "REB_LEADER", "ast": "AST_LEADER"}',
         'STAT_BOOK_AWARD = {"pts": "PTS_LEADER", "reb": "REB_LEADER", "ast": "AST_LEADER", '
         '"stl": "STL_LEADER", "blk": "BLK_LEADER"}'),
        ("scripts/backtest/stat_leader/stat_pricejoin.py",
         r'    "ast": re.compile(r"assist|\bapg\b", re.I),',
         r'    "ast": re.compile(r"assist|\bapg\b", re.I),' + '\n'
         + r'    "stl": re.compile(r"steal|\bspg\b|top thief", re.I),' + '\n'
         + r'    "blk": re.compile(r"block|\bbpg\b|top shot ?blocker", re.I),'),
        ("scripts/backtest/stat_leader/stat_pricejoin.py",
         r'    "ast": re.compile(r"\b(assists?|apg)\b", re.I),',
         r'    "ast": re.compile(r"\b(assists?|apg)\b", re.I),' + '\n'
         + r'    "stl": re.compile(r"\b(steals?|spg)\b", re.I),' + '\n'
         + r'    "blk": re.compile(r"\b(blocks?|bpg)\b", re.I),'),
        # ---- stat_samples.py: cli ----
        ("scripts/backtest/stat_leader/stat_samples.py",
         '    p.add_argument("--stat", default="reb", choices=["reb", "pts", "ast"])',
         '    p.add_argument("--stat", default="reb", choices=["reb", "pts", "ast", "stl", "blk"])'),
        # ---- registry.py: one-token loop extension (rows added after DEFAULT_AWARDS) ----
        ("scripts/backtest/registry.py",
         'for _b in ("PTS", "REB", "AST"):',
         'for _b in ("PTS", "REB", "AST", "STL", "BLK"):'),
    ],
    "unseal": [
        # ---- config.py: validation-only 2024 budgets + unseal 2024 for STL/BLK ----
        ("scripts/common/config.py",
         '           "PTS": 1000, "REB": 1000, "AST": 1000},',
         '           "PTS": 1000, "REB": 1000, "AST": 1000,\n'
         '           "STL": 1000, "BLK": 1000},'),
        ("scripts/common/config.py",
         '    "AST": [2025],',
         '    "AST": [2025],\n    "STL": [2025],\n    "BLK": [2025],'),
    ],
}


def _classify(text: str, old: str, new: str) -> str:
    """Return one of: apply, done, ambiguous, drift.

    NEW is checked first: several edits append while leaving the anchor verbatim
    (NEW = OLD + additions), so the anchor survives a successful apply. NEW is the
    full post-edit text and its additions never exist pre-apply, so its verbatim
    presence is the reliable done-signal for every edit shape."""
    if new in text:
        return "done"
    n_old = text.count(old)
    if n_old == 1:
        return "apply"
    if n_old == 0:
        return "drift"
    return "ambiguous"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="STL/BLK stat-leader patcher.")
    ap.add_argument("--phase", required=True, choices=sorted(EDITS))
    ap.add_argument("--apply", action="store_true", help="write changes (default dry run)")
    ap.add_argument("--root", default=".", help="repo root (default cwd)")
    args = ap.parse_args(argv)

    root = os.path.abspath(args.root)
    edits = EDITS[args.phase]

    # ---- pass 1: classify everything, abort the whole phase on any problem ----
    cache: dict[str, str] = {}
    plan: list[tuple[str, str, str, str, str]] = []  # (rel, abspath, old, new, status)
    bad = False
    for rel, old, new in edits:
        path = os.path.join(root, rel)
        if not os.path.exists(path):
            print(f"  MISSING FILE  {rel}")
            bad = True
            continue
        if path not in cache:
            with open(path, "r", encoding="utf-8") as fh:
                cache[path] = fh.read()
        status = _classify(cache[path], old, new)
        tag = {"apply": "APPLY", "done": "skip (done)",
               "ambiguous": "DRIFT (anchor not unique)",
               "drift": "DRIFT (anchor missing)"}[status]
        print(f"  {tag:<26} {rel}")
        if status in ("ambiguous", "drift"):
            bad = True
        plan.append((rel, path, old, new, status))

    if bad:
        print(f"\nphase {args.phase}: ABORTED, anchor drift or missing file. Nothing written.")
        return 2

    to_apply = [p for p in plan if p[4] == "apply"]
    if not to_apply:
        print(f"\nphase {args.phase}: all {len(plan)} edits already applied. No-op.")
        return 0

    if not args.apply:
        print(f"\nphase {args.phase}: dry run OK, {len(to_apply)} edit(s) ready. "
              f"Re-run with --apply to write.")
        return 0

    # ---- pass 2: back up each file once, then write ----
    new_text: dict[str, str] = dict(cache)
    for rel, path, old, new, status in to_apply:
        new_text[path] = new_text[path].replace(old, new, 1)
    touched = {p[1] for p in to_apply}
    for path in sorted(touched):
        bak = f"{path}.bak.stlblk_{args.phase}"
        if not os.path.exists(bak):
            shutil.copy2(path, bak)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(new_text[path])
        print(f"  wrote  {os.path.relpath(path, root)}  (backup {os.path.basename(bak)})")
    print(f"\nphase {args.phase}: applied {len(to_apply)} edit(s) across {len(touched)} file(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
