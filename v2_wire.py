"""Anchored idempotent patcher: wire --avail-hier and --corr2.

Two independent, off-by-default features:

  AVAIL_HIER. After the availability pool draw, recentres (remaining_frac,
  remaining_mpg) onto a reliability-weighted blend of the player's own banked
  availability and his prior-season same-week-index remainder (avail_hier.py),
  keeping the pool's dispersion and tail. Fixes the durable-elite low centre.

  CORR2. Heterogeneous per-opponent shared shock (corr2.py): one shock per
  opponent team per replicate, each contender multiplied by the average shock of
  the opponents he plays, so schedule-overlapping contenders co-move and their gap
  variance shrinks. This is the order-statistic lever a common (identical-to-all)
  factor cannot be, since a common factor cancels in the ranking.

Both nest to the base engine when off, so the v1 gate and all baselines are
bit-identical. Apply AFTER hier_fano_wire.py and corr_wire.py; single-line
anchors make the order safe. Prereqs: avail_hier.py and corr2.py in
scripts/features/stat_leader/.

Run from repo root:
  uv run python3 v2_wire.py
  uv run python3 v2_wire.py --apply
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

F_MC = "scripts/modelling/stat_leader/mc.py"
F_SCORE = "scripts/modelling/stat_leader/scorecard.py"
F_ABL = "scripts/modelling/stat_leader/ablate.py"

EDITS = [
    # ---- imports
    (F_MC,
     '    from scripts.features.stat_leader import reb_env as RENV\n',
     '    from scripts.features.stat_leader import reb_env as RENV\n'
     '    from scripts.features.stat_leader import avail_hier as AH\n'
     '    from scripts.features.stat_leader import corr2 as C2\n',
     'import avail_hier as AH\n'),

    (F_MC,
     '    import reb_env as RENV  # type: ignore\n',
     '    import reb_env as RENV  # type: ignore\n'
     '    import avail_hier as AH  # type: ignore\n'
     '    import corr2 as C2  # type: ignore\n',
     'import avail_hier as AH  # type: ignore'),

    # ---- globals
    (F_MC,
     'OWN_PRIOR_K = None\n',
     'OWN_PRIOR_K = None\n'
     '# Hierarchical availability recentre (avail_hier.py) and heterogeneous v2\n'
     '# correlation (corr2.py). Off => identical engine.\n'
     'AVAIL_HIER = False\n'
     'CORR2 = False\n'
     '_CORR2_MULT = None\n'
     '_AVAIL_PRIOR = None\n'
     '_CORR2_SD = None\n',
     'AVAIL_HIER = False'),

    # ---- _c2mul helper (prepend to _gamma_rate; stacks with _fano/_corr_mu)
    (F_MC,
     'def _gamma_rate(rng, priors, node, cohort, vc, vm, k, own_rate=None, own_min=0.0):\n',
     'def _c2mul(rate):\n'
     '    """Multiply a contender rate by his per-replicate opponent shock when v2\n'
     '    correlation is on; identity otherwise (_CORR2_MULT stays None)."""\n'
     '    return rate if _CORR2_MULT is None else rate * _CORR2_MULT\n'
     '\n'
     '\n'
     'def _gamma_rate(rng, priors, node, cohort, vc, vm, k, own_rate=None, own_min=0.0):\n',
     'def _c2mul(rate):'),

    # ---- apply the shock in each node (after the rate draw; stacks with _corr_mu)
    (F_MC,
     '    rate = _gamma_rate(rng, vpriors, "reb", cohort, vc, vm, k,\n'
     '                       own_rate=d.get("prior_rate_reb"), own_min=d.get("prior_min") or 0.0)',
     '    rate = _gamma_rate(rng, vpriors, "reb", cohort, vc, vm, k,\n'
     '                       own_rate=d.get("prior_rate_reb"), own_min=d.get("prior_min") or 0.0)\n'
     '    rate = _c2mul(rate)',
     'own_min=d.get("prior_min") or 0.0)\n    rate = _c2mul(rate)'),

    (F_MC,
     '    rate = _gamma_rate(rng, vpriors, "ast_create", cohort, vc, vm, k,\n'
     '                       own_rate=d.get("prior_rate_ast_create"), own_min=d.get("prior_min") or 0.0)',
     '    rate = _gamma_rate(rng, vpriors, "ast_create", cohort, vc, vm, k,\n'
     '                       own_rate=d.get("prior_rate_ast_create"), own_min=d.get("prior_min") or 0.0)\n'
     '    rate = _c2mul(rate)',
     'prior_rate_ast_create"), own_min=d.get("prior_min") or 0.0)\n    rate = _c2mul(rate)'),

    (F_MC,
     '    rate = _gamma_rate(rng, vpriors, "usage", cohort, vc, vm, k,\n'
     '                       own_rate=d.get("prior_rate_usage"), own_min=d.get("prior_min") or 0.0)',
     '    rate = _gamma_rate(rng, vpriors, "usage", cohort, vc, vm, k,\n'
     '                       own_rate=d.get("prior_rate_usage"), own_min=d.get("prior_min") or 0.0)\n'
     '    rate = _c2mul(rate)',
     'prior_rate_usage"), own_min=d.get("prior_min") or 0.0)\n    rate = _c2mul(rate)'),

    # ---- _eff_matrix: draw the per-team shocks for this snapshot
    (F_MC,
     '    if stat == "reb" and REB_ENV_VAR and REB_ENV_VAR > 0:\n'
     '        v = float(REB_ENV_VAR)\n'
     '        _REB_ENV = rng.gamma(1.0 / v, v, size=k)\n',
     '    if stat == "reb" and REB_ENV_VAR and REB_ENV_VAR > 0:\n'
     '        v = float(REB_ENV_VAR)\n'
     '        _REB_ENV = rng.gamma(1.0 / v, v, size=k)\n'
     '    global _CORR2_MULT\n'
     '    _CORR2_MULT = None\n'
     '    team_shocks = None\n'
     '    if CORR2:\n'
     '        teams = set()\n'
     '        for _p in field:\n'
     '            _w = counts[(season, snap, _p)].get("opp_w")\n'
     '            if _w:\n'
     '                teams.update(_w)\n'
     '        team_shocks = C2.draw_team_shocks(rng, teams, _CORR2_SD or C2.DEFAULT_SD, k)\n',
     'team_shocks = C2.draw_team_shocks'),

    # ---- _eff_matrix: recentre availability and set the contender shock
    (F_MC,
     '        rf, rm = _draw_availability(pools, rc, tcut, k, rng)\n'
     '        if rf is None:\n'
     '            continue\n',
     '        rf, rm = _draw_availability(pools, rc, tcut, k, rng)\n'
     '        if rf is None:\n'
     '            continue\n'
     '        if AVAIL_HIER and _AVAIL_PRIOR is not None:\n'
     '            rf, rm = AH.recentre(rf, rm, rc, d, pid, season, _AVAIL_PRIOR)\n'
     '        _CORR2_MULT = C2.contender_mult(d.get("opp_w"), team_shocks, k) if team_shocks is not None else None\n',
     'AH.recentre(rf, rm, rc, d, pid, season, _AVAIL_PRIOR)'),

    # ---- load_all: fit prior lookup and shock SD (gated)
    (F_MC,
     '    reb_env_var = RENV.fit_env_var(counts, finals)\n',
     '    reb_env_var = RENV.fit_env_var(counts, finals)\n'
     '    global _AVAIL_PRIOR, _CORR2_SD\n'
     '    if AVAIL_HIER:\n'
     '        _AVAIL_PRIOR = AH.fit(conn, list(range(fit_lo, fit_hi + 1)))\n'
     '    if CORR2:\n'
     '        _CORR2_SD = C2.fit_sd()\n',
     '_AVAIL_PRIOR = AH.fit(conn'),

    # ---- load_all: attach opponent weights (gated)
    (F_MC,
     '    ctx = _load_context(conn, eval_season)\n'
     '    ftg = _load_ftg(conn, eval_season)\n',
     '    ctx = _load_context(conn, eval_season)\n'
     '    ftg = _load_ftg(conn, eval_season)\n'
     '    if CORR2:\n'
     '        C2.attach_weights(conn, counts, ctx, eval_season)\n',
     'C2.attach_weights(conn, counts, ctx, eval_season)'),

    # ---- scorecard.py
    (F_SCORE,
     '    p.add_argument("--own-prior", action="store_true", help="blend the rate prior mean toward the player\'s prior-season rate")\n',
     '    p.add_argument("--own-prior", action="store_true", help="blend the rate prior mean toward the player\'s prior-season rate")\n'
     '    p.add_argument("--avail-hier", action="store_true", help="hierarchical availability recentre (own history for the centre)")\n'
     '    p.add_argument("--corr2", action="store_true", help="v2 correlation: heterogeneous per-opponent shared shock")\n',
     '--avail-hier", action="store_true"'),

    (F_SCORE,
     '        try:\n            B = MC.load_all(conn, s, args.fit_lookback)\n',
     '        MC.AVAIL_HIER = bool(args.avail_hier); MC.CORR2 = bool(args.corr2)\n'
     '        try:\n            B = MC.load_all(conn, s, args.fit_lookback)\n',
     'MC.AVAIL_HIER = bool(args.avail_hier); MC.CORR2 = bool(args.corr2)'),

    (F_SCORE,
     'MC.OWN_PRIOR_K = None',
     'MC.OWN_PRIOR_K = None; MC.AVAIL_HIER = False; MC.CORR2 = False',
     'MC.AVAIL_HIER = False; MC.CORR2 = False'),

    (F_SCORE,
     '        if args.own_prior:\n            tag += " own_prior=on"\n',
     '        if args.own_prior:\n            tag += " own_prior=on"\n'
     '        if args.avail_hier:\n            tag += " avail_hier=on"\n'
     '        if args.corr2:\n            tag += " corr2=on"\n',
     'tag += " corr2=on"'),

    # ---- ablate.py (single-line log anchors, order-safe vs corr_wire)
    (F_ABL,
     '    p.add_argument("--own-prior", action="store_true", help="blend rate prior toward prior-season rate")\n',
     '    p.add_argument("--own-prior", action="store_true", help="blend rate prior toward prior-season rate")\n'
     '    p.add_argument("--avail-hier", action="store_true", help="hierarchical availability recentre")\n'
     '    p.add_argument("--corr2", action="store_true", help="v2 heterogeneous correlation")\n',
     '--avail-hier", action="store_true"'),

    (F_ABL,
     '            log.info("season %d: fit rolling %d-%d and separate", s, s - args.fit_lookback, s - 1)\n',
     '            log.info("season %d: fit rolling %d-%d and separate", s, s - args.fit_lookback, s - 1)\n'
     '            MC.AVAIL_HIER = bool(args.avail_hier); MC.CORR2 = bool(args.corr2)\n',
     'and separate", s, s - args.fit_lookback, s - 1)\n            MC.AVAIL_HIER'),

    (F_ABL,
     '        log.info("season %d: fit rolling %d-%d and ablate", s, s - args.fit_lookback, s - 1)\n',
     '        log.info("season %d: fit rolling %d-%d and ablate", s, s - args.fit_lookback, s - 1)\n'
     '        MC.AVAIL_HIER = bool(args.avail_hier); MC.CORR2 = bool(args.corr2)\n',
     'and ablate", s, s - args.fit_lookback, s - 1)\n        MC.AVAIL_HIER'),
]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Wire --avail-hier and --corr2.")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--root", default=".")
    args = ap.parse_args(argv)
    root = Path(args.root)

    plan, ok = [], True
    for path, anchor, repl, sentinel in EDITS:
        fp = root / path
        if not fp.exists():
            print(f"MISSING  {path}"); ok = False; continue
        text = fp.read_text()
        if sentinel in text:
            print(f"SKIP     {path}: already applied ({sentinel[:40]!r})"); continue
        n = text.count(anchor)
        if n != 1:
            print(f"DRIFT    {path}: anchor matched {n} times, expected 1 ({anchor[:50]!r})")
            ok = False; continue
        print(f"OK       {path}: 1 anchor -> will apply")
        plan.append(fp)

    if not ok:
        print("\nABORT: drift or missing files, nothing written."); return 1
    if not args.apply:
        print(f"\nDRY-RUN: {len(plan)} edit(s) ready. Re-run with --apply to write."); return 0

    for fp in {p for p in plan}:
        shutil.copy2(fp, fp.with_suffix(fp.suffix + ".bak"))
    for path, anchor, repl, sentinel in EDITS:
        fp = root / path
        if not fp.exists():
            continue
        text = fp.read_text()
        if sentinel in text or text.count(anchor) != 1:
            continue
        fp.write_text(text.replace(anchor, repl, 1))
    print(f"\nAPPLIED: {len(plan)} edit(s), .bak written per touched file.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
