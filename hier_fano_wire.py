"""Anchored idempotent patcher: wire the hierarchical per-game fano.

Edits five files with single-occurrence anchors. Dry-run by default (prints what
each edit would do and asserts every anchor matches exactly once); pass --apply to
write, taking a .bak of each touched file first. Idempotent: an edit already
carrying its sentinel is skipped, so re-running is safe and a double-apply is
refused. If any anchor does not match exactly once (drift), NOTHING is written and
the run aborts, so a partial apply is impossible.

The change is inert until --hier-fano is passed to the scorecard/ablate, which
sets MC.HIER_FANO; with it off the engine draws the current global fano, so the
v1 gate and every prior stat-leader baseline are bit-identical.

Prereq: fano_hier.py must already be in scripts/features/stat_leader/, and
stat_rate_counts_asof must be rebuilt after this patch so the own_fano_* columns
populate (rates.py migrates the columns in on the next build).

Run from repo root:
  uv run python3 hier_fano_wire.py            # dry-run
  uv run python3 hier_fano_wire.py --apply     # apply with .bak backups
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

F_RATES = "scripts/features/stat_leader/rates.py"
F_VOL = "scripts/features/stat_leader/volume.py"
F_MC = "scripts/modelling/stat_leader/mc.py"
F_SCORE = "scripts/modelling/stat_leader/scorecard.py"
F_ABL = "scripts/modelling/stat_leader/ablate.py"

# Each edit: (path, anchor, replacement, sentinel). sentinel present in the file
# means already applied. anchor must occur exactly once when not yet applied.
EDITS = [
    # ---- rates.py: own_fano_* columns, additive migration, per-game accumulators
    (F_RATES,
     '    "fta", "ftm", "reb", "potential_ast_asof", "ast",\n]',
     '    "fta", "ftm", "reb", "potential_ast_asof", "ast",\n'
     '    "own_fano_reb", "own_fano_usage",\n]',
     '"own_fano_reb", "own_fano_usage",'),

    (F_RATES,
     '        f"PRIMARY KEY (nba_api_id, season, snapshot_date))"\n'
     '    )\n'
     '    conn.commit()',
     '        f"PRIMARY KEY (nba_api_id, season, snapshot_date))"\n'
     '    )\n'
     '    have = {r[1] for r in conn.execute("PRAGMA table_info(stat_rate_counts_asof)")}\n'
     '    for _c in COUNT_COLS:\n'
     '        if _c not in have:\n'
     '            conn.execute(f"ALTER TABLE stat_rate_counts_asof ADD COLUMN {_c} REAL")\n'
     '    conn.commit()',
     'ALTER TABLE stat_rate_counts_asof ADD COLUMN'),

    (F_RATES,
     '        c = {k: 0.0 for k in COUNT_COLS}\n'
     '        for snap in grid:',
     '        c = {k: 0.0 for k in COUNT_COLS}\n'
     '        sr = srr = su = suu = 0.0\n'
     '        for snap in grid:',
     'sr = srr = su = suu = 0.0'),

    (F_RATES,
     '                    c["reb"] += g["rebounds"] or 0.0\n'
     '                    c["ast"] += g["assists"] or 0.0\n'
     '                gi += 1',
     '                    c["reb"] += g["rebounds"] or 0.0\n'
     '                    c["ast"] += g["assists"] or 0.0\n'
     '                    reb_g = g["rebounds"] or 0.0\n'
     '                    use_g = fga + FT_POSS_COEF * fta + tov\n'
     '                    sr += reb_g; srr += reb_g * reb_g\n'
     '                    su += use_g; suu += use_g * use_g\n'
     '                gi += 1',
     'use_g = fga + FT_POSS_COEF * fta + tov'),

    (F_RATES,
     '            c["potential_ast_asof"] = potast.get(pid, {}).get(snap, 0.0)\n',
     '            c["potential_ast_asof"] = potast.get(pid, {}).get(snap, 0.0)\n'
     '            n_g = c["gp_played_asof"]\n'
     '            if n_g > 0:\n'
     '                _mr = sr / n_g\n'
     '                c["own_fano_reb"] = max((srr / n_g - _mr * _mr) / _mr, 0.0) if _mr > 0 else 0.0\n'
     '                _mu = su / n_g\n'
     '                c["own_fano_usage"] = max((suu / n_g - _mu * _mu) / _mu, 0.0) if _mu > 0 else 0.0\n',
     'c["own_fano_reb"] = max((srr / n_g'),

    # ---- volume.py: import fano_hier, stash the hierarchy in fit_priors
    (F_VOL,
     '    from nodes import _load, _cohort, MIN_MPG, MIN_COHORT  # type: ignore\n',
     '    from nodes import _load, _cohort, MIN_MPG, MIN_COHORT  # type: ignore\n'
     '\n'
     'try:\n'
     '    from scripts.features.stat_leader import fano_hier as _FH\n'
     'except ImportError:  # pragma: no cover\n'
     '    import fano_hier as _FH  # type: ignore\n',
     'import fano_hier as _FH'),

    (F_VOL,
     '    priors["fano"] = _fit_fano(priors, counts, finals, pos, firstyr)\n'
     '    return priors',
     '    priors["fano"] = _fit_fano(priors, counts, finals, pos, firstyr)\n'
     '    priors["fano_hier"] = _FH.fit(counts, finals, pos, firstyr, priors["fano"], mpg_cuts)\n'
     '    return priors',
     'priors["fano_hier"] = _FH.fit('),

    # ---- mc.py: import fano_hier, HIER_FANO global, _fano helper, use in _rem_*
    (F_MC,
     '    from scripts.features.stat_leader import reb_env as RENV\n',
     '    from scripts.features.stat_leader import reb_env as RENV\n'
     '    from scripts.features.stat_leader import fano_hier as FH\n',
     'import fano_hier as FH'),

    (F_MC,
     '    import reb_env as RENV  # type: ignore\n',
     '    import reb_env as RENV  # type: ignore\n'
     '    import fano_hier as FH  # type: ignore\n',
     'import fano_hier as FH  # type: ignore'),

    (F_MC,
     'OWN_PRIOR_K = None\n',
     'OWN_PRIOR_K = None\n'
     '# Hierarchical per-game fano (see fano_hier.py). False disables (global fano,\n'
     '# identical engine); True redistributes dispersion by own-history -> mpg x\n'
     '# volume cohort -> league. Driver sets it from the --hier-fano flag.\n'
     'HIER_FANO = False\n',
     'HIER_FANO = False'),

    (F_MC,
     'def _gamma_rate(rng, priors, node, cohort, vc, vm, k, own_rate=None, own_min=0.0):\n',
     'def _fano(vpriors, node, d):\n'
     '    """Per-game fano for a contender: the global coverage-matched scalar, or,\n'
     '    when HIER_FANO is on, the hierarchical own->cohort->league redistribution.\n'
     '    Backs off to the global scalar whenever the hierarchy or the inputs are\n'
     '    absent (e.g. ast_create), so it is identical to the base engine off."""\n'
     '    base = vpriors["fano"].get(node, 1.0)\n'
     '    if not HIER_FANO:\n'
     '        return base\n'
     '    hier = vpriors.get("fano_hier")\n'
     '    if not hier:\n'
     '        return base\n'
     '    vc, vm = V._vol_count(d, node)\n'
     '    gp = d.get("gp_played_asof") or 0.0\n'
     '    if vc is None or vm <= 0 or gp <= 0:\n'
     '        return base\n'
     '    of = d.get(FH.OWN_FANO_COL[node]) if node in FH.OWN_FANO_COL else None\n'
     '    return FH.fano_for(hier, node, vc / vm, vm / gp, of, gp)\n'
     '\n'
     '\n'
     'def _gamma_rate(rng, priors, node, cohort, vc, vm, k, own_rate=None, own_min=0.0):\n',
     'def _fano(vpriors, node, d):'),

    (F_MC,
     '    return V._draw_count(rng, rate, rem_min, vpriors["fano"].get("reb", 1.0))',
     '    return V._draw_count(rng, rate, rem_min, _fano(vpriors, "reb", d))',
     '_fano(vpriors, "reb", d)'),

    (F_MC,
     '    pot = V._draw_count(rng, rate, rem_min, vpriors["fano"].get("ast_create", 1.0))',
     '    pot = V._draw_count(rng, rate, rem_min, _fano(vpriors, "ast_create", d))',
     '_fano(vpriors, "ast_create", d)'),

    (F_MC,
     '    used = V._draw_count(rng, rate, rem_min, vpriors["fano"].get("usage", 1.0)).astype(float)',
     '    used = V._draw_count(rng, rate, rem_min, _fano(vpriors, "usage", d)).astype(float)',
     '_fano(vpriors, "usage", d)'),

    # ---- scorecard.py: flag, set, reset, tag
    (F_SCORE,
     '    p.add_argument("--own-prior", action="store_true", help="blend the rate prior mean toward the player\'s prior-season rate")\n',
     '    p.add_argument("--own-prior", action="store_true", help="blend the rate prior mean toward the player\'s prior-season rate")\n'
     '    p.add_argument("--hier-fano", action="store_true", help="hierarchical per-game fano (own-history -> mpg x volume cohort -> league)")\n',
     '--hier-fano", action="store_true", help="hierarchical per-game fano'),

    (F_SCORE,
     '        MC.OWN_PRIOR_K = MC.V.REF_MIN if args.own_prior else None\n',
     '        MC.OWN_PRIOR_K = MC.V.REF_MIN if args.own_prior else None\n'
     '        MC.HIER_FANO = bool(args.hier_fano)\n',
     'MC.HIER_FANO = bool(args.hier_fano)'),

    (F_SCORE,
     '    MC.MPG_K = None; MC.GAMES_K = None; MC.REB_ENV_VAR = 0.0; MC.OWN_PRIOR_K = None\n',
     '    MC.MPG_K = None; MC.GAMES_K = None; MC.REB_ENV_VAR = 0.0; MC.OWN_PRIOR_K = None; MC.HIER_FANO = False\n',
     'MC.OWN_PRIOR_K = None; MC.HIER_FANO = False'),

    (F_SCORE,
     '        if args.own_prior:\n'
     '            tag += " own_prior=on"\n',
     '        if args.own_prior:\n'
     '            tag += " own_prior=on"\n'
     '        if args.hier_fano:\n'
     '            tag += " hier_fano=on"\n',
     'tag += " hier_fano=on"'),

    # ---- ablate.py: flag, set in both loops, use _fano in the freezable count
    (F_ABL,
     '    p.add_argument("--own-prior", action="store_true", help="blend rate prior toward prior-season rate")\n',
     '    p.add_argument("--own-prior", action="store_true", help="blend rate prior toward prior-season rate")\n'
     '    p.add_argument("--hier-fano", action="store_true", help="hierarchical per-game fano")\n',
     '--hier-fano", action="store_true", help="hierarchical per-game fano"'),

    (F_ABL,
     '            MC.OWN_PRIOR_K = MC.V.REF_MIN if args.own_prior else None\n',
     '            MC.OWN_PRIOR_K = MC.V.REF_MIN if args.own_prior else None\n'
     '            MC.HIER_FANO = bool(args.hier_fano)\n',
     'MC.HIER_FANO = bool(args.hier_fano)'),

    (F_ABL,
     '        log.info("season %d: fit rolling %d-%d and ablate", s, s - args.fit_lookback, s - 1)\n'
     '        B = MC.load_all(conn, s, args.fit_lookback)\n',
     '        log.info("season %d: fit rolling %d-%d and ablate", s, s - args.fit_lookback, s - 1)\n'
     '        B = MC.load_all(conn, s, args.fit_lookback)\n'
     '        MC.HIER_FANO = bool(args.hier_fano)\n',
     'MC.HIER_FANO = bool(args.hier_fano)\n        for st in stats:'),

    (F_ABL,
     '    vol = _count(rng, rate, rem_min, vpriors["fano"].get(volnode, 1.0), fz_vol)',
     '    vol = _count(rng, rate, rem_min, MC._fano(vpriors, volnode, d), fz_vol)',
     'MC._fano(vpriors, volnode, d)'),
]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Wire the hierarchical per-game fano.")
    ap.add_argument("--apply", action="store_true", help="write changes (default: dry-run)")
    ap.add_argument("--root", default=".", help="repo root (default: cwd)")
    args = ap.parse_args(argv)
    root = Path(args.root)

    plan = []          # (path, new_text) to write
    ok = True
    for path, anchor, repl, sentinel in EDITS:
        fp = root / path
        if not fp.exists():
            print(f"MISSING  {path}")
            ok = False
            continue
        text = fp.read_text()
        if sentinel in text:
            print(f"SKIP     {path}: already applied ({sentinel[:40]!r})")
            continue
        n = text.count(anchor)
        if n != 1:
            print(f"DRIFT    {path}: anchor matched {n} times, expected 1 ({anchor[:50]!r})")
            ok = False
            continue
        print(f"OK       {path}: 1 anchor -> will apply")
        plan.append((fp, text.replace(anchor, repl, 1)))

    if not ok:
        print("\nABORT: drift or missing files, nothing written.")
        return 1
    if not args.apply:
        print(f"\nDRY-RUN: {len(plan)} edit(s) ready. Re-run with --apply to write.")
        return 0

    # group by file so one .bak per file even with several edits
    by_file = {}
    for fp, _ in plan:
        by_file.setdefault(fp, None)
    for fp in by_file:
        shutil.copy2(fp, fp.with_suffix(fp.suffix + ".bak"))
    # apply edits in order, re-reading between edits to the same file
    for path, anchor, repl, sentinel in EDITS:
        fp = root / path
        if not fp.exists():
            continue
        text = fp.read_text()
        if sentinel in text:
            continue
        if text.count(anchor) != 1:
            continue
        fp.write_text(text.replace(anchor, repl, 1))
    print(f"\nAPPLIED: {len(plan)} edit(s), .bak written per touched file.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
