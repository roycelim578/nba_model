"""Anchored idempotent patcher: pin the MC seed, wire --corr mu-sharpening.

Two things, both inert until asked for:

  SEED. _snap_seed used Python's salted hash(), so every process drew a different
  stream and A/B deltas floated on the seed. This replaces it with a deterministic
  zlib.crc32 of the same inputs, so runs are reproducible and the --corr delta is
  readable against the base run.

  CORR. Adds MC.CORR (default False) and a measured remaining-opponent nudge:
  when --corr is passed, load_all fits the opponent BETA and attaches each
  contender's schedule-softness z, and the rate draw is multiplied by
  _corr_mu = clip(1 + BETA * opp_z). With --corr off, _corr_mu is 1.0 and the
  engine is bit-identical, so the v1 gate and every prior baseline are unchanged.

Anchors do not depend on whether hier_fano_wire has been applied. Dry-run by
default; --apply writes a .bak per touched file. Idempotent and abort-on-drift as
before. Prereq: correlation.py in scripts/features/stat_leader/.

Run from repo root:
  uv run python3 corr_wire.py
  uv run python3 corr_wire.py --apply
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
    # ---- seed pin
    (F_MC,
     'import math\nimport sys\n',
     'import math\nimport sys\nimport zlib\n',
     'import zlib'),

    (F_MC,
     'def _snap_seed(season, stat, snap):\n'
     '    return hash((season, stat, snap)) & 0xFFFFFFFF',
     'def _snap_seed(season, stat, snap):\n'
     '    return zlib.crc32(f"{season}|{stat}|{snap}".encode()) & 0xFFFFFFFF',
     'zlib.crc32(f"{season}'),

    # ---- correlation import (both branches)
    (F_MC,
     '    from scripts.features.stat_leader import reb_env as RENV\n',
     '    from scripts.features.stat_leader import reb_env as RENV\n'
     '    from scripts.features.stat_leader import correlation as CORR_MOD\n',
     'import correlation as CORR_MOD\n'),

    (F_MC,
     '    import reb_env as RENV  # type: ignore\n',
     '    import reb_env as RENV  # type: ignore\n'
     '    import correlation as CORR_MOD  # type: ignore\n',
     'import correlation as CORR_MOD  # type: ignore'),

    # ---- CORR global
    (F_MC,
     'OWN_PRIOR_K = None\n',
     'OWN_PRIOR_K = None\n'
     '# Named-driver correlation (see correlation.py). False disables (no remaining-\n'
     '# opponent nudge, identical engine); True applies the measured mu-sharpening.\n'
     'CORR = False\n',
     'CORR = False'),

    # ---- _corr_mu helper (prepend to _gamma_rate; hier_fano may also prepend _fano
    #      here, both stack cleanly before _gamma_rate)
    (F_MC,
     'def _gamma_rate(rng, priors, node, cohort, vc, vm, k, own_rate=None, own_min=0.0):\n',
     'def _corr_mu(vpriors, node, d):\n'
     '    """Deterministic remaining-opponent nudge: measured BETA times the\n'
     '    standardised softness of the contender\'s remaining schedule. 1.0 when\n'
     '    correlation is off or the driver is unavailable, so it nests to base."""\n'
     '    if not CORR:\n'
     '        return 1.0\n'
     '    b = vpriors.get("corr_beta", {}).get(node)\n'
     '    z = d.get("opp_z")\n'
     '    if b is None or z is None:\n'
     '        return 1.0\n'
     '    return min(1.10, max(0.90, 1.0 + b * z))\n'
     '\n'
     '\n'
     'def _gamma_rate(rng, priors, node, cohort, vc, vm, k, own_rate=None, own_min=0.0):\n',
     'def _corr_mu(vpriors, node, d):'),

    # ---- apply the nudge right after each node's rate draw (before reb_env / count;
    #      anchored on the stable _gamma_rate call, hier_fano-independent)
    (F_MC,
     '    rate = _gamma_rate(rng, vpriors, "reb", cohort, vc, vm, k,\n'
     '                       own_rate=d.get("prior_rate_reb"), own_min=d.get("prior_min") or 0.0)',
     '    rate = _gamma_rate(rng, vpriors, "reb", cohort, vc, vm, k,\n'
     '                       own_rate=d.get("prior_rate_reb"), own_min=d.get("prior_min") or 0.0)\n'
     '    rate = rate * _corr_mu(vpriors, "reb", d)',
     '_corr_mu(vpriors, "reb", d)'),

    (F_MC,
     '    rate = _gamma_rate(rng, vpriors, "ast_create", cohort, vc, vm, k,\n'
     '                       own_rate=d.get("prior_rate_ast_create"), own_min=d.get("prior_min") or 0.0)',
     '    rate = _gamma_rate(rng, vpriors, "ast_create", cohort, vc, vm, k,\n'
     '                       own_rate=d.get("prior_rate_ast_create"), own_min=d.get("prior_min") or 0.0)\n'
     '    rate = rate * _corr_mu(vpriors, "ast_create", d)',
     '_corr_mu(vpriors, "ast_create", d)'),

    (F_MC,
     '    rate = _gamma_rate(rng, vpriors, "usage", cohort, vc, vm, k,\n'
     '                       own_rate=d.get("prior_rate_usage"), own_min=d.get("prior_min") or 0.0)',
     '    rate = _gamma_rate(rng, vpriors, "usage", cohort, vc, vm, k,\n'
     '                       own_rate=d.get("prior_rate_usage"), own_min=d.get("prior_min") or 0.0)\n'
     '    rate = rate * _corr_mu(vpriors, "usage", d)',
     '_corr_mu(vpriors, "usage", d)'),

    # ---- load_all: fit BETA and attach opp_z, both gated by CORR (zero cost off)
    (F_MC,
     '    reb_env_var = RENV.fit_env_var(counts, finals)\n',
     '    reb_env_var = RENV.fit_env_var(counts, finals)\n'
     '    if CORR:\n'
     '        vpriors["corr_beta"] = CORR_MOD.fit_beta(conn, list(range(fit_lo, fit_hi + 1)))\n',
     'vpriors["corr_beta"] = CORR_MOD.fit_beta'),

    (F_MC,
     '    ctx = _load_context(conn, eval_season)\n'
     '    ftg = _load_ftg(conn, eval_season)\n',
     '    ctx = _load_context(conn, eval_season)\n'
     '    ftg = _load_ftg(conn, eval_season)\n'
     '    if CORR:\n'
     '        CORR_MOD.attach_opp_z(conn, counts, ctx, eval_season)\n',
     'CORR_MOD.attach_opp_z(conn, counts, ctx, eval_season)'),

    # ---- scorecard.py
    (F_SCORE,
     '    p.add_argument("--own-prior", action="store_true", help="blend the rate prior mean toward the player\'s prior-season rate")\n',
     '    p.add_argument("--own-prior", action="store_true", help="blend the rate prior mean toward the player\'s prior-season rate")\n'
     '    p.add_argument("--corr", action="store_true", help="named-driver correlation: measured remaining-opponent mu-sharpening")\n',
     '--corr", action="store_true", help="named-driver correlation'),

    (F_SCORE,
     '        try:\n            B = MC.load_all(conn, s, args.fit_lookback)\n',
     '        MC.CORR = bool(args.corr)\n'
     '        try:\n            B = MC.load_all(conn, s, args.fit_lookback)\n',
     'MC.CORR = bool(args.corr)'),

    (F_SCORE,
     'MC.OWN_PRIOR_K = None',
     'MC.OWN_PRIOR_K = None; MC.CORR = False',
     'MC.CORR = False'),

    (F_SCORE,
     '        if args.own_prior:\n            tag += " own_prior=on"\n',
     '        if args.own_prior:\n            tag += " own_prior=on"\n'
     '        if args.corr:\n            tag += " corr=on"\n',
     'tag += " corr=on"'),

    # ---- ablate.py
    (F_ABL,
     '    p.add_argument("--own-prior", action="store_true", help="blend rate prior toward prior-season rate")\n',
     '    p.add_argument("--own-prior", action="store_true", help="blend rate prior toward prior-season rate")\n'
     '    p.add_argument("--corr", action="store_true", help="named-driver correlation: remaining-opponent mu-sharpening")\n',
     '--corr", action="store_true", help="named-driver correlation'),

    (F_ABL,
     '            log.info("season %d: fit rolling %d-%d and separate", s, s - args.fit_lookback, s - 1)\n'
     '            B = MC.load_all(conn, s, args.fit_lookback)\n',
     '            log.info("season %d: fit rolling %d-%d and separate", s, s - args.fit_lookback, s - 1)\n'
     '            MC.CORR = bool(args.corr)\n'
     '            B = MC.load_all(conn, s, args.fit_lookback)\n',
     'and separate", s, s - args.fit_lookback, s - 1)\n            MC.CORR = bool(args.corr)'),

    (F_ABL,
     '        log.info("season %d: fit rolling %d-%d and ablate", s, s - args.fit_lookback, s - 1)\n'
     '        B = MC.load_all(conn, s, args.fit_lookback)\n',
     '        log.info("season %d: fit rolling %d-%d and ablate", s, s - args.fit_lookback, s - 1)\n'
     '        MC.CORR = bool(args.corr)\n'
     '        B = MC.load_all(conn, s, args.fit_lookback)\n',
     'and ablate", s, s - args.fit_lookback, s - 1)\n        MC.CORR = bool(args.corr)'),
]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Pin the seed and wire --corr mu-sharpening.")
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
