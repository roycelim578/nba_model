#!/usr/bin/env python3
"""Anchored idempotent patch: wire the volume mu-overlay into STL (two-part) and BLK (direct).

Gated per book by OVERLAY_STL / OVERLAY_BLK env flags, defaulting OFF (byte-identical).
When on for a book, the volume-leg per-minute rate is scaled onto the ElasticNet driver
prediction v_star and widened by s_hat, using the fitted artefact for that eval-season:
    v_banked = banked_volume / gp            (per game)
    v_star, s_hat = volume_overlay.apply(art, drivers, mpg, position, v_banked)
    rate <- rate * (v_star / v_banked) + Normal(0, s_hat / mpg)   (clipped >= 0)
At v_star == v_banked (w*=0) the mean is unchanged and only the s_hat band is added.

STL overlays the two-part deflection volume (_rem_stl_twopart); BLK overlays the DIRECT
block volume (_rem_blk), since the two-part BLK lost on the scorecard and BLK ships direct.
Drivers are merged into counts in nodes._load (STL: cont3_std, dloose_std, dfga_fg3_std,
dpct_overall_std; BLK: pfd). Missing artefact or driver => apply falls back to v_banked,
so a gap degrades to the base model rather than erroring.

Requires the two-part branch patch already applied (anchors on _rem_stl_twopart and the
direct _rem_blk) and fitted artefacts under models/stat_leader/overlay/ before flipping on.

Run from repo root:
  uv run python3 patch_overlay.py            # dry run
  uv run python3 patch_overlay.py --apply     # write (.bak.overlay backup)
Idempotent, aborts on drift, no-ops on re-run.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

NODES = "scripts/features/stat_leader/nodes.py"
MC = "scripts/modelling/stat_leader/mc.py"

EDITS = [
    # 1. merge overlay drivers into counts (nodes._load), before the finals block
    (NODES,
     '    # final-season rate per (season, pid): realised end-of-season, the calibration label\n'
     '    finals = {}',
     '    for _k in counts:\n'
     '        for _c in ("cont3_std", "dloose_std", "dfga_fg3_std", "dpct_overall_std", "pfd"):\n'
     '            counts[_k].setdefault(_c, None)\n'
     '    for r in conn.execute(f"SELECT season, snapshot_date, nba_api_id, cont3_std, dloose_std "\n'
     '                          f"FROM stg_nba_hustle_asof WHERE season IN ({qs})", seasons):\n'
     '        d = counts.get((r["season"], r["snapshot_date"], r["nba_api_id"]))\n'
     '        if d is not None:\n'
     '            d["cont3_std"] = r["cont3_std"]; d["dloose_std"] = r["dloose_std"]\n'
     '    for r in conn.execute(f"SELECT season, snapshot_date, nba_api_id, dfga_fg3_std, dpct_overall_std "\n'
     '                          f"FROM stg_nba_defend_asof WHERE season IN ({qs})", seasons):\n'
     '        d = counts.get((r["season"], r["snapshot_date"], r["nba_api_id"]))\n'
     '        if d is not None:\n'
     '            d["dfga_fg3_std"] = r["dfga_fg3_std"]; d["dpct_overall_std"] = r["dpct_overall_std"]\n'
     '    for r in conn.execute(f"SELECT season, snapshot_date, nba_api_id, pfd "\n'
     '                          f"FROM stg_nba_player_asof_ext WHERE season IN ({qs})", seasons):\n'
     '        d = counts.get((r["season"], r["snapshot_date"], r["nba_api_id"]))\n'
     '        if d is not None:\n'
     '            d["pfd"] = r["pfd"]\n'
     '    # final-season rate per (season, pid): realised end-of-season, the calibration label\n'
     '    finals = {}'),
    # 2. import volume_overlay
    (MC,
     '    from scripts.features.stat_leader import avail_hier as AH\n'
     'except ImportError:  # pragma: no cover\n'
     '    from db import connect  # type: ignore\n'
     '    import nodes as N  # type: ignore\n'
     '    import volume as V  # type: ignore\n'
     '    import availability as A  # type: ignore\n'
     '    import minutes as MIN  # type: ignore\n'
     '    import avail_hier as AH  # type: ignore',
     '    from scripts.features.stat_leader import avail_hier as AH\n'
     '    from scripts.modelling.stat_leader import volume_overlay as VO\n'
     'except ImportError:  # pragma: no cover\n'
     '    from db import connect  # type: ignore\n'
     '    import nodes as N  # type: ignore\n'
     '    import volume as V  # type: ignore\n'
     '    import availability as A  # type: ignore\n'
     '    import minutes as MIN  # type: ignore\n'
     '    import avail_hier as AH  # type: ignore\n'
     '    import volume_overlay as VO  # type: ignore'),
    # 3. OVERLAY env flag + artefact cache
    (MC,
     'TWO_PART = {"stl": os.environ.get("TWO_PART_STL", "0") == "1",\n'
     '            "blk": os.environ.get("TWO_PART_BLK", "0") == "1"}',
     'TWO_PART = {"stl": os.environ.get("TWO_PART_STL", "0") == "1",\n'
     '            "blk": os.environ.get("TWO_PART_BLK", "0") == "1"}\n'
     '# Per-book volume mu-overlay (volume_overlay.py). Env-driven for ProcessPool workers.\n'
     '# Off => no overlay, byte-identical. On => blend the volume-leg mean toward the\n'
     '# ElasticNet driver prediction and widen by s_hat, from the per-eval-season artefact.\n'
     'OVERLAY = {"stl": os.environ.get("OVERLAY_STL", "0") == "1",\n'
     '           "blk": os.environ.get("OVERLAY_BLK", "0") == "1",\n'
     '           "ast": os.environ.get("OVERLAY_AST", "0") == "1"}\n'
     '_OVERLAY_ART = {}'),
    # 4. _overlay_rate helper after _beta_draw
    (MC,
     'def _beta_draw(rng, npriors, node, cohort, d, mk_col, at_col, k):\n'
     '    pa, pb = N.beta_posterior(npriors, node, cohort, d.get(mk_col) or 0.0, d.get(at_col) or 0.0)\n'
     '    return rng.beta(pa, pb, size=k) if (pa > 0 and pb > 0) else np.zeros(k)',
     'def _beta_draw(rng, npriors, node, cohort, d, mk_col, at_col, k):\n'
     '    pa, pb = N.beta_posterior(npriors, node, cohort, d.get(mk_col) or 0.0, d.get(at_col) or 0.0)\n'
     '    return rng.beta(pa, pb, size=k) if (pa > 0 and pb > 0) else np.zeros(k)\n'
     '\n'
     '\n'
     'def _overlay_rate(rng, book, d, cohort, rate, banked_cnt, banked_min, k):\n'
     '    """Scale the per-minute volume rate onto the overlay mean v_star and widen by\n'
     '    s_hat. No-op when the book\'s overlay is off, the artefact is absent, or the\n'
     '    per-game context is degenerate; apply() itself falls back to v_banked on a\n'
     '    missing driver, so the whole path degrades to the base rate."""\n'
     '    if not OVERLAY.get(book):\n'
     '        return rate\n'
     '    art = _OVERLAY_ART.get(book)\n'
     '    if art is None:\n'
     '        return rate\n'
     '    gp = d.get("gp_played_asof") or 0.0\n'
     '    if gp <= 0 or banked_min <= 0 or banked_cnt <= 0:\n'
     '        return rate\n'
     '    mpg = banked_min / gp\n'
     '    if mpg <= 0:\n'
     '        return rate\n'
     '    v_banked = banked_cnt / gp\n'
     '    drivers = {c: d.get(c) for (_t, c, _r) in art["drv"]}\n'
     '    v_star, s_hat = VO.apply(art, drivers, mpg, cohort[0], v_banked)\n'
     '    scaled = rate * (v_star / v_banked)\n'
     '    if s_hat and s_hat > 0:\n'
     '        scaled = scaled + rng.normal(0.0, s_hat / mpg, size=k)\n'
     '    return np.maximum(scaled, 0.0)'),
    # 5. load artefacts per eval-season inside _assemble
    (MC,
     '    global _AVAIL_PRIOR\n'
     '    if AVAIL_HIER:\n'
     '        _AVAIL_PRIOR = priors["avail_prior"]',
     '    global _AVAIL_PRIOR\n'
     '    if AVAIL_HIER:\n'
     '        _AVAIL_PRIOR = priors["avail_prior"]\n'
     '    global _OVERLAY_ART\n'
     '    _OVERLAY_ART = {}\n'
     '    for _bk in ("stl", "blk", "ast"):\n'
     '        if OVERLAY.get(_bk):\n'
     '            try:\n'
     '                _OVERLAY_ART[_bk] = VO.load(_bk, eval_season)\n'
     '            except Exception:\n'
     '                _OVERLAY_ART[_bk] = None'),
    # 6. STL hook: overlay the deflection volume rate
    (MC,
     '    rate = _gamma_rate(rng, vpriors, "defl", cohort, vc, vm, k,\n'
     '                       own_rate=d.get("prior_rate_defl"), own_min=d.get("prior_min") or 0.0)\n'
     '    defl_ct = V._draw_count(rng, rate, rem_min, vpriors["fano"].get("defl", 1.0))',
     '    rate = _gamma_rate(rng, vpriors, "defl", cohort, vc, vm, k,\n'
     '                       own_rate=d.get("prior_rate_defl"), own_min=d.get("prior_min") or 0.0)\n'
     '    rate = _overlay_rate(rng, "stl", d, cohort, rate, vc, vm, k)\n'
     '    defl_ct = V._draw_count(rng, rate, rem_min, vpriors["fano"].get("defl", 1.0))'),
    # 7. BLK hook: overlay the direct block volume rate
    (MC,
     '    rate = _gamma_rate(rng, vpriors, "blk", cohort, vc, vm, k,\n'
     '                       own_rate=d.get("prior_rate_blk"), own_min=d.get("prior_min") or 0.0)\n'
     '    return V._draw_count(rng, rate, rem_min, vpriors["fano"].get("blk", 1.0))',
     '    rate = _gamma_rate(rng, vpriors, "blk", cohort, vc, vm, k,\n'
     '                       own_rate=d.get("prior_rate_blk"), own_min=d.get("prior_min") or 0.0)\n'
     '    rate = _overlay_rate(rng, "blk", d, cohort, rate, vc, vm, k)\n'
     '    return V._draw_count(rng, rate, rem_min, vpriors["fano"].get("blk", 1.0))'),
]


def _classify(text, old, new):
    if new in text:
        return "done"
    n = text.count(old)
    if n == 1:
        return "apply"
    return "drift" if n == 0 else "ambiguous"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--root", default=".")
    args = ap.parse_args(argv)
    root = os.path.abspath(args.root)
    cache, plan, bad = {}, [], False
    for rel, old, new in EDITS:
        path = os.path.join(root, rel)
        if not os.path.exists(path):
            print(f"  MISSING FILE  {rel}"); bad = True; continue
        if path not in cache:
            cache[path] = open(path, encoding="utf-8").read()
        st = _classify(cache[path], old, new)
        tag = {"apply": "APPLY", "done": "skip (done)",
               "drift": "DRIFT (anchor missing)", "ambiguous": "DRIFT (not unique)"}[st]
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
        bak = f"{path}.bak.overlay"
        if not os.path.exists(bak):
            shutil.copy2(path, bak)
        open(path, "w", encoding="utf-8").write(new_text[path])
        print(f"  wrote  {os.path.relpath(path, root)}")
    print(f"\nApplied {len(todo)} edit(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
