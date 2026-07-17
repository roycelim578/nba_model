"""Mean-versus-variance decomposition for the stat-leader P(lead) failures.

Book-agnostic. Splits the P(lead) error into mean-bias and under-dispersion by
within-season stage, using the full per-contender simulated eff distribution
mc._eff_matrix exposes. Requires the projection qualifier patch first.

This version fixes the selection confound in the dispersion residual. The
residual z = (realised_final_eff - proj_mu) / proj_sd is reported over four
cohorts so the sign is not a projection-selection artefact:

  - field      : all field contenders with proj_sd>0 (neutral calibration base)
  - proj8      : top-8 by proj_mu (the old, projection-selected view)
  - realtop3   : the realised top-3 by final eff (outcome-selected)
  - WINNER vs FRONTRUNNER : the decisive within-race paired contrast. WINNER is
    the eventual champion's z; FRONTRUNNER is the model's earliest-snapshot
    argmax (when it is not the champion). If winners run z>0 (under-projected)
    and faded front-runners run z<0 (over-projected), that is the heterogeneous
    mean error a per-player leading indicator could fix, and it cannot be a
    selection artefact because it is a paired contrast inside each race.

Also: the argmax mean-lever recovery (how much a perfect mean would fix) and the
overconfidence gate metric (high-confidence hit-rate, close vs wide).

Run:
  caffeinate -i uv run python3 -m scripts.modelling.stat_leader.decomp \
      --stat all --seasons 2018-2023
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

try:
    from scripts.common.db import connect
    from scripts.modelling.stat_leader import mc as MC
except ImportError:  # pragma: no cover
    from db import connect  # type: ignore
    import mc as MC  # type: ignore

HI = 0.70
CLOSE = 0.05
STAT_FLOOR = {"pts": 1997, "reb": 1997, "ast": 2013, "stl": 1997, "blk": 1997}
SEALED = {"pts": {2025}, "reb": {2025}, "ast": {2025},
          "stl": {2024, 2025}, "blk": {2024, 2025}}


def _p_lead(eff):
    return np.bincount(np.argmax(eff, axis=0), minlength=eff.shape[0]) / eff.shape[1]


def _stage(frac):
    return "early" if frac > 2.0 / 3 else ("mid" if frac > 1.0 / 3 else "late")


def _acc(d, stg, val):
    d.setdefault(stg, []).append(val)


def decompose(conn, stat, seasons):
    Z = {c: {} for c in ("field", "proj8", "realtop3", "winner", "frontrunner")}
    cover = {}
    wrong = mean_fix = var_limited = 0
    hi_close_hit = hi_close_n = hi_wide_hit = hi_wide_n = 0
    max_plead = 0.0

    for season in seasons:
        MC.OWN_PRIOR_K = MC.V.REF_MIN
        MC.AVAIL_HIER = True
        B = MC.load_all(conn, season, 10)
        MC.MPG_K = B["mpg_k"]
        MC.GAMES_K = B["games_k"]
        realised = MC.realised_eff(B["finals"], B["ftg"], season, stat)
        if not realised:
            continue
        true_leader = max(realised, key=realised.get)
        realtop3 = {pid for pid, _ in sorted(realised.items(),
                    key=lambda kv: kv[1], reverse=True)[:3]}
        snaps = sorted({s for (y, s, p) in B["counts"] if y == season})

        frontrunner = None
        for snap in snaps:
            cs = B["ctx"].get(snap, {})
            if not cs:
                continue
            field = MC._field_at(B["counts"], cs, season, snap, stat, MC.FIELD_N)
            if len(field) < 2:
                continue
            eff = MC._eff_matrix(stat, season, snap, field, B["counts"], cs,
                                 B["vpriors"], B["npriors"], B["pools"], B["tcut"],
                                 B["pos"], B["firstyr"], MC.DEFAULT_K)
            mu, sd = eff.mean(1), eff.std(1)
            pl = _p_lead(eff)
            rvec = np.array([realised.get(pid, 0.0) for pid in field])
            frac = float(np.mean([cs[p]["rem_team"] / cs[p]["ftg"]
                                  for p in field if cs[p].get("ftg")]))
            stg = _stage(frac)

            if frontrunner is None:
                frontrunner = field[int(np.argmax(pl))]

            def z_of(pid):
                if pid in field:
                    j = field.index(pid)
                    if sd[j] > 0:
                        return (rvec[j] - mu[j]) / sd[j]
                return None

            for j in range(len(field)):
                if sd[j] > 0:
                    zz = (rvec[j] - mu[j]) / sd[j]
                    _acc(Z["field"], stg, zz)
                    _acc(cover, stg, 1 if abs(zz) < 1.0 else 0)
            for j in np.argsort(-mu)[:8]:
                if sd[j] > 0:
                    _acc(Z["proj8"], stg, (rvec[j] - mu[j]) / sd[j])
            for pid in realtop3:
                zz = z_of(pid)
                if zz is not None:
                    _acc(Z["realtop3"], stg, zz)
            zw = z_of(true_leader)
            if zw is not None:
                _acc(Z["winner"], stg, zw)
            if frontrunner != true_leader:
                zf = z_of(frontrunner)
                if zf is not None:
                    _acc(Z["frontrunner"], stg, zf)

            model_leader = field[int(np.argmax(pl))]
            oracle_leader = field[int(np.argmax(_p_lead(eff - mu[:, None] + rvec[:, None])))]
            if model_leader != true_leader:
                wrong += 1
                mean_fix += (oracle_leader == true_leader)
                var_limited += (oracle_leader != true_leader)

            mp = float(pl.max())
            max_plead = max(max_plead, mp)
            if mp > HI:
                sr = np.sort(rvec)[::-1]
                margin = (sr[0] - sr[1]) / sr[0] if len(sr) > 1 and sr[0] > 0 else 1.0
                hit = int(model_leader == true_leader)
                if margin < CLOSE:
                    hi_close_n += 1; hi_close_hit += hit
                else:
                    hi_wide_n += 1; hi_wide_hit += hit

    print("\n" + "=" * 66)
    print(f"stat={stat}  seasons={seasons[0]}-{seasons[-1]}")
    print("-" * 66)
    print("dispersion residual z=(realised-proj_mu)/proj_sd  (honest: mean 0, sd 1)")
    for c in ("field", "proj8", "realtop3"):
        parts = []
        for s in ("early", "mid", "late"):
            v = Z[c].get(s, [])
            if v:
                parts.append(f"{s} {np.mean(v):+.2f}/{np.std(v):.2f}")
        if parts:
            print(f"  {c:<9} " + "   ".join(parts) + "   (mean/sd)")
    print("within-race mean contrast (the decisive, selection-free view):")
    for c in ("winner", "frontrunner"):
        parts = []
        for s in ("early", "mid", "late"):
            v = Z[c].get(s, [])
            if v:
                parts.append(f"{s} {np.mean(v):+.2f} (n={len(v)})")
        if parts:
            print(f"  {c:<11} " + "   ".join(parts))
    print("coverage68 by stage (honest ~0.68): " +
          "  ".join(f"{s} {np.mean(cover.get(s, [0])):.2f}" for s in ("early", "mid", "late")))
    print("argmax error decomposition:")
    if wrong:
        print(f"  wrong {wrong}   mean-fixable {mean_fix/wrong:.1%} ({mean_fix})   "
              f"variance-limited {var_limited/wrong:.1%} ({var_limited})")
    line = f"overconfidence gate (p_lead>{HI}): book max {max_plead:.3f}"
    if hi_close_n:
        line += f"   close hit {hi_close_hit/hi_close_n:.1%} ({hi_close_hit}/{hi_close_n})"
    if hi_wide_n:
        line += f"   wide hit {hi_wide_hit/hi_wide_n:.1%} ({hi_wide_hit}/{hi_wide_n})"
    print(line)


def _parse_seasons(s):
    if "-" in s:
        lo, hi = s.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in s.split(",")]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/awards.db")
    ap.add_argument("--stat", default="all",
                    choices=["reb", "pts", "ast", "stl", "blk", "pra", "all"])
    ap.add_argument("--seasons", default="2016-2023")
    ap.add_argument("--allow-sealed", action="store_true")
    a = ap.parse_args(argv)

    stats = {"all": ["pts", "reb", "ast", "stl", "blk"],
             "pra": ["pts", "reb", "ast"]}.get(a.stat, [a.stat])
    seasons = _parse_seasons(a.seasons)

    conn = connect(a.db)
    for stat in stats:
        yrs = [y for y in seasons if y >= STAT_FLOOR[stat]]
        bad = [y for y in yrs if y in SEALED[stat]]
        if bad and not a.allow_sealed:
            print(f"REFUSE stat={stat}: sealed {bad}. Drop them or --allow-sealed.")
            continue
        if not yrs:
            continue
        try:
            decompose(conn, stat, yrs)
        except Exception as e:
            print(f"stat={stat}: ERROR {e!r}")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
