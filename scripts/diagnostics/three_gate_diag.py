"""Three-gate diagnostic for the stat-leader books. For each book (pts, reb, ast)
and each of an early / mid / late snapshot, it mirrors mc._eff_matrix line for line
(same draws, same seed) but records the per-player leg means instead of only the
eff draws, then reports the realised leader against the model's top pick and
attributes the ranking gap to one of the three gates:

  availability  E[remaining games]  = mean(rf * rem_team)
  minutes       E[minutes per game] = mean(rm)
  rate          E[remaining per-game stat] = E[rem_total] / E[remaining games]

The point is to see which gate lets a non-leader outrank the true leader in the
early-mid window, and whether the observed banked per-game rate (which feeds the
rate gate as a summed sufficient statistic) is the thing pulling it. Read-only.

  uv run python3 -m scripts.diagnostics.three_gate_diag --season 2024
  uv run python3 -m scripts.diagnostics.three_gate_diag --season 2024 --book pts --stages 0.15 0.45 0.8
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
from scripts.common.db import connect
from scripts.modelling.stat_leader import mc as MC


def _ship(B):
    MC.MPG_K = B["mpg_k"]
    MC.GAMES_K = B["games_k"]
    MC.OWN_PRIOR_K = MC.V.REF_MIN
    MC.AVAIL_HIER = True


def _names(conn, pids):
    if not pids:
        return {}
    q = ",".join("?" * len(pids))
    return {int(r[0]): r[1] for r in conn.execute(
        f"SELECT player_id, name FROM players WHERE player_id IN ({q})", list(pids))}


def _decompose(stat, season, snap, field, B, k):
    """Mirror of mc._eff_matrix, returning per-player leg means and the eff mean."""
    rng = np.random.default_rng(MC._snap_seed(season, stat, snap))
    branch = MC.BRANCH[stat]
    counts, ctx_snap = B["counts"], B["ctx"][snap]
    vpriors, npriors, pools, tcut = B["vpriors"], B["npriors"], B["pools"], B["tcut"]
    pos, firstyr = B["pos"], B["firstyr"]
    rows = {}
    for pid in field:
        d = counts[(season, snap, pid)]
        rc = ctx_snap[pid]
        gp = d.get("gp_played_asof") or 0.0
        mn = d.get("min_asof") or 0.0
        banked = MC.BANKED[stat](d)
        cohort = MC.N._cohort(pid, season, d, pos, firstyr, vpriors["mpg_cuts"])
        rf, rm = MC._draw_availability(pools, rc, tcut, k, rng)
        if rf is None:
            continue
        if MC.AVAIL_HIER and MC._AVAIL_PRIOR is not None:
            rf, rm = MC.AH.recentre(rf, rm, rc, d, pid, season, MC._AVAIL_PRIOR)
        banked_mpg = (mn / gp) if gp else 0.0
        rm = np.where(np.isnan(rm), banked_mpg, rm)
        if MC.GAMES_K is not None:
            tg_asof = rc["ftg"] - rc["rem_team"]
            avail = (gp / tg_asof) if tg_asof > 0 else 1.0
            rf = MC.MIN.shrink_frac(rf, avail, tg_asof, MC.GAMES_K)
        if MC.MPG_K is not None:
            rm = MC.MIN.shrink_mpg(rm, banked_mpg, mn, MC.MPG_K)
        rem_games = rf * rc["rem_team"]
        rem_min = np.maximum(rem_games * rm, 0.0)
        rem_total = branch(rng, d, cohort, vpriors, npriors, rem_min, k)
        season_total = banked + rem_total
        season_games = gp + rem_games
        denom = np.maximum(season_games, MC._qual(rc["ftg"]))
        eff = np.where(denom > 0, season_total / denom, 0.0)
        e_remg = float(rem_games.mean())
        rows[pid] = dict(
            banked=float(banked), gp=float(gp),
            bpg=float(banked / gp) if gp else 0.0,
            e_remg=e_remg, e_mpg=float(rm.mean()),
            fwd_pg=float(rem_total.mean() / max(e_remg, 1e-9)),
            e_eff=float(eff.mean()))
    return rows


def _stage_snaps(snaps, stages):
    n = len(snaps)
    return [(s, snaps[min(n - 1, max(0, int(round(s * (n - 1)))))]) for s in stages]


def _report_book(conn, stat, season, B, stages, k):
    book = MC.STAT_AWARD[stat]
    eff_real = MC.realised_eff(B["finals"], B["ftg"], season, stat)
    if not eff_real:
        print(f"\n[{book}] no realised leader, skipping"); return
    leader = int(max(eff_real, key=eff_real.get))
    snaps = sorted(B["ctx"].keys())
    print(f"\n{'='*78}\n{book}  season {season}  realised leader = {leader}\n{'='*78}")
    for frac, snap in _stage_snaps(snaps, stages):
        field = MC._field_at(B["counts"], B["ctx"][snap], season, snap, stat, MC.FIELD_N)
        if leader not in field:
            print(f"\n-- stage {frac:.2f} ({snap}): leader not yet in field --"); continue
        rows = _decompose(stat, season, snap, field, B, k)
        if leader not in rows:
            print(f"\n-- stage {frac:.2f} ({snap}): leader not scored --"); continue
        model_top = max(rows, key=lambda p: rows[p]["e_eff"])
        nm = _names(conn, list(rows))
        misrank = model_top != leader
        print(f"\n-- stage {frac:.2f} ({snap})  model top = "
              f"{nm.get(model_top, model_top)}{'  MISRANK' if misrank else '  (correct)'} --")
        hdr = f"  {'player':22s} {'bank/g':>7s} {'E[remG]':>8s} {'E[mpg]':>7s} {'fwd/g':>7s} {'E[eff]':>7s}"
        print(hdr)
        show = [leader] if not misrank else [leader, model_top]
        for pid in show:
            r = rows[pid]
            tag = "LEADER" if pid == leader else "modeltop"
            print(f"  {nm.get(pid, str(pid))[:22]:22s} {r['bpg']:7.2f} {r['e_remg']:8.1f} "
                  f"{r['e_mpg']:7.1f} {r['fwd_pg']:7.2f} {r['e_eff']:7.3f}  <{tag}>")
        if misrank:
            L, M = rows[leader], rows[model_top]
            gaps = {"availability(E[remG])": (M["e_remg"] - L["e_remg"], L["e_remg"]),
                    "minutes(E[mpg])": (M["e_mpg"] - L["e_mpg"], L["e_mpg"]),
                    "rate(fwd/g)": (M["fwd_pg"] - L["fwd_pg"], L["fwd_pg"]),
                    "observed(bank/g)": (M["bpg"] - L["bpg"], L["bpg"])}
            worst = max(gaps, key=lambda g: abs(gaps[g][0]) / (abs(gaps[g][1]) + 1e-9))
            print("   gate gap (modeltop - leader): " +
                  "  ".join(f"{g.split('(')[0]}={v[0]:+.2f}" for g, v in gaps.items()))
            print(f"   -> largest relative edge to modeltop via: {worst}")


def main(argv=None):
    p = argparse.ArgumentParser(description="Three-gate stat-leader decomposition.")
    p.add_argument("--db", default="data/awards.db")
    p.add_argument("--season", type=int, default=2024)
    p.add_argument("--book", default="all", choices=["all", "pts", "reb", "ast"])
    p.add_argument("--stages", type=float, nargs="+", default=[0.2, 0.5, 0.8])
    p.add_argument("--fit-lookback", type=int, default=10)
    p.add_argument("--k", type=int, default=MC.DEFAULT_K)
    args = p.parse_args(argv)

    conn = connect(args.db)
    books = ["pts", "reb", "ast"] if args.book == "all" else [args.book]
    MC.AVAIL_HIER = True
    B = MC.load_all(conn, args.season, args.fit_lookback)
    _ship(B)
    print(f"three-gate diagnostic  season {args.season}  k={args.k}  stages={args.stages}")
    for stat in books:
        _report_book(conn, stat, args.season, B, args.stages, args.k)
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
