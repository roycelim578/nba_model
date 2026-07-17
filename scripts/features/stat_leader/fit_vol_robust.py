"""Walk-forward fit of the volume robustification scale s for usage (pts), reb,
and potast (ast creation volume), swept across half-lives, reporting two families
of metric so we can judge s the way the market actually pays, not only on average
error.

  RMSE family: exposure-weighted forward error of the robust as-of rate against
  realised remaining rate. Non-circular (target is realised rate, never the
  leaderboard/P&L). Good for "is the correction real", weak for "does it matter".

  Rank family: among the contender pool, per snapshot, top-1 hit rate (model's
  rate-argmax == eventual full-season rate-leader) and the mean rank the model
  gives that eventual leader. This is scored the way a winner-take-all market
  pays. It is isolated to the rate gate (ranks by per-minute rate, which is what
  s touches); the true eff-level argmax is the separate three-gate diagnostic.

Both are broken out by season stage (early/mid/late), since the outlier
contamination s targets lives in the small-sample early window and a pooled
number dilutes it. Seasons 2024/2025 excluded (2024 stat OOS, 2025 sealed).

  uv run python3 -m scripts.features.stat_leader.fit_vol_robust
  uv run python3 -m scripts.features.stat_leader.fit_vol_robust --node usage --hl-min 500
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
from scripts.common.db import connect
from scripts.features.stat_leader import rates as R

GRID = [0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 10.0, None]
RANK_S = [None, 1.0, 0.5, 0.2]
MIN_GP = 15
MIN_REM_MIN = 300.0
CONTENDER_PCTL = 70.0
STAGES = {"early": (0.0, 1 / 3), "mid": (1 / 3, 2 / 3), "late": (2 / 3, 1.0 + 1e-9)}
STAGE_KEYS = ["all", "early", "mid", "late"]


def _count(g, node):
    if node == "reb":
        return g["rebounds"] or 0.0
    return (g["fga"] or 0.0) + R.FT_POSS_COEF * (g["fta"] or 0.0) + (g["turnovers"] or 0.0)


def _season_league_rate(logs, node):
    tc = tm = 0.0
    for games in logs.values():
        for g in games:
            m = g["minutes"] or 0.0
            if m > 0:
                tc += _count(g, node); tm += m
    return (tc / tm) if tm > 0 else 0.0


def _season_league_rate_potast(logs, potast):
    tc = tm = 0.0
    for pid, vals in potast.items():
        if not vals:
            continue
        last = vals[max(vals)]
        pmin = sum((g["minutes"] or 0.0) for g in logs.get(pid, []) if (g["minutes"] or 0) > 0)
        if pmin > 0:
            tc += last; tm += pmin
    return (tc / tm) if tm > 0 else 0.0


def _season_obs(logs, node, s, hl, mu0, snaps):
    """(robust as-of rate, realised remaining rate, remaining minutes, banked
    as-of rate, frac, pid, snapshot index, realised full-season rate)."""
    obs = []
    n_snap = len(snaps)
    for pid, games in logs.items():
        games = sorted(games, key=lambda x: x["game_date"])
        tot_c = sum(_count(g, node) for g in games if (g["minutes"] or 0) > 0)
        tot_m = sum((g["minutes"] or 0.0) for g in games if (g["minutes"] or 0) > 0)
        if tot_m <= 0:
            continue
        srate = tot_c / tot_m
        ref = R._RunRef(mu0, R.REF_SEED_MIN, hl)
        raw_c = rob_c = mn_c = 0.0
        gp = gi = 0
        for si, snap in enumerate(snaps):
            while gi < len(games) and games[gi]["game_date"] <= snap:
                g = games[gi]; m = g["minutes"] or 0.0
                if m > 0:
                    cnt = _count(g, node)
                    f = R._robust_factor(cnt, m, ref, s)
                    raw_c += cnt; rob_c += cnt * f; mn_c += m; gp += 1
                gi += 1
            if gp >= MIN_GP and mn_c > 0:
                rem_c = tot_c - raw_c; rem_m = tot_m - mn_c
                if rem_m >= MIN_REM_MIN:
                    frac = si / max(1, n_snap - 1)
                    obs.append((rob_c / mn_c, rem_c / rem_m, rem_m, raw_c / mn_c, frac, pid, si, srate))
    return obs


def _season_obs_potast(logs, potast, s, hl, mu0, snaps):
    obs = []
    n_snap = len(snaps)
    for pid, games in logs.items():
        games = sorted(games, key=lambda x: x["game_date"])
        tot_m = sum((g["minutes"] or 0.0) for g in games if (g["minutes"] or 0) > 0)
        vals = potast.get(pid, {})
        if tot_m <= 0 or not vals:
            continue
        tot_c = vals[max(vals)]
        srate = tot_c / tot_m
        ref = R._RunRef(mu0, R.REF_SEED_MIN, hl)
        rob_cum = raw_prev = min_prev = mn_c = 0.0
        gp = gi = 0
        for si, snap in enumerate(snaps):
            while gi < len(games) and games[gi]["game_date"] <= snap:
                m = games[gi]["minutes"] or 0.0
                if m > 0:
                    mn_c += m; gp += 1
                gi += 1
            raw_now = vals.get(snap, 0.0)
            win_min = mn_c - min_prev
            win_cnt = raw_now - raw_prev
            f = R._robust_factor(win_cnt, win_min, ref, s)
            rob_cum += win_cnt * f
            raw_prev = raw_now; min_prev = mn_c
            if gp >= MIN_GP and mn_c > 0:
                rem_c = tot_c - raw_now; rem_m = tot_m - mn_c
                if rem_m >= MIN_REM_MIN:
                    frac = si / max(1, n_snap - 1)
                    obs.append((rob_cum / mn_c, rem_c / rem_m, rem_m, raw_now / mn_c, frac, pid, si, srate))
    return obs


def _lohi(stage):
    return STAGES[stage] if stage in STAGES else (-1.0, 2.0)


def _fit_error(obs, thr, stage):
    sse = wsum = 0.0
    lo, hi = _lohi(stage)
    for o in obs:
        rob, rem, w, banked, frac = o[0], o[1], o[2], o[3], o[4]
        if banked < thr or not (lo <= frac < hi):
            continue
        sse += w * (rob - rem) ** 2; wsum += w
    return sse, wsum


def _rank_stats(obs, thr, stage):
    """Per snapshot among contenders: is the model's rate-argmax the eventual
    rate-leader, and what rank does the model give that leader. Returns
    (n_snaps, n_top1_hits, sum_leader_rank)."""
    lo, hi = _lohi(stage)
    groups = {}
    for o in obs:
        rob, banked, frac, pid, si, srate = o[0], o[3], o[4], o[5], o[6], o[7]
        if banked < thr or not (lo <= frac < hi):
            continue
        groups.setdefault(si, []).append((rob, pid, srate))
    n = hits = 0
    sumrank = 0
    for rows in groups.values():
        if len(rows) < 2:
            continue
        model_pid = max(rows, key=lambda x: x[0])[1]
        true_pid = max(rows, key=lambda x: x[2])[1]
        true_rob = [r[0] for r in rows if r[1] == true_pid][0]
        n += 1
        if model_pid == true_pid:
            hits += 1
        sumrank += 1 + sum(1 for r in rows if r[0] > true_rob)
    return n, hits, sumrank


def _fit_one(conn, node, seasons, hl):
    per = {}
    for season in seasons:
        logs = R._load_logs(conn, season)
        snaps = R._grid(conn, season)
        if not snaps:
            continue
        if node == "potast":
            potast = R._load_potast(conn, season)
            mu0 = _season_league_rate_potast(logs, potast)
            raw_obs = _season_obs_potast(logs, potast, None, hl, mu0, snaps)
        else:
            mu0 = _season_league_rate(logs, node)
            raw_obs = _season_obs(logs, node, None, hl, mu0, snaps)
        if not raw_obs:
            continue
        thr = float(np.percentile([o[3] for o in raw_obs], CONTENDER_PCTL))
        per[season] = {}
        for s in GRID:
            if s is None:
                obs = raw_obs
            elif node == "potast":
                obs = _season_obs_potast(logs, potast, s, hl, mu0, snaps)
            else:
                obs = _season_obs(logs, node, s, hl, mu0, snaps)
            per[season][s] = {
                "rmse": {st: _fit_error(obs, thr, st) for st in STAGE_KEYS},
                "rank": {st: _rank_stats(obs, thr, st) for st in STAGE_KEYS}}
    return per


def _report(node, hl, per):
    avail = list(per)
    if not avail:
        print(f"  node={node} hl={hl}: no seasons with data"); return
    print(f"\n{'-'*78}\nnode = {node}   half-life = {hl:.0f} min (~{hl/34:.1f} games)\n{'-'*78}")

    print("  RMSE gain by stage (pooled s* / gain vs off):")
    for st in ("early", "mid", "late"):
        best_s = best_e = None
        for s in GRID:
            sse = sum(per[q][s]["rmse"][st][0] for q in avail); w = sum(per[q][s]["rmse"][st][1] for q in avail)
            if w <= 0:
                continue
            e = sse / w
            if best_e is None or e < best_e:
                best_e, best_s = e, s
        if best_e is None:
            print(f"    {st:>5}: no observations"); continue
        iw = sum(per[q][None]["rmse"][st][1] for q in avail)
        ie = (sum(per[q][None]["rmse"][st][0] for q in avail) / iw) ** 0.5 if iw > 0 else float("nan")
        ss = "off" if best_s is None else f"{best_s:.2f}"
        gain = 100 * (ie - best_e ** 0.5) / ie if ie > 0 else float("nan")
        print(f"    {st:>5}: s*={ss:>4}  rmse@s*={best_e**0.5:.4f}  rmse@off={ie:.4f}  gain={gain:+.1f}%")

    print("  rank correctness among contenders  (top-1 hit% / mean rank of the true rate-leader):")
    print("      stage " + "".join(f"{('off' if s is None else f's={s:g}'):>17}" for s in RANK_S))
    for st in STAGE_KEYS:
        cells = []
        for s in RANK_S:
            n = sum(per[q][s]["rank"][st][0] for q in avail)
            h = sum(per[q][s]["rank"][st][1] for q in avail)
            sr = sum(per[q][s]["rank"][st][2] for q in avail)
            cells.append(f"{100*h/n:>5.0f}% /{sr/n:>5.2f}" if n > 0 else f"{'--':>12}")
        print(f"    {st:>6} " + "".join(f"{c:>17}" for c in cells))

    print("  pooled RMSE curve:")
    for s in GRID:
        sse = sum(per[q][s]["rmse"]["all"][0] for q in avail); w = sum(per[q][s]["rmse"]["all"][1] for q in avail)
        lab = "off" if s is None else f"{s:.2f}"
        print(f"     s={lab:>4}  rmse={(sse/w)**0.5:.4f}")


def main(argv=None):
    p = argparse.ArgumentParser(description="Walk-forward fit of the volume robustification scale s.")
    p.add_argument("--db", default="data/awards.db")
    p.add_argument("--node", default="all", choices=["all", "usage", "reb", "potast"])
    p.add_argument("--min-season", type=int, default=2008)
    p.add_argument("--max-season", type=int, default=2023)
    p.add_argument("--hl-min", type=float, nargs="+", default=[500.0])
    args = p.parse_args(argv)

    conn = connect(args.db)
    nodes = ["usage", "reb", "potast"] if args.node == "all" else [args.node]
    seasons = list(range(args.min_season, args.max_season + 1))
    print(f"walk-forward s fit  seasons {seasons[0]}-{seasons[-1]}  half-lives={args.hl_min}  "
          f"grid={['off' if g is None else g for g in GRID]}")
    for hl in args.hl_min:
        for node in nodes:
            per = _fit_one(conn, node, seasons, hl)
            _report(node, hl, per)
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
