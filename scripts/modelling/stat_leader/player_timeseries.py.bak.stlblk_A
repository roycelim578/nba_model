"""Stat-leader arm: per-player probability time-series (the pre-ship sanity check).

Runs the ship-config model (own_prior + avail_hier on) across seasons and dumps,
per snapshot, each contender's model P(lead) and P(top3), a leaderboard reference,
and the realised outcome, then renders multi-panel PNGs so the trajectories can be
graded against basketball intuition before the 2024 hold-out is unlocked.

For each (stat, season): one panel, x = season progress, y = probability. A line
per player who ever holds meaningful mass or finishes top-3; the realised leader
is drawn bold; a dashed leaderboard reference (softmax of banked per-game rate at
the scorecard's per-stat temperature, an approximation of the scorecard baseline)
is shown for the realised leader. Names resolved from the players table.

Always writes a CSV of the raw series; renders PNGs if matplotlib is present.
Read-only, dev window only; it never touches held-out seasons.

Run:
  caffeinate -i uv run python3 -m scripts.modelling.stat_leader.player_timeseries \
      --stat all --year-min 2014 --year-max 2023 --out out/viz
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import sys
from collections import defaultdict

import numpy as np

try:
    from scripts.common.db import connect
    from scripts.modelling.stat_leader import mc as MC
except ImportError:  # pragma: no cover
    from db import connect  # type: ignore
    import mc as MC  # type: ignore

log = logging.getLogger("stat_leader.player_timeseries")

STAT_FLOOR = {"pts": 1997, "reb": 1997, "ast": 2013}
LB_T = {"reb": 0.5, "pts": 1.5, "ast": 0.5}   # leaderboard softmax temperature per scorecard
MAX_LINES = 10


def _name_map(conn):
    try:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(players)")]
    except Exception:  # noqa: BLE001
        return {}
    idc = next((c for c in ("nba_api_id", "nba_id", "player_nba_id") if c in cols), None)
    nmc = next((c for c in ("full_name", "display_name", "player_name", "name") if c in cols), None)
    if not idc or not nmc:
        return {}
    return {r[0]: r[1] for r in conn.execute(f"SELECT {idc}, {nmc} FROM players WHERE {idc} IS NOT NULL")}


def _lead_top3(eff):
    """P(lead) and P(top3) per row of an eff matrix (len(field) x k)."""
    k = eff.shape[1]
    lead = np.bincount(np.argmax(eff, axis=0), minlength=eff.shape[0]) / k
    order = np.argsort(-eff, axis=0)[:3, :]           # top-3 row indices per column
    t3 = np.zeros(eff.shape[0])
    for r in range(3):
        t3 += np.bincount(order[r], minlength=eff.shape[0])
    return lead, t3 / k


def _banked_rate(d, stat):
    gp = d.get("gp_played_asof") or 0.0
    return (MC.BANKED[stat](d) / gp) if gp > 0 else 0.0


def _lb_plead(rates, t):
    r = np.asarray(rates, dtype=float)
    if len(r) == 0 or r.std() == 0:
        return np.ones(len(r)) / max(1, len(r))
    z = (r - r.mean()) / (r.std() + 1e-9)
    e = np.exp(z / t)
    return e / e.sum()


def collect(conn, stat, season, lookback, k):
    MC.OWN_PRIOR_K = MC.V.REF_MIN
    MC.AVAIL_HIER = True
    MC.HIER_FANO = False
    MC.CORR = False
    MC.CORR2 = False
    B = MC.load_all(conn, season, lookback)
    eff_real = MC.realised_eff(B["finals"], B["ftg"], season, stat)
    if not eff_real:
        return None
    ranked = sorted(eff_real, key=eff_real.get, reverse=True)
    leader = ranked[0]
    top3 = set(ranked[:3])
    snaps = sorted(B["ctx"].keys())
    n = len(snaps)
    series = defaultdict(lambda: {"frac": [], "lead": [], "top3": [], "lb": []})
    peak = defaultdict(float)
    for si, snap in enumerate(snaps):
        ctx_snap = B["ctx"].get(snap, {})
        field = MC._field_at(B["counts"], ctx_snap, season, snap, stat, MC.FIELD_N)
        if not field:
            continue
        eff = MC._eff_matrix(stat, season, snap, field, B["counts"], ctx_snap,
                             B["vpriors"], B["npriors"], B["pools"], B["tcut"],
                             B["pos"], B["firstyr"], k)
        lead, t3 = _lead_top3(eff)
        rates = [_banked_rate(B["counts"][(season, snap, p)], stat) for p in field]
        lb = _lb_plead(rates, LB_T[stat])
        frac = si / max(1, n - 1)
        for i, pid in enumerate(field):
            s = series[pid]
            s["frac"].append(frac); s["lead"].append(lead[i])
            s["top3"].append(t3[i]); s["lb"].append(lb[i])
            peak[pid] = max(peak[pid], lead[i])
    keep = set(top3) | {leader} | {p for p, v in peak.items() if v >= 0.10}
    keep = sorted(keep, key=lambda p: peak.get(p, 0), reverse=True)[:MAX_LINES]
    return {"series": {p: series[p] for p in keep if p in series},
            "leader": leader, "top3": top3, "n": n}


def render(figdata, names, stat, metric, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        log.warning("matplotlib unavailable (%s); wrote CSV only", e)
        return False
    years = sorted(figdata)
    ncol = min(3, len(years))
    nrow = math.ceil(len(years) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(6 * ncol, 4 * nrow), squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    for idx, yr in enumerate(years):
        ax = axes[idx // ncol][idx % ncol]
        ax.set_visible(True)
        fd = figdata[yr]
        for pid, s in fd["series"].items():
            nm = names.get(pid, f"id={pid}")
            bold = pid == fd["leader"]
            ax.plot(s["frac"], s[metric], lw=2.6 if bold else 1.2,
                    label=(nm[:18] + (" *" if bold else "")), zorder=3 if bold else 2)
        ld = fd["leader"]
        if ld in fd["series"]:
            ax.plot(fd["series"][ld]["frac"], fd["series"][ld]["lb"], "k--", lw=1.0,
                    alpha=0.6, label="leaderboard (leader)")
        ax.set_title(f"{stat.upper()} {yr}  leader: {names.get(fd['leader'], fd['leader'])[:20]}",
                     fontsize=9)
        ax.set_xlabel("season progress"); ax.set_ylabel(f"P({metric})")
        ax.set_ylim(-0.02, 1.02); ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return True


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Per-player probability time-series.")
    p.add_argument("--db", default="data/awards.db")
    p.add_argument("--stat", default="all", choices=["reb", "pts", "ast", "all"])
    p.add_argument("--year-min", type=int, default=2014)
    p.add_argument("--year-max", type=int, default=2023)
    p.add_argument("--fit-lookback", type=int, default=10)
    p.add_argument("--k", type=int, default=1000)
    p.add_argument("--out", default="out/viz")
    args = p.parse_args(argv)
    stats = ["reb", "pts", "ast"] if args.stat == "all" else [args.stat]
    os.makedirs(args.out, exist_ok=True)

    conn = connect(args.db)
    names = _name_map(conn)
    for stat in stats:
        figdata = {}
        csv_path = os.path.join(args.out, f"timeseries_{stat}.csv")
        with open(csv_path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["season", "frac", "pid", "name", "p_lead", "p_top3",
                        "p_lead_lb", "is_realised_leader", "is_realised_top3"])
            for yr in range(args.year_min, args.year_max + 1):
                if yr < STAT_FLOOR[stat]:
                    continue
                try:
                    r = collect(conn, stat, yr, args.fit_lookback, args.k)
                except Exception as e:  # noqa: BLE001
                    log.warning("%s %d skipped (%s)", stat, yr, e)
                    continue
                if not r:
                    continue
                figdata[yr] = r
                for pid, s in r["series"].items():
                    nm = names.get(pid, f"id={pid}")
                    for j in range(len(s["frac"])):
                        w.writerow([yr, f"{s['frac'][j]:.3f}", pid, nm,
                                    f"{s['lead'][j]:.4f}", f"{s['top3'][j]:.4f}",
                                    f"{s['lb'][j]:.4f}",
                                    int(pid == r["leader"]), int(pid in r["top3"])])
        log.info("wrote %s (%d seasons)", csv_path, len(figdata))
        for metric in ("lead", "top3"):
            png = os.path.join(args.out, f"timeseries_{stat}_{metric}.png")
            if render(figdata, names, stat, metric, png):
                log.info("wrote %s", png)
    MC.OWN_PRIOR_K = None
    MC.AVAIL_HIER = False
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
