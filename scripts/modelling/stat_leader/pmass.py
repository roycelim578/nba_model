"""Stat-leader arm: P(lead) mass-vs-rank diagnostic and leader-tracking plot.

Adjudicates the early-mid under-confidence between two hypotheses:
  correlation   the leaked probability mass sits on genuine near-contenders
                (banked ranks 2-5); the field is jointly over-dispersed and the
                fix is cross-contender correlation.
  contamination the leaked mass sits on ranks 6-30 who never win; the field is
                dirty and the fix is a tighter candidate filter.

For every snapshot it buckets each contender by banked rank (1, 2-5, 6-10,
11-30) and sums the model's P(lead) within each bucket, then compares that mass
against how often the realised leader actually came from each bucket, split by
season phase. It also tracks the eventual leader's predicted P(lead) against
season fraction, the direct picture of under-confidence. Runs on raw model
output (shrinks/env disabled), no retrain.

Run:
  uv run python3 -m scripts.modelling.stat_leader.pmass --stat all --eval-min 2008 --eval-max 2023
"""

from __future__ import annotations

import argparse
import logging
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

log = logging.getLogger("stat_leader.pmass")

STAT_FLOOR = {"pts": 1997, "reb": 1997, "ast": 2013}
PHASES = ("early", "mid", "late")
RANK_BUCKETS = (("rank1", 1, 1), ("rank2_5", 2, 5), ("rank6_10", 6, 10), ("rank11_30", 11, 30))


def _phase(i, n):
    if n <= 1:
        return "mid"
    f = i / (n - 1)
    return "early" if f < 1 / 3 else ("mid" if f < 2 / 3 else "late")


def _bucket(rank):
    if rank is None:
        return "out"
    for name, lo, hi in RANK_BUCKETS:
        if lo <= rank <= hi:
            return name
    return "out"


def collect(B, season, stat, k, field_n):
    er = MC.realised_eff(B["finals"], B["ftg"], season, stat)
    if not er:
        return []
    leader = max(er, key=er.get)
    snaps = sorted(B["ctx"].keys())
    rows = []
    for si, snap in enumerate(snaps):
        field, p_lead, _ = MC.snapshot_probs(
            stat, season, snap, B["counts"], B["ctx"], B["vpriors"], B["npriors"],
            B["pools"], B["tcut"], B["pos"], B["firstyr"], k, field_n)
        if not field:
            continue
        banked = []
        for pid in field:
            d = B["counts"].get((season, snap, pid), {})
            gp = d.get("gp_played_asof") or 0.0
            banked.append((MC.BANKED[stat](d) / gp) if gp else 0.0)
        order = np.argsort(-np.asarray(banked))
        rank_of = {field[order[j]]: j + 1 for j in range(len(field))}
        mass = defaultdict(float)
        for i, pid in enumerate(field):
            mass[_bucket(rank_of[pid])] += float(p_lead[i])
        lead_rank = rank_of.get(leader)
        lead_p = float(p_lead[field.index(leader)]) if leader in field else 0.0
        rows.append({"season": season, "phase": _phase(si, len(snaps)),
                     "frac": si / max(len(snaps) - 1, 1), "mass": dict(mass),
                     "lead_rank": lead_rank, "lead_p": lead_p,
                     "lead_bucket": _bucket(lead_rank)})
    return rows


def report(stat, rows):
    print("\n" + "=" * 96)
    print(f"stat={stat}  snapshots={len(rows)}  (model P(lead) mass vs realised-leader share, by banked rank)")
    print("  contamination signature: rank11_30 carries mass but ~0 real; correlation signature:")
    print("  ranks 6-30 near-empty, rank1 mass << its real rate, excess in rank2_5.")
    hdr = f"  {'phase':>6} {'n':>5} |"
    for name, _, _ in RANK_BUCKETS:
        hdr += f" {name:>10} mass/real |"
    print(hdr + f" {'leadP':>6} {'leadRank':>8}")
    for ph in PHASES:
        sub = [r for r in rows if r["phase"] == ph]
        if not sub:
            continue
        line = f"  {ph:>6} {len(sub):>5} |"
        for name, _, _ in RANK_BUCKETS:
            m = np.mean([r["mass"].get(name, 0.0) for r in sub])
            real = np.mean([1.0 if r["lead_bucket"] == name else 0.0 for r in sub])
            line += f"  {m:>5.2f}/{real:<5.2f}  |"
        leadp = np.mean([r["lead_p"] for r in sub])
        lr = [r["lead_rank"] for r in sub if r["lead_rank"]]
        line += f" {leadp:>6.2f} {np.mean(lr) if lr else float('nan'):>8.1f}"
        print(line)
    out = np.mean([1.0 if r["lead_bucket"] == "out" else 0.0 for r in rows])
    print(f"  realised leader out-of-field rate: {out:.3f}")
    print("=" * 96)


def plot(stat, rows, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib unavailable; skipping plot"); return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fr = np.array([r["frac"] for r in rows]); lp = np.array([r["lead_p"] for r in rows])
    bins = np.linspace(0, 1, 11); idx = np.clip(np.digitize(fr, bins) - 1, 0, 9)
    xs, ys = [], []
    for bi in range(10):
        m = idx == bi
        if m.sum():
            xs.append((bins[bi] + bins[bi + 1]) / 2); ys.append(float(lp[m].mean()))
    ax1.plot(xs, ys, "o-", label="eventual leader")
    ax1.axhline(1.0, ls="--", c="grey", lw=0.8)
    ax1.set_xlabel("season fraction"); ax1.set_ylabel("mean predicted P(lead)")
    ax1.set_title(f"{stat}: eventual-leader under-confidence"); ax1.set_ylim(0, 1.05); ax1.legend()
    width = 0.2
    for j, (name, _, _) in enumerate(RANK_BUCKETS):
        mvals = [np.mean([r["mass"].get(name, 0.0) for r in rows if r["phase"] == ph]) for ph in PHASES]
        ax2.bar(np.arange(len(PHASES)) + j * width, mvals, width, label=name)
    ax2.set_xticks(np.arange(len(PHASES)) + 1.5 * width); ax2.set_xticklabels(PHASES)
    ax2.set_ylabel("mean model P(lead) mass"); ax2.set_title(f"{stat}: where the mass sits"); ax2.legend()
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
    log.info("wrote %s", path)


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Stat-leader P(lead) mass-vs-rank diagnostic.")
    p.add_argument("--db", default="data/awards.db")
    p.add_argument("--stat", default="all", choices=["reb", "pts", "ast", "all"])
    p.add_argument("--eval-min", type=int, default=2008)
    p.add_argument("--eval-max", type=int, default=2023)
    p.add_argument("--fit-lookback", type=int, default=10)
    p.add_argument("--k", type=int, default=MC.DEFAULT_K)
    p.add_argument("--field-n", type=int, default=MC.FIELD_N)
    p.add_argument("--outdir", default="out/stat_leader")
    args = p.parse_args(argv)

    stats = ["reb", "pts", "ast"] if args.stat == "all" else [args.stat]
    try:
        from scripts.common.config import assert_not_sealed
    except ImportError:
        from config import assert_not_sealed  # type: ignore
    seasons = list(range(args.eval_min, args.eval_max + 1))
    for st in stats:
        for s in seasons:
            assert_not_sealed(MC.STAT_AWARD[st], s)

    pooled = {st: [] for st in stats}
    conn = connect(args.db)
    for s in seasons:
        active = [st for st in stats if s >= STAT_FLOOR[st]]
        if not active:
            continue
        try:
            B = MC.load_all(conn, s, args.fit_lookback)
        except Exception as e:
            log.warning("season %d skipped (%s)", s, e); continue
        MC.MPG_K = None; MC.GAMES_K = None; MC.REB_ENV_VAR = 0.0; MC.OWN_PRIOR_K = None
        log.info("season %d", s)
        for st in active:
            pooled[st].extend(collect(B, s, st, args.k, args.field_n))
    conn.close()

    os.makedirs(args.outdir, exist_ok=True)
    for st in stats:
        if not pooled[st]:
            print(f"\nstat={st}: no rows"); continue
        report(st, pooled[st])
        plot(st, pooled[st], os.path.join(args.outdir, f"pmass_{st}.png"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
