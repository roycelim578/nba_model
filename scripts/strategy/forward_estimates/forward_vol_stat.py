"""Stat-leader forward-vol fit: reads the self-contained stat corpus and writes
models/forward_vol/forward_vol_stat.pkl via the parametrised fit_forward_vol.

SHIP SPEC (decided from the model-selection runs): jump ON and NFL excluded, i.e.
the high-g fit. The parametric jump term earned its place (it corrects a pervasive
under-coverage of the centre on stat series, not just the tails), and excluding
short tournaments (NFL at 17 games) removes a late-vol and early-half-life
inflation that does not apply to the 82-game NBA book. These are the defaults, so
a bare --fit reproduces the shipped model; --no-jump and --exclude-leagues ""
override.

_load_stat mirrors pm_corpus._load exactly but against corpus_market_stat /
corpus_price_daily_stat, so every downstream pm_corpus internal is reused
unchanged. The fit passes lowconf_cells='auto' (derive the low-confidence cells
from the stat coverage surface). With jump on the pkl carries jump_disabled=False
and ForwardVolModel assembles the jump term; forward_edge constructs the model
identically either way.

  uv run python3 -m scripts.strategy.forward_estimates.forward_vol_stat --fit
  uv run python3 -m scripts.strategy.forward_estimates.forward_vol_stat --fit --no-jump
  uv run python3 -m scripts.strategy.forward_estimates.forward_vol_stat --fit --exclude-leagues ""
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from scripts.common.db import connect
from scripts.strategy.forward_estimates.forward_vol import fit_forward_vol, ForwardVolModel
from scripts.strategy.forward_estimates.pm_corpus import _logit

DB_PATH = "data/awards.db"
OUT_PATH = "models/forward_vol/forward_vol_stat.pkl"
DEFAULT_EXCLUDE_LEAGUES = ("NFL",)  # high-g ship spec: short tournaments out


def _load_stat(conn, eps, exclude_leagues=None):
    """Mirror of pm_corpus._load against the stat corpus tables. exclude_leagues
    drops series whose event_slug LEAGUE prefix is in the set, so a high-g
    (NBA-representative) subset can be fit and validated by excluding the short
    tournaments (e.g. NFL at 17 games)."""
    excl = {l.upper() for l in (exclude_leagues or [])}
    raw = {}
    for r in conn.execute(
        "SELECT market_id, day, yes_price FROM corpus_price_daily_stat "
        "ORDER BY market_id, day"
    ):
        raw.setdefault(r["market_id"], []).append(float(r["yes_price"]))
    meta = {r["market_id"]: r["event_slug"] for r in conn.execute(
        "SELECT market_id, event_slug FROM corpus_market_stat")}
    out = {}
    for mid, prices in raw.items():
        if len(prices) < 12:
            continue
        slug = meta.get(mid, mid)
        if excl and str(slug).split("|")[0].upper() in excl:
            continue
        lo = [_logit(p, eps) for p in prices]
        dstep = np.array([lo[i] - lo[i - 1] for i in range(1, len(lo))])
        out[mid] = {"prices": prices, "lo": lo, "dstep": dstep, "event": slug}
    return out


def fit(db_path=DB_PATH, out_path=OUT_PATH, jump=True, no_progress=False,
        exclude_leagues=DEFAULT_EXCLUDE_LEAGUES):
    loader = (lambda conn, eps: _load_stat(conn, eps, exclude_leagues))
    return fit_forward_vol(db_path, out_path, loader=loader,
                           lowconf_cells="auto", jump=jump, no_progress=no_progress)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Stat-leader forward-vol fit (ship: jump on, NFL excluded)")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--no-jump", action="store_true",
                    help="drop the parametric jump term (default keeps it)")
    ap.add_argument("--exclude-leagues", default=None,
                    help='comma list of LEAGUE prefixes to drop; default NFL; "" for none')
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("--out", default=OUT_PATH)
    ap.add_argument("--no-progress", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    if args.exclude_leagues is None:
        excl = list(DEFAULT_EXCLUDE_LEAGUES)
    elif args.exclude_leagues.strip().upper() in ("", "NONE"):
        excl = None
    else:
        excl = [s.strip() for s in args.exclude_leagues.split(",")]
    if args.fit:
        p = fit(args.db, args.out, jump=not args.no_jump, no_progress=args.no_progress,
                exclude_leagues=excl)
        m = ForwardVolModel(p)
        print(f"jump_disabled={m.a.get('jump_disabled', False)} "
              f"lowconf_cells={m.a['lowconf_cells']}")
    if args.smoke:
        m = ForwardVolModel(args.out)
        rng = np.random.default_rng(0); hist = rng.normal(0, 0.15, 40)
        r = m.forward_move(14, 0.3, 0.6, hist, alpha=0.10)
        print(f"smoke: vol={r.point_vol:.3f} down={r.down_move:+.3f} up={r.up_move:+.3f} "
              f"lowconf={r.low_confidence}")
    if not (args.fit or args.smoke):
        ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
