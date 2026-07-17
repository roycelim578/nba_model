"""Cross-arm dependence analysis: voter arm versus stat PRA arm on 2024.

Answers the go/no-go for a shared-bankroll portfolio sizer. If the two arms' daily
mark-to-market returns are weakly or negatively correlated, and their P&L is earned
at staggered points in the season, then letting both size against one shared pool
gains leverage without a proportionate rise in joint drawdown. If they move
together, keep isolated sleeves.

It reconstructs each book's daily equity from the ledger MTM series (ledger._equity,
via risk_metrics.pooled_curve), NOT the flat static-budget equity_curve CSV, then
pools by arm, aligns the two arms on a common date axis, and reports Pearson and
Spearman correlation, ordinary-least-squares beta both directions, rolling
correlation, joint-drawdown overlap, and a cumulative-P&L-versus-season-fraction
curve per arm. It runs two weight regimes: equal within each arm, and the
shrunk-BSS split (voter via book_weighting, stat via stat_book_weighting). The bss
regime doubles as the bss-vs-equal strategy read on 2024 PRA, so the pooled Sharpe,
drawdown and P&L are printed for both regimes.

2024 is the burnt voter dev season and the burnt stat PRA dev season, so both arms
are already priced here and nothing sealed is spent. STL and BLK have no 2024 prices
and are out of scope; the arm-level read is voter versus PRA. British English.

  USE_REGION=1 REGION_CONFIRM=2 REGION_HYST=1.0 REGION_BAND_FLOOR=0.02 ASYM_TRIM=1 \
  caffeinate -i uv run python3 -m scripts.backtest.engine.arm_correlation \
      --season 2024 --bankroll-per-arm 3000 --out out/arm_corr_2024
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

try:
    from scripts.backtest.engine.backtest_singlepass import run_singlepass
    from scripts.common import risk_metrics as RM
except ImportError:  # pragma: no cover
    from backtest_singlepass import run_singlepass  # type: ignore
    import risk_metrics as RM  # type: ignore

VOTER = ("MVP", "DPOY", "ROTY")
STAT = ("PTS", "REB", "AST")


def _run_arm(season, awards, budgets):
    """Run one arm and return {award: ledger}. budgets None means equal split at the
    per-book default; otherwise a dict passed through ALLOC_BUDGETS_JSON."""
    saved = os.environ.get("ALLOC_BUDGETS_JSON")
    if budgets is None:
        os.environ.pop("ALLOC_BUDGETS_JSON", None)
        per_book = 1000.0
    else:
        os.environ["ALLOC_BUDGETS_JSON"] = json.dumps(budgets)
        per_book = 1000.0
    try:
        results, _ = run_singlepass(season, awards=tuple(awards),
                                    budget_per_book=per_book, compound=False)
    finally:
        if saved is None:
            os.environ.pop("ALLOC_BUDGETS_JSON", None)
        else:
            os.environ["ALLOC_BUDGETS_JSON"] = saved
    return {aw: results[aw]["ledger"] for aw in awards}


def _fill_arm(ledgers, master):
    """Sum an arm's book MTM equity onto the master date axis, forward-filling each
    book from its first recorded equity. Mirrors risk_metrics.pooled_curve but on a
    shared axis so the two arms line up."""
    total = np.zeros(len(master), float)
    for l in ledgers:
        if not getattr(l, "_equity", None):
            continue
        d = dict(zip(l._equity_dates, l._equity))
        first = float(l._equity[0])
        last = None
        series = []
        for dt in master:
            if dt in d:
                last = d[dt]
            series.append(last if last is not None else first)
        total += np.asarray(series, float)
    return total


def _returns(equity):
    e = np.asarray(equity, float)
    if e.size < 2:
        return np.array([], float)
    prev = e[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(prev != 0.0, np.diff(e) / prev, 0.0)


def _rank(x):
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), float)
    r[order] = np.arange(len(x), dtype=float)
    return r


def _corr(a, b):
    if a.size < 2 or b.size < 2 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _beta(y, x):
    if x.size < 2 or x.std() == 0:
        return float("nan")
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(coef[0])


def _rolling_corr(a, b, w=20):
    n = min(a.size, b.size)
    out = []
    for i in range(n):
        lo = max(0, i - w + 1)
        out.append(_corr(a[lo:i + 1], b[lo:i + 1]) if i - lo >= 1 else float("nan"))
    return out


def _analyse(voter_led, stat_led):
    vl = [l for l in voter_led.values() if getattr(l, "_equity", None)]
    sl = [l for l in stat_led.values() if getattr(l, "_equity", None)]
    master = sorted(set().union(*[set(l._equity_dates) for l in vl + sl]))
    v_eq = _fill_arm(vl, master)
    s_eq = _fill_arm(sl, master)
    v_r, s_r = _returns(v_eq), _returns(s_eq)

    v_dates = set().union(*[set(l._equity_dates) for l in vl])
    s_dates = set().union(*[set(l._equity_dates) for l in sl])
    both = sorted(v_dates & s_dates)
    v_map = dict(zip(master, v_eq)); s_map = dict(zip(master, s_eq))
    v_both = _returns(np.array([v_map[d] for d in both], float))
    s_both = _returns(np.array([s_map[d] for d in both], float))

    v_pool = RM.report_pooled(vl)[0]
    s_pool = RM.report_pooled(sl)[0]
    port = RM.report_pooled(vl + sl)[0]

    return {
        "master": master, "v_eq": v_eq, "s_eq": s_eq, "v_r": v_r, "s_r": s_r,
        "rolling": _rolling_corr(v_r, s_r),
        "metrics": {
            "pearson_union": _corr(v_r, s_r),
            "spearman_union": _corr(_rank(v_r), _rank(s_r)) if v_r.size > 2 else float("nan"),
            "pearson_both_active": _corr(v_both, s_both),
            "n_both_active_days": len(both),
            "beta_stat_on_voter": _beta(s_r, v_r),
            "beta_voter_on_stat": _beta(v_r, s_r),
            "voter_pool": v_pool,
            "stat_pool": s_pool,
            "portfolio_pool": port,
            "portfolio_diversification_ratio": RM.diversification_ratio(vl + sl),
        },
    }


def _plot(res_by_mode, out_png):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"(plot skipped: {e})")
        return
    modes = list(res_by_mode)
    fig, ax = plt.subplots(len(modes), 3, figsize=(15, 4.2 * len(modes)), squeeze=False)
    for i, m in enumerate(modes):
        r = res_by_mode[m]
        x = np.arange(len(r["master"]))
        ax[i][0].plot(x, r["v_eq"], label="voter")
        ax[i][0].plot(x, r["s_eq"], label="stat PRA")
        ax[i][0].set_title(f"{m}: arm equity (MTM)")
        ax[i][0].legend(); ax[i][0].set_xlabel("trading day")
        ax[i][1].scatter(r["v_r"], r["s_r"], s=8, alpha=0.5)
        b = r["metrics"]["beta_stat_on_voter"]
        if b == b:
            xr = np.linspace(r["v_r"].min(), r["v_r"].max(), 50)
            ax[i][1].plot(xr, b * xr, color="red", lw=1)
        ax[i][1].set_title(f"{m}: daily returns, beta={b:+.2f}")
        ax[i][1].set_xlabel("voter"); ax[i][1].set_ylabel("stat")
        ax[i][2].plot(r["rolling"])
        ax[i][2].axhline(0, color="grey", lw=0.6)
        ax[i][2].set_title(f"{m}: rolling 20d correlation")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    print(f"wrote {out_png}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Voter vs stat PRA cross-arm dependence, 2024.")
    ap.add_argument("--season", type=int, default=2024)
    ap.add_argument("--bankroll-per-arm", type=float, default=3000.0)
    ap.add_argument("--stat-bss-json", default="models/stat_leader/bss_by_season_pra.json")
    ap.add_argument("--modes", nargs="+", default=["equal", "bss"], choices=["equal", "bss"])
    ap.add_argument("--out", default="out/arm_corr_2024")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    res_by_mode = {}
    for mode in args.modes:
        if mode == "equal":
            v_budgets = s_budgets = None
        else:
            try:
                from scripts.strategy.allocation import book_weighting as BW
                from scripts.strategy.allocation import stat_book_weighting as SBW
            except ImportError:  # pragma: no cover
                import book_weighting as BW  # type: ignore
                import stat_book_weighting as SBW  # type: ignore
            if not os.path.exists(args.stat_bss_json):
                print(f"(mode=bss skipped: {args.stat_bss_json} not found; run "
                      f"stat_bss_persist first, then re-run this with --modes bss)")
                continue
            v_budgets = BW.compute_budgets(args.season, args.bankroll_per_arm, awards=VOTER)
            s_budgets = SBW.compute_budgets(args.season, args.bankroll_per_arm,
                                            awards=STAT, path=args.stat_bss_json)
            print(f"[bss] voter {v_budgets} | stat {s_budgets}")
        voter_led = _run_arm(args.season, VOTER, v_budgets)
        stat_led = _run_arm(args.season, STAT, s_budgets)
        res_by_mode[mode] = _analyse(voter_led, stat_led)

    if not res_by_mode:
        print("no modes ran")
        return 1

    dump = {m: r["metrics"] for m, r in res_by_mode.items()}
    with open(os.path.join(args.out, f"arm_corr_{args.season}.json"), "w") as fh:
        json.dump(dump, fh, indent=2, default=float)
    _plot(res_by_mode, os.path.join(args.out, f"arm_corr_{args.season}.png"))

    for m, r in res_by_mode.items():
        mt = r["metrics"]
        print(f"\n===== {m} =====")
        print(f"  Pearson (union days)      {mt['pearson_union']:+.3f}")
        print(f"  Spearman (union days)     {mt['spearman_union']:+.3f}")
        print(f"  Pearson (both-active)     {mt['pearson_both_active']:+.3f} "
              f"(n={mt['n_both_active_days']})")
        print(f"  beta stat~voter           {mt['beta_stat_on_voter']:+.3f}")
        print(f"  beta voter~stat           {mt['beta_voter_on_stat']:+.3f}")
        print(f"  diversification ratio     {mt['portfolio_diversification_ratio']}")
        for label, key in (("voter", "voter_pool"), ("stat", "stat_pool"),
                           ("portfolio", "portfolio_pool")):
            pl = mt[key]
            print(f"  {label:10s} Sharpe {pl.get('sharpe')} maxDD {pl.get('max_drawdown_pct')}% "
                  f"PnL {pl.get('final', 0) - pl.get('start', 0):+.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
