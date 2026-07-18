"""Cross-arm dependence and within-arm strategy analysis for the NBA Awards Trader.

Two jobs, split by what each season allows out-of-sample:

  Cross-arm correlation (voter vs stat) must be measured on 2025, the only season
  where both arms are OOS. The voter final model is trained through 2024, so a 2024
  voter backtest is in-sample and its own train-seasons guard refuses it; the stat
  arm is OOS on both 2024 and 2025. Run this as --arms both --season 2025 --modes
  equal, after the 2025 stat prices are pulled.

  Within-arm bss-vs-equal validation is stat-only on 2024, where the stat model runs
  OOS. Run this as --arms stat --season 2024 --modes equal bss, to see whether the
  shrunk leaderboard-BSS split beats equal weight before it is ever spent on 2025.

It reconstructs each book's daily equity from the ledger MTM series (ledger._equity
via risk_metrics.pooled_curve), never the flat static-budget equity_curve CSV. A
requested arm that a season will not allow (the voter train-seasons guard on 2024)
is skipped with a message rather than aborting the whole run. British English.
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


def _run_arm(season, awards, budgets):
    """Run one arm, returning {award: ledger}, or {} if the season guards it out."""
    saved = os.environ.get("ALLOC_BUDGETS_JSON")
    if budgets is None:
        os.environ.pop("ALLOC_BUDGETS_JSON", None)
    else:
        os.environ["ALLOC_BUDGETS_JSON"] = json.dumps(budgets)
    try:
        results, _ = run_singlepass(season, awards=tuple(awards),
                                    budget_per_book=1000.0, compound=False)
        return {aw: results[aw]["ledger"] for aw in awards}
    except Exception as e:  # noqa: BLE001
        print(f"(arm {awards} on {season} skipped: {e})")
        return {}
    finally:
        if saved is None:
            os.environ.pop("ALLOC_BUDGETS_JSON", None)
        else:
            os.environ["ALLOC_BUDGETS_JSON"] = saved


def _fill_arm(ledgers, master):
    total = np.zeros(len(master), float)
    for l in ledgers:
        if not getattr(l, "_equity", None):
            continue
        d = dict(zip(l._equity_dates, l._equity))
        first = float(l._equity[0]); last = None; series = []
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
    order = np.argsort(x, kind="mergesort"); r = np.empty(len(x), float)
    r[order] = np.arange(len(x), dtype=float); return r


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
    n = min(a.size, b.size); out = []
    for i in range(n):
        lo = max(0, i - w + 1)
        out.append(_corr(a[lo:i + 1], b[lo:i + 1]) if i - lo >= 1 else float("nan"))
    return out


def _pair(voter_led, stat_led):
    vl = [l for l in voter_led.values() if getattr(l, "_equity", None)]
    sl = [l for l in stat_led.values() if getattr(l, "_equity", None)]
    master = sorted(set().union(*[set(l._equity_dates) for l in vl + sl]))
    v_eq, s_eq = _fill_arm(vl, master), _fill_arm(sl, master)
    v_r, s_r = _returns(v_eq), _returns(s_eq)
    v_dates = set().union(*[set(l._equity_dates) for l in vl])
    s_dates = set().union(*[set(l._equity_dates) for l in sl])
    both = sorted(v_dates & s_dates)
    vmap, smap = dict(zip(master, v_eq)), dict(zip(master, s_eq))
    v_both = _returns(np.array([vmap[d] for d in both], float))
    s_both = _returns(np.array([smap[d] for d in both], float))
    return {
        "master": master, "v_eq": v_eq, "s_eq": s_eq, "v_r": v_r, "s_r": s_r,
        "rolling": _rolling_corr(v_r, s_r),
        "pearson_union": _corr(v_r, s_r),
        "spearman_union": _corr(_rank(v_r), _rank(s_r)) if v_r.size > 2 else float("nan"),
        "pearson_both_active": _corr(v_both, s_both),
        "n_both_active_days": len(both),
        "beta_stat_on_voter": _beta(s_r, v_r),
        "beta_voter_on_stat": _beta(v_r, s_r),
        "portfolio_pool": RM.report_pooled(vl + sl)[0],
        "portfolio_diversification_ratio": RM.diversification_ratio(vl + sl),
    }


def _plot(pairs, out_png):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"(plot skipped: {e})"); return
    modes = list(pairs)
    fig, ax = plt.subplots(len(modes), 3, figsize=(15, 4.2 * len(modes)), squeeze=False)
    for i, m in enumerate(modes):
        r = pairs[m]; x = np.arange(len(r["master"]))
        ax[i][0].plot(x, r["v_eq"], label="voter"); ax[i][0].plot(x, r["s_eq"], label="stat")
        ax[i][0].set_title(f"{m}: arm equity (MTM)"); ax[i][0].legend()
        ax[i][1].scatter(r["v_r"], r["s_r"], s=8, alpha=0.5)
        b = r["beta_stat_on_voter"]
        if b == b:
            xr = np.linspace(r["v_r"].min(), r["v_r"].max(), 50)
            ax[i][1].plot(xr, b * xr, color="red", lw=1)
        ax[i][1].set_title(f"{m}: daily returns, beta={b:+.2f}")
        ax[i][2].plot(r["rolling"]); ax[i][2].axhline(0, color="grey", lw=0.6)
        ax[i][2].set_title(f"{m}: rolling 20d correlation")
    fig.tight_layout(); fig.savefig(out_png, dpi=110); print(f"wrote {out_png}")


def _arm_strategy(ledgers):
    ll = [l for l in ledgers.values() if getattr(l, "_equity", None)]
    m = RM.report_pooled(ll)[0]
    m["diversification_ratio"] = RM.diversification_ratio(ll)
    return m


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Cross-arm correlation and within-arm strategy read.")
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--arms", choices=["voter", "stat", "both"], default="both")
    ap.add_argument("--stat-awards", nargs="+", default=["PTS", "REB", "AST"])
    ap.add_argument("--bankroll-per-arm", type=float, default=3000.0)
    ap.add_argument("--stat-bss-json", default="models/stat_leader/bss_by_season_pra.json")
    ap.add_argument("--modes", nargs="+", default=["equal"], choices=["equal", "bss"])
    ap.add_argument("--out", default="out/arm_corr")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    stat_awards = tuple(args.stat_awards)
    want_voter = args.arms in ("voter", "both")
    want_stat = args.arms in ("stat", "both")

    pairs, dump = {}, {}
    for mode in args.modes:
        v_budgets = s_budgets = None
        if mode == "bss":
            try:
                from scripts.strategy.allocation import book_weighting as BW
                from scripts.strategy.allocation import stat_book_weighting as SBW
            except ImportError:  # pragma: no cover
                import book_weighting as BW  # type: ignore
                import stat_book_weighting as SBW  # type: ignore
            if want_voter:
                v_budgets = BW.compute_budgets(args.season, args.bankroll_per_arm, awards=VOTER)
            if want_stat:
                if not os.path.exists(args.stat_bss_json):
                    print(f"(mode=bss stat skipped: {args.stat_bss_json} missing)")
                else:
                    s_budgets = SBW.compute_budgets(args.season, args.bankroll_per_arm,
                                                    awards=stat_awards, path=args.stat_bss_json)
            print(f"[bss] voter {v_budgets} | stat {s_budgets}")

        run_stat = want_stat and not (mode == "bss" and s_budgets is None)
        voter_led = _run_arm(args.season, VOTER, v_budgets) if want_voter else {}
        stat_led = _run_arm(args.season, stat_awards, s_budgets) if run_stat else {}

        entry = {}
        if voter_led:
            entry["voter_strategy"] = _arm_strategy(voter_led)
        if stat_led:
            entry["stat_strategy"] = _arm_strategy(stat_led)
        if voter_led and stat_led:
            p = _pair(voter_led, stat_led)
            pairs[mode] = p
            entry["cross_arm"] = {k: v for k, v in p.items()
                                  if k not in ("master", "v_eq", "s_eq", "v_r", "s_r", "rolling")}
        dump[mode] = entry

    with open(os.path.join(args.out, f"arm_corr_{args.season}.json"), "w") as fh:
        json.dump(dump, fh, indent=2, default=float)
    if pairs:
        _plot(pairs, os.path.join(args.out, f"arm_corr_{args.season}.png"))

    for mode, entry in dump.items():
        print(f"\n===== {mode} =====")
        for arm in ("voter_strategy", "stat_strategy"):
            if arm in entry:
                m = entry[arm]
                print(f"  {arm:16s} Sharpe {m.get('sharpe')} maxDD {m.get('max_drawdown_pct')}% "
                      f"PnL {m.get('final', 0) - m.get('start', 0):+.0f} "
                      f"divratio {m.get('diversification_ratio')}")
        if "cross_arm" in entry:
            ca = entry["cross_arm"]
            print(f"  Pearson union      {ca['pearson_union']:+.3f}")
            print(f"  Spearman union     {ca['spearman_union']:+.3f}")
            print(f"  Pearson both-live  {ca['pearson_both_active']:+.3f} "
                  f"(n={ca['n_both_active_days']})")
            print(f"  beta stat~voter    {ca['beta_stat_on_voter']:+.3f}")
            print(f"  portfolio divratio {ca['portfolio_diversification_ratio']}")
            pp = ca["portfolio_pool"]
            print(f"  portfolio          Sharpe {pp.get('sharpe')} "
                  f"maxDD {pp.get('max_drawdown_pct')}% PnL {pp.get('final',0)-pp.get('start',0):+.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
