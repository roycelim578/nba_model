"""Single-pass backtest engine: the three award books run as independent daily
loops, one process each, over their own trading days, each advancing its own ledger,
then pooled. Books share no state during stepping in the static and compound cases,
which is why they parallelise cleanly.

  COMPOUND=1  sizes each book off its current realised equity (cash plus open
              positions at cost), not a static budget.
  ASYM_TRIM=1 (read inside notrade_region) widens the same-side trim band on
              converging winners.

Static budgets come from ALLOC_BUDGETS_JSON (the shrunk-BSS split); absent that, an
equal per-book split. British English.
"""
from __future__ import annotations
import os
import json
import datetime as _dt

from scripts.backtest.registry import DEFAULT_AWARDS, get_spec


def _daykey(day) -> str:
    return _dt.date.fromisoformat(str(day)[:10]).isoformat()


def _book_equity_at_cost(ledger) -> float:
    """Book equity for the Kelly frame: uninvested cash plus open positions at entry
    cost (starting_cash plus realised PnL; deliberately not mark-to-market)."""
    open_cost = 0.0
    for p in ledger._pos.values():
        open_cost += float(getattr(p, "outlay_eff", 0.0) or 0.0)
    return float(ledger.cash) + open_cost


def _run_book_worker(job):
    """One book, whole day-loop, in its own process. Returns picklable results."""
    season, aw, budget_per_book, use_stub, verbose, mode, static_budget = job
    from scripts.common.db import connect
    from scripts.backtest.engine.backtest_orchestrator_daily import (
        prepare_award_daily, step_award_day, finalize_award_daily)
    spec = get_spec(aw)
    ctx = prepare_award_daily(connect("data/awards.db"), spec, season, budget_per_book,
                              use_stub=use_stub, verbose=verbose)
    entries = []
    for day in sorted({_daykey(d) for d in ctx.days}):
        if mode == "compound":
            eq = _book_equity_at_cost(ctx.ledger)
            ctx.portfolio_usd = eq
            ctx.award_budget_default = eq
        else:
            ctx.portfolio_usd = static_budget
            ctx.award_budget_default = static_budget
        entries.append({"date": str(day), "award": aw,
                        "equity_at_cost": round(ctx.portfolio_usd, 4)})
        step_award_day(ctx, day)
    ledger, ev = finalize_award_daily(ctx)
    return aw, ledger, ev, entries


def run_singlepass(season, awards=DEFAULT_AWARDS, budget_per_book=1000.0,
                   use_stub=False, verbose=False, compound=False, rf=0.03):
    """Fan the independent books across processes and pool. Equity curve re-sorted to
    day-major for stable output. Budgets from ALLOC_BUDGETS_JSON when present."""
    import json as _json
    from concurrent.futures import ProcessPoolExecutor, as_completed
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None
    _ov = os.environ.get("ALLOC_BUDGETS_JSON")
    if _ov:
        budgets = {aw: float(_json.loads(_ov)[aw]) for aw in awards}
        print(f"[equity] fixed override -> {budgets} (sum {sum(budgets.values()):.0f})")
        print(f"[alloc] fixed override -> {budgets} (sum {sum(budgets.values()):.0f})")
    else:
        from scripts.common import config
        _row = config.BOOK_WEIGHTS.get(season)
        if _row:
            budgets = {aw: float(_row[get_spec(aw).book_key]) for aw in awards}
            _shown = {aw: round(budgets[aw]) for aw in awards}
            print(f"[alloc] pinned BOOK_WEIGHTS[{season}] -> {_shown} (sum {sum(_shown.values())})")
        else:
            budgets = {aw: float(budget_per_book) for aw in awards}
            print(f"[alloc] no pinned weights for {season}; equal split {budget_per_book:.0f}/book")
    mode = "compound" if compound else "static"
    jobs = [(season, aw, budget_per_book, use_stub, verbose, mode, budgets[aw])
            for aw in awards]

    collected, equity_all = {}, []
    with ProcessPoolExecutor(max_workers=min(3, len(awards))) as ex:
        futs = {ex.submit(_run_book_worker, j): j[1] for j in jobs}
        it = as_completed(futs)
        if tqdm is not None:
            it = tqdm(it, total=len(futs), desc="books", unit="book")
        for fut in it:
            aw, ledger, ev, entries = fut.result()
            collected[aw] = (ledger, ev)
            equity_all.extend(entries)

    results = {}
    for aw in awards:
        ledger, ev = collected[aw]
        results[aw] = {"ledger": ledger, "model_eval": ev,
                       "book_summary": ledger.book_summary()}
    union_days = sorted({e["date"] for e in equity_all})
    di = {d: i for i, d in enumerate(union_days)}
    ai = {aw: i for i, aw in enumerate(awards)}
    equity_log = sorted(equity_all, key=lambda e: (di[e["date"]], ai[e["award"]]))

    starting = sum(r["book_summary"]["starting_cash"] for r in results.values())
    realised = sum(r["book_summary"]["realised_pnl"] for r in results.values())
    pooled = {"season": season, "awards": list(awards), "compound": bool(compound),
              "starting_cash_total": starting, "realised_pnl_total": realised,
              "return_pct_static": 100.0 * realised / starting if starting else float("nan"),
              "n_transactions_total": sum(r["book_summary"]["n_transactions"] for r in results.values()),
              "equity_curve": equity_log}
    from scripts.common.risk_metrics import report_pooled
    _mtm, _mtm_dates, _mtm_eq = report_pooled([r["ledger"] for r in results.values()], rf=rf)
    pooled["mtm"] = _mtm
    print("POOLED MTM:", _mtm)
    return results, pooled


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Single-pass backtest engine.")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--awards", nargs="+", default=list(DEFAULT_AWARDS))
    ap.add_argument("--budget", type=float, default=1000.0)
    ap.add_argument("--out", default="out/singlepass")
    ap.add_argument("--stub-cost", action="store_true")
    ap.add_argument("--compound", action="store_true")
    ap.add_argument("--rf", type=float, default=0.03,
                    help="annual risk-free rate for Sharpe/Sortino (default 0.03)")
    args = ap.parse_args()

    compound = args.compound or os.environ.get("COMPOUND") == "1"
    from scripts.backtest.engine import backtest_orchestrator as bo
    os.makedirs(args.out, exist_ok=True)

    results, pooled = run_singlepass(
        args.season, awards=tuple(args.awards), budget_per_book=args.budget,
        use_stub=args.stub_cost, compound=compound, rf=args.rf)

    trade_rows, pos_rows, book_rows, model_eval = [], [], [], []
    print(f"config: region={os.environ.get('USE_REGION')=='1'} "
          f"asym_trim={os.environ.get('ASYM_TRIM')=='1'} compound={compound}")
    for aw in args.awards:
        r = results[aw]; b = r["book_summary"]
        trade_rows += r["ledger"].trade_log
        pos_rows += r["ledger"].position_log
        book_rows.append(b)
        model_eval += r["model_eval"]
        print(f"  {aw}: PnL ${b['realised_pnl']:.0f} ({b.get('return_pct', 0):.1f}%) "
              f"n_txn={b.get('n_transactions', 0)}")
    print(f"POOLED PnL ${pooled['realised_pnl_total']:.0f} "
          f"({pooled['return_pct_static']:.1f}% static bankroll) -> {args.out}")

    bo._dump_csv(os.path.join(args.out, f"trade_log_{args.season}.csv"), trade_rows)
    bo._dump_csv(os.path.join(args.out, f"position_log_{args.season}.csv"), pos_rows)
    bo._dump_csv(os.path.join(args.out, f"model_eval_{args.season}.csv"), model_eval)
    bo._dump_csv(os.path.join(args.out, f"equity_curve_{args.season}.csv"), pooled["equity_curve"])
    with open(os.path.join(args.out, f"book_summary_{args.season}.json"), "w") as f:
        json.dump(book_rows, f, indent=2, default=str)


if __name__ == "__main__":
    main()
