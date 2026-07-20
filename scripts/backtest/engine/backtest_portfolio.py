"""Portfolio-cadence backtest: one shared bankroll across every book.

Where backtest_singlepass runs each book as an independent process against its own
budget, this engine steps every book in lockstep over the union of their trading days
so the books can be coordinated at each snapshot. The coordination is the whole point:

  1. SHARED EQUITY FRAME. Total portfolio equity-at-cost is initial capital plus the
     summed realised PnL of every book (deliberately at cost, not mark-to-market, to
     match the per-book Kelly frame). Every book sizes its own within-book Kelly solve
     against this shared total, not against a per-book slice, so idle capital in one
     book is available to another.

  2. BSS TILT. Each book's sizing base is the shared equity times a mean-one within-arm
     skill tilt (portfolio_alloc.within_arm_tilts), so a book the model prices more
     skilfully leans in and a weaker one leans out, with mean deployment preserved.

  3. STRUCTURAL CAPS + JOINT CONSTRAINT. After every book has proposed its per-leg
     targets for the day, the flat list of legs is clamped by the player, award and
     type caps and then by the joint capital constraint (portfolio_alloc), each binding
     clamp trimming from the lowest risk-adjusted edge upward. This is where the
     cross-book coupling lives, in place of a cross-book covariance we do not estimate.

The per-book cash ledgers cannot literally share cash, so each is given a non-binding
buffer equal to the whole initial portfolio; the real limit is the joint constraint in
(3), and every settled PnL stays per-position exact. Pooled return is measured against
the true initial portfolio, not the summed buffers.

No sealed gate is asserted here: the allocation is a deliberate departure from the
per-book design, so bit-identity with the +$460 golden is neither expected nor wanted.

The at-cost base is rebased monthly, not daily, so intra-month realisations do not feed
the sizing loop. The concave cap and the no-trade band are referenced to raw equity, not
the tilted sizing base, so neither wobbles with the tilt. HOLD_ITM holds a deep in-the-
radj is a size-independent edge-quality gate and the open decision ramps with it, so a
positive-edge name is sized down rather than refused. The band widens on idle capital: when
portfolio slack is high and a held position's own-side edge is still non-negative within a
noise band it rides rather than churns, and when capital is scarce the widening vanishes and
the band reallocates. This coordinator publishes day-start slack to notrade_region each day.

EXEC_SCHEDULE=1 turns on the Almgren-Chriss daily fill pacing (exec_schedule.pace); it is
off by default so the engine fills to target in one day unless asked otherwise. Tune with
EXEC_PHI_BASE (fill fraction at median vol, default 0.4) and EXEC_PHI_MIN (floor, 0.15).
Band widening tunes with SLACK_KAPPA (default 2.0) and EDGE_NOISE (default 0.02).

Run (2024 validation, six books, $6000):
  caffeinate -i env USE_REGION=1 REGION_CONFIRM=2 REGION_HYST=1.0 \
    REGION_BAND_FLOOR=0.02 ASYM_TRIM=1 CARRY_R=0.05 \
    CONC_FORM=saturating CONC_S_CAP_NO=0.05 CONC_S_CAP_YES=0.15 COST_FILL_FLOOR=0.02 \
    uv run python -m scripts.backtest.engine.backtest_portfolio \
      --season 2024 --awards MVP DPOY ROTY PTS REB AST \
      --equity 6000 --out out/portfolio_2024

British English, no em dashes.
"""
from __future__ import annotations

import os
import json
import argparse
import datetime as _dt

from scripts.backtest.engine import backtest_orchestrator as bo
from scripts.backtest.engine import backtest_orchestrator_daily as bod
from scripts.backtest.registry import get_spec
from scripts.strategy.allocation import portfolio_alloc as pa
from scripts.strategy.trade_regions import notrade_region as _ntr


DEFAULT_PORTFOLIO_AWARDS = ("MVP", "DPOY", "ROTY", "PTS", "REB", "AST")


def _daykey(day) -> str:
    return _dt.date.fromisoformat(str(day)[:10]).isoformat()


def _book_equity_at_cost(ledger) -> float:
    """Uninvested cash plus open positions at entry cost (starting_cash plus realised
    PnL); the same at-cost frame the per-book engine uses."""
    open_cost = 0.0
    for p in ledger._pos.values():
        open_cost += float(getattr(p, "outlay_eff", 0.0) or 0.0)
    return float(ledger.cash) + open_cost


def _arm_of(spec) -> str:
    """Award type / arm inferred from the draw source: the voter arm is the LightGBM
    Plackett-Luce cloud, the stat arm is the NGBoost + Monte Carlo pool."""
    return "voter" if spec.pwin_kind == "cloud" else "stat"


def _stat_skill(book, season, path):
    """Pooled BSS-vs-leaderboard for a stat book over seasons strictly before `season`,
    read from the persisted per-season Brier artefact. None if unavailable."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        d = json.load(f)
    book_d = d.get(book)
    if not book_d:
        return None
    num = den = 0.0
    for s, v in book_d.items():
        if int(s) < int(season):
            num += float(v["n"]) * float(v["brier_model"])
            den += float(v["n"]) * float(v["brier_lead"])
    if den <= 0:
        return None
    return 1.0 - num / den


def _voter_skill(book, season):
    """Voter per-book skill signal taken from the pinned shrunk-skill book weights for
    the season (falling back to the nearest pinned season when the target is unpinned,
    since voter skill is near-stable across adjacent seasons). None if unavailable."""
    from scripts.common import config
    row = config.BOOK_WEIGHTS.get(season)
    if row is None and config.BOOK_WEIGHTS:
        nearest = min(config.BOOK_WEIGHTS, key=lambda s: abs(int(s) - int(season)))
        row = config.BOOK_WEIGHTS.get(nearest)
    if not row or book not in row:
        return None
    return float(row[book])


def _build_tilts(season, awards):
    """Mean-one within-arm skill tilt per book. Absent skills are neutralised to their
    arm mean so a book with no artefact sits at tilt one rather than being penalised."""
    from scripts.common import config
    shrink = float(getattr(config, "PORTFOLIO_TILT_SHRINK", 0.5))
    bss_path = str(getattr(config, "PORTFOLIO_BSS_PATH",
                           "models/stat_leader/bss_by_season_pra.json"))
    arm_by_book = {aw: _arm_of(get_spec(aw)) for aw in awards}
    raw = {}
    for aw in awards:
        raw[aw] = (_voter_skill(aw, season) if arm_by_book[aw] == "voter"
                   else _stat_skill(aw, season, bss_path))
    by_arm = {}
    for aw in awards:
        by_arm.setdefault(arm_by_book[aw], []).append(aw)
    skill = {}
    for arm, books in by_arm.items():
        known = [raw[b] for b in books if raw[b] is not None]
        fill = (sum(known) / len(known)) if known else 0.0
        for b in books:
            skill[b] = raw[b] if raw[b] is not None else fill
    tilts = pa.within_arm_tilts(skill, arm_by_book, shrink=shrink)
    return tilts, arm_by_book, skill


def _execute_book_day(ctx, day, out, clamped):
    """Replicates step_award_day's post-solve execution, but drives the ledger to the
    coordinated (clamped) targets rather than the book's own uncoordinated ones."""
    ledger = ctx.ledger
    samples = out["samples"]
    yes_mids = out["yes_mids"]
    if out["empty"]:
        bo.self_mark(ledger, day, yes_mids, samples)
        ledger.record_deployed(day, 0.0)
        return
    names = ctx.names
    candidates = ctx.candidates
    verbose = ctx.verbose
    pid_to_idx = out["pid_to_idx"]

    if os.environ.get("EXEC_SCHEDULE") == "1":
        from scripts.strategy.sizing import exec_schedule as _es
        _cur = {}
        for _p in set(clamped) | set(ledger._pos.keys()):
            _pos = ledger._pos.get(_p)
            _cur[_p] = (_pos.outlay_eff if _pos.side == "YES" else -_pos.outlay_eff) if _pos else 0.0
        clamped = _es.pace(clamped, _cur, out.get("psig_by_pid", {}), out.get("frac", 1.0),
                           phi_base=float(os.environ.get("EXEC_PHI_BASE", "0.4")),
                           phi_min=float(os.environ.get("EXEC_PHI_MIN", "0.15")))

    for pid, i in pid_to_idx.items():
        leg = ledger._pos.get(pid)
        if leg is None:
            continue
        pwin_win = float(samples.vote_share_pred[i])
        pwin_leg = pwin_win if leg.side == "YES" else 1.0 - pwin_win
        ledger.set_model_context(pid, fv_yes_terminal=pwin_win, pwin_leg_terminal=pwin_leg)

    act_pids = set(clamped) | {pid for pid in list(ledger._pos.keys()) if pid in yes_mids}
    _fill_floor = float(os.environ.get("COST_FILL_FLOOR", bo.FILL_FLOOR))
    for pid in act_pids:
        tgt = float(clamped.get(pid, 0.0))
        i = pid_to_idx.get(pid)
        fv = float(samples.vote_share_pred[i]) if i is not None else None
        bo._rebalance_to(ledger, candidates[pid], day, pid, tgt, yes_mids[pid],
                         fv_yes=fv, name=names.get(pid), verbose=verbose,
                         fill_floor=_fill_floor)

    bo.self_mark(ledger, day, yes_mids, samples)
    ledger.record_deployed(day, float(sum(abs(v) for v in clamped.values())))


def _real_mtm(eq, initial, rf=0.03, ppy=252):
    """Recompute pooled MTM against the true portfolio base. report_pooled sums the per-book
    ledgers, each carrying a non-binding buffer equal to the whole portfolio, so its curve
    starts near n_books * initial and dilutes every percentage by that factor. Offsetting the
    curve so it starts at `initial` restores the real return, volatility and drawdown; the
    ratios shift too because the risk-free drag is no longer diluted."""
    import numpy as np
    eq = np.asarray(eq, dtype=float)
    if eq.size < 2:
        return {"n": int(eq.size)}
    real = eq - (float(eq[0]) - float(initial))          # start exactly at `initial`
    rets = np.diff(real) / real[:-1]
    n = rets.size
    total = real[-1] / real[0] - 1.0
    ann_ret = (real[-1] / real[0]) ** (ppy / n) - 1.0 if (real[0] > 0 and n > 0) else 0.0
    ann_vol = float(np.std(rets, ddof=1) * np.sqrt(ppy)) if n > 1 else 0.0
    downside = rets[rets < 0]
    dvol = float(np.sqrt(np.mean(downside ** 2)) * np.sqrt(ppy)) if downside.size else 0.0
    run_max = np.maximum.accumulate(real)
    maxdd = float(((real - run_max) / run_max).min())
    sh = (ann_ret - rf) / ann_vol if ann_vol > 0 else 0.0
    so = (ann_ret - rf) / dvol if dvol > 0 else 0.0
    cal = ann_ret / abs(maxdd) if maxdd < 0 else 0.0
    return {"start": round(float(real[0]), 1), "final": round(float(real[-1]), 1),
            "total_return_pct": round(float(100 * total), 1),
            "annualised_return_pct": round(float(100 * ann_ret), 1),
            "annualised_vol_pct": round(float(100 * ann_vol), 1), "sharpe": round(float(sh), 2),
            "sortino": round(float(so), 2), "calmar": round(float(cal), 2),
            "max_drawdown_pct": round(float(100 * maxdd), 1), "n": int(eq.size), "rf": float(rf)}


def run_portfolio(season, awards=DEFAULT_PORTFOLIO_AWARDS, equity=6000.0,
                  use_stub=False, verbose=False, rf=0.03):
    from scripts.common import config
    from scripts.common.db import connect

    player_base = float(getattr(config, "PORTFOLIO_PLAYER_CAP_BASE", 0.15))
    player_alpha = float(getattr(config, "PORTFOLIO_PLAYER_CAP_ALPHA", 0.5))
    award_frac = float(getattr(config, "PORTFOLIO_AWARD_CAP_FRAC", 0.50))
    type_c = float(getattr(config, "PORTFOLIO_TYPE_CAP_C", 1.0))
    validated = dict(getattr(config, "PORTFOLIO_TYPE_VALIDATED", {"voter": 3, "stat": 3}))
    initial = float(equity)

    conn = connect("data/awards.db")
    tilts, arm_by_book, skill = _build_tilts(season, awards)
    print(f"[portfolio] initial ${initial:.0f}  awards={list(awards)}")
    print(f"[portfolio] arms={arm_by_book}")
    print(f"[portfolio] skill={ {k: round(v, 4) for k, v in skill.items()} }")
    print(f"[portfolio] tilt={ {k: round(v, 3) for k, v in tilts.items()} }")

    ctxs = {}
    daykeys = {}
    for aw in awards:
        spec = get_spec(aw)
        ctx = bod.prepare_award_daily(conn, spec, season, initial,
                                      use_stub=use_stub, verbose=verbose)
        ctx.start_cash = initial
        ctxs[aw] = ctx
        daykeys[aw] = {_daykey(d) for d in ctx.days}

    arms_present = sorted({arm_by_book[aw] for aw in awards})
    union_days = sorted(set().union(*daykeys.values())) if daykeys else []
    equity_log = []

    _eq_ref = initial
    _eq_month = None
    for day in union_days:
        _month = str(day)[:7]
        if _month != _eq_month:
            _eq_ref = max(initial + sum(_book_equity_at_cost(c.ledger) - c.start_cash
                                        for c in ctxs.values()), 1.0)
            _eq_month = _month
        eq = _eq_ref
        _deployed = sum(abs(getattr(p, "outlay_eff", 0.0) or 0.0)
                        for c in ctxs.values() for p in c.ledger._pos.values())
        _ntr._PORTFOLIO_SLACK = float(max(0.0, min(1.0, 1.0 - _deployed / eq)))
        tc = pa.type_ceilings(arms_present, validated, type_c, eq)

        day_outs = {}
        legs = []
        for aw in awards:
            if day not in daykeys[aw]:
                continue
            ctx = ctxs[aw]
            budget_book = eq * float(tilts.get(aw, 1.0))
            out = bod._award_core(ctx, day, budget_book, ctx.region_state, record=True,
                                  ref_budget=eq)
            ctx.region_state = out["region_state"]
            day_outs[aw] = out
            if out["empty"]:
                continue
            radj_by_pid = out.get("radj_by_pid", {})
            for pid, tgt in out["target_by_pid"].items():
                legs.append({"book": aw, "pid": pid, "arm": arm_by_book[aw],
                             "target": float(tgt),
                             "radj": float(radj_by_pid.get(pid, float("nan")))})

        pa.apply_structural_caps(legs, eq, player_base, player_alpha, award_frac, tc)

        clamped_by_book = {}
        for l in legs:
            clamped_by_book.setdefault(l["book"], {})[l["pid"]] = l["target"]

        deployed_today = 0.0
        for aw in awards:
            if day not in daykeys[aw]:
                continue
            out = day_outs[aw]
            clamped = clamped_by_book.get(aw, {})
            _execute_book_day(ctxs[aw], day, out, clamped)
            deployed_today += sum(abs(v) for v in clamped.values())

        equity_log.append({"date": str(day), "portfolio_equity_at_cost": round(eq, 4),
                           "deployed": round(deployed_today, 4),
                           "capital_usage_pct": round(100.0 * deployed_today / eq, 2)})

    results = {}
    for aw in awards:
        ledger, ev = bod.finalize_award_daily(ctxs[aw])
        results[aw] = {"ledger": ledger, "model_eval": ev,
                       "book_summary": ledger.book_summary()}

    realised = sum(r["book_summary"]["realised_pnl"] for r in results.values())
    pooled = {"season": season, "awards": list(awards), "portfolio": True,
              "initial_portfolio": initial, "realised_pnl_total": realised,
              "return_pct": 100.0 * realised / initial if initial else float("nan"),
              "n_transactions_total": sum(r["book_summary"]["n_transactions"]
                                          for r in results.values()),
              "tilt": tilts, "skill": skill,
              "equity_curve": equity_log}
    from scripts.common.risk_metrics import report_pooled
    _mtm_diluted, _dates, _eq = report_pooled([r["ledger"] for r in results.values()], rf=rf)
    pooled["mtm"] = _real_mtm(_eq, initial, rf=rf)
    pooled["mtm_diluted_buffer_base"] = _mtm_diluted
    print("POOLED MTM (real base):", pooled["mtm"])
    return results, pooled


def main():
    ap = argparse.ArgumentParser(description="Shared-bankroll portfolio backtest.")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--awards", nargs="+", default=list(DEFAULT_PORTFOLIO_AWARDS),
                    help="books sharing the bankroll (exclude the season's sealed books)")
    ap.add_argument("--equity", type=float, default=6000.0,
                    help="total initial portfolio (default 6000)")
    ap.add_argument("--out", default="out/portfolio")
    ap.add_argument("--stub-cost", action="store_true")
    ap.add_argument("--rf", type=float, default=0.03,
                    help="annual risk-free rate for Sharpe/Sortino (default 0.03)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    results, pooled = run_portfolio(
        args.season, awards=tuple(args.awards), equity=args.equity,
        use_stub=args.stub_cost, rf=args.rf)

    trade_rows, pos_rows, book_rows, model_eval = [], [], [], []
    print(f"config: region={os.environ.get('USE_REGION')=='1'} "
          f"asym_trim={os.environ.get('ASYM_TRIM')=='1'} portfolio=True")
    for aw in args.awards:
        r = results[aw]
        b = r["book_summary"]
        trade_rows += r["ledger"].trade_log
        pos_rows += r["ledger"].position_log
        book_rows.append(b)
        model_eval += r["model_eval"]
        print(f"  {aw}: PnL ${b['realised_pnl']:.0f} "
              f"n_txn={b.get('n_transactions', 0)} tilt={pooled['tilt'].get(aw, 1.0):.3f}")
    print(f"POOLED PnL ${pooled['realised_pnl_total']:.0f} "
          f"({pooled['return_pct']:.1f}% of ${pooled['initial_portfolio']:.0f}) -> {args.out}")

    bo._dump_csv(os.path.join(args.out, f"trade_log_{args.season}.csv"), trade_rows)
    bo._dump_csv(os.path.join(args.out, f"position_log_{args.season}.csv"), pos_rows)
    bo._dump_csv(os.path.join(args.out, f"model_eval_{args.season}.csv"), model_eval)
    bo._dump_csv(os.path.join(args.out, f"equity_curve_{args.season}.csv"),
                 pooled["equity_curve"])
    with open(os.path.join(args.out, f"book_summary_{args.season}.json"), "w") as f:
        json.dump(book_rows, f, indent=2, default=str)
    with open(os.path.join(args.out, f"pooled_{args.season}.json"), "w") as f:
        json.dump(pooled, f, indent=2, default=str)


if __name__ == "__main__":
    main()
