"""Daily-cadence sized backtest / live path.

Same model, same reweights, same sizer as backtest_orchestrator, but the trade
loop runs once per day instead of once per feature snapshot. Three cadences:

  model score + fatigue + jsfloor FP point : HELD from the carried feature
      snapshot (all read feature_stats_asof, which exists only at the ~20
      model_predictions snapshots, and none move meaningfully intra-week).
  eligibility (injury)                      : DAILY (eligibility_factors reads
      injuries and game-logs as-of the day passed to it).
  price + sizing + trade                    : DAILY.

All weekly helpers and constants are imported from backtest_orchestrator so the
two paths cannot drift. The weekly run_award is left untouched for A/B.

Reuses load_daily_grid (carry-forward + D+1 seal, unit-tested separately).

CRITICAL: JointSamples.vote_share_pred / sizing_weights are mutated in place by
the reweight block. A carried snapshot is reused across many days, so both are
reset from a per-snapshot pristine base at the top of every day, else the
reweights would compound day over day.

Run: uv run python -m scripts.backtest.engine.backtest_orchestrator_daily --season 2024 \
        --awards MVP DPOY ROTY --budget 1000 --out out/final_test_daily
"""
from __future__ import annotations

import os
import json
import argparse
import datetime as _dt

import numpy as np
from concurrent.futures import ProcessPoolExecutor as _PPE

from scripts.backtest.engine import backtest_orchestrator as bo
from scripts.backtest.engine.backtest_pricejoin_daily import load_daily_grid
from scripts.strategy.trade_regions import region_adapter as _region
from scripts.backtest.registry import AwardSpec, EXPLICIT_MASK, get_spec, load_vol_model


def _interp_frac(snap_frac_pairs, day):
    """Linear-interpolate season fraction for a daily date from the known
    (feature_snap_date, frac) pairs. Clamps outside the snapshot range."""
    xs = [(_dt.date.fromisoformat(str(s)[:10]).toordinal(), f) for s, f in snap_frac_pairs]
    xs.sort()
    d = _dt.date.fromisoformat(str(day)[:10]).toordinal()
    if d <= xs[0][0]:
        return xs[0][1]
    if d >= xs[-1][0]:
        return xs[-1][1]
    for (x0, f0), (x1, f1) in zip(xs, xs[1:]):
        if x0 <= d <= x1:
            w = (d - x0) / (x1 - x0) if x1 > x0 else 0.0
            return f0 + w * (f1 - f0)
    return xs[-1][1]


class AwardDailyCtx:
    """Per-book daily state carried across the single-pass loop."""
    pass


def prepare_award_daily(conn, award_or_spec, season, budget, ceiling=bo.KELLY_FILL_CEILING,
                    use_stub=False, verbose=True, turnover=None, n_restarts=6,
                    warn_spread=False, budget_asof=None):
    from scripts.common import config as _cfg
    spec = award_or_spec if isinstance(award_or_spec, AwardSpec) else get_spec(award_or_spec)
    award = spec.name
    _cfg.assert_not_sealed(spec.seal_key, season)
    from scripts.strategy.forward_estimates import forward_edge as fe
    from scripts.strategy.sizing import sizer_fill
    from scripts.strategy.sizing.sizer import rank_floor_mask
    from scripts.strategy.sizing.sizer import solve_award_v2
    from scripts.strategy.sizing.size_scaling import scale_allocation, ScaleParams
    from scripts.backtest.settle.trade_ledger import TradeLedger
    from scripts.strategy.cost.cost_model import CostModel

    feature_snaps, daily = spec.pricejoin_fn(conn, award, season)
    if not daily:
        raise RuntimeError(f"no daily grid for {award} {season}")
    samples_by_snap = spec.samples_fn(conn, award, season, feature_snaps)
    vol_model = load_vol_model(spec.vol_pkl)

    base_vsp = {s: np.asarray(v.vote_share_pred, float).copy() for s, v in samples_by_snap.items()}
    base_sw = {s: np.asarray(v.sizing_weights, float).copy() for s, v in samples_by_snap.items()}
    snap_frac_pairs = [(s, float(v.frac)) for s, v in samples_by_snap.items()]

    all_pids = sorted({int(p) for s in samples_by_snap.values() for p in s.player_ids})
    _nm = dict(conn.execute(
        "SELECT player_id, name FROM players WHERE player_id IN (%s)"
        % ",".join("?" * len(all_pids)), all_pids).fetchall())
    names = {int(p): _nm.get(int(p), str(p)) for p in all_pids}

    price_map = {}
    snap_frac = {}
    for day, (carried, yes_prices) in daily.items():
        snap_frac[str(day)] = _interp_frac(snap_frac_pairs, day)
        for pid, y in yes_prices.items():
            price_map[(int(pid), str(day))] = float(y)
    cm = CostModel.load(bo.COST_PARAMS_PATH)
    candidates = {int(pid): bo._RealCandidate(pid, price_map, snap_frac, cm) for pid in all_pids}

    ledger = TradeLedger(award, season, starting_cash=budget, names=names)
    hist_logit = {pid: [] for pid in all_pids}
    denom_diag = []
    model_eval = []
    last_day = list(daily.keys())[-1]
    _region_state = {}

    ctx = AwardDailyCtx()
    ctx.ScaleParams = ScaleParams
    ctx.award = award
    ctx.spec = spec
    ctx.fatigue_reweight = spec.fatigue_reweight
    ctx.guard_top_k = spec.guard_top_k
    ctx.jsfloor = spec.jsfloor
    ctx.pwin_kind = spec.pwin_kind
    ctx.rank_floor = spec.rank_floor
    ctx.base_sw = base_sw
    ctx.base_vsp = base_vsp
    ctx.candidates = candidates
    ctx.ceiling = ceiling
    ctx.conn = conn
    ctx.denom_diag = denom_diag
    ctx.fe = fe
    ctx.hist_logit = hist_logit
    ctx.last_day = last_day
    ctx.ledger = ledger
    ctx.model_eval = model_eval
    ctx.n_restarts = n_restarts
    ctx.names = names
    ctx.rank_floor_mask = rank_floor_mask
    ctx.samples_by_snap = samples_by_snap
    ctx.scale_allocation = scale_allocation
    ctx.season = season
    ctx.sizer_fill = sizer_fill
    ctx.snap_frac = snap_frac
    ctx.solve_award_v2 = solve_award_v2
    ctx.turnover = turnover
    ctx.verbose = verbose
    ctx.vol_model = vol_model
    ctx.warn_spread = warn_spread
    ctx.daily = daily
    ctx.days = list(daily.keys())
    ctx.budget_asof = budget_asof
    ctx.portfolio_usd = float(budget)
    ctx.award_budget_default = float(budget)
    ctx.region_state = _region_state
    return ctx


def _pwin_source(ctx, samples):
    """Per-candidate pwin draw source selected by AwardSpec.pwin_kind. "cloud"
    returns the eta-widened score cloud consumed by forward_edge exactly as the
    voter arm does. "pool" is the stat path: the per-candidate draw pool lives on
    the samples object (samples.pool, [n_draws, n_cand]) and is read directly by
    _composite through forward_edge's pw_pool argument, so the cloud is unused on
    this path and this returns None. Every cloud-consuming step in _award_core (eta
    widening, the renorm_set market-floor blend, the eligibility reweight) is guarded
    off on the pool path so fair value stays the calibrated full-field P(lead)."""
    kind = ctx.pwin_kind
    if kind == "cloud":
        return bo._cloud(samples)
    if kind == "pool":
        return None
    raise ValueError(f"unknown pwin_kind {kind!r}")


def _award_core(ctx, day, budget, region_state, record):
    ScaleParams = ctx.ScaleParams
    award = ctx.award
    base_sw = ctx.base_sw
    base_vsp = ctx.base_vsp
    candidates = ctx.candidates
    ceiling = ctx.ceiling
    conn = ctx.conn
    fe = ctx.fe
    hist_logit = ctx.hist_logit
    ledger = ctx.ledger
    model_eval = ctx.model_eval
    n_restarts = ctx.n_restarts
    names = ctx.names
    rank_floor_mask = ctx.rank_floor_mask
    samples_by_snap = ctx.samples_by_snap
    scale_allocation = ctx.scale_allocation
    season = ctx.season
    sizer_fill = ctx.sizer_fill
    snap_frac = ctx.snap_frac
    solve_award_v2 = ctx.solve_award_v2
    turnover = ctx.turnover
    verbose = ctx.verbose
    vol_model = ctx.vol_model
    warn_spread = ctx.warn_spread
    budget_asof = ctx.budget_asof
    _region_state = region_state
    carried, yes_prices = ctx.daily[day]
    samples = samples_by_snap[carried]
    samples.vote_share_pred[:] = base_vsp[carried]
    samples.sizing_weights[:] = base_sw[carried]
    pids = [int(p) for p in samples.player_ids]
    frac = snap_frac[str(day)]
    yes_mids = {int(pid): float(y) for pid, y in yes_prices.items()}
    cloud = _pwin_source(ctx, samples)

    _fat_detail = []
    if ctx.fatigue_reweight:
        from scripts.strategy.pricing import fatigue_reweight
        _fat, _fat_detail = fatigue_reweight.apply_fatigue(
            samples.vote_share_pred, pids, conn, award, season, snap=carried,
            return_detail=True)
        samples.vote_share_pred[:] = _fat

    if record:
        for pid in pids:
            m = yes_mids.get(pid)
            if m is not None:
                hist_logit[pid].append(float(bo._logit(m)))
    _stage_raw = list(map(float, samples.vote_share_pred))
    _pe = {} if ctx.pwin_kind == "pool" else bo._elig_factors(conn, bo._ELIG_DIST, award, season, day, pids)
    _stage_prefatigue = _stage_raw
    _stage_postfatigue = list(map(float, samples.vote_share_pred))
    if ctx.pwin_kind != "pool":
        samples.vote_share_pred[:] = bo._elig_reweight(samples.vote_share_pred, pids, _pe)
    _stage_postelig = list(map(float, samples.vote_share_pred))
    _fat_mult_map = ({int(d["player_id"]): float(d["mult"]) for d in _fat_detail}
                     if ctx.fatigue_reweight else {})
    _rw = np.array([float(_pe.get(int(pids[i]), 1.0)) * _fat_mult_map.get(int(pids[i]), 1.0)
                    for i in range(len(pids))], dtype=float)
    _logrw = np.where(_rw > 0, np.log(np.clip(_rw, 1e-300, None)), -np.inf)
    if ctx.pwin_kind != "pool":
        cloud = np.asarray(cloud, dtype=float) + _logrw[:, None]
    if ctx.pwin_kind != "pool":
        samples.sizing_weights[:] = bo._elig_reweight(samples.sizing_weights, pids, _pe)
    if ctx.rank_floor is EXPLICIT_MASK:
        rf_mask = np.asarray(samples.tradeable_mask, dtype=bool)
    else:
        _rf_v = np.asarray(samples.vote_share_pred, dtype=float)
        rf_mask = np.zeros(_rf_v.size, dtype=bool)
        rf_mask[np.argsort(-_rf_v)[:ctx.rank_floor]] = True
    tradeable_now = np.array([pids[i] in yes_mids for i in range(len(pids))], dtype=bool)
    alloc_mask = rf_mask & tradeable_now
    if bo.RENORM_MODE != "baseline" and ctx.pwin_kind != "pool":
        from scripts.strategy.pricing import renorm_set
        from scripts.strategy.pricing import fp_point_loader
        _yesmap = {pids[i]: yes_mids.get(pids[i]) for i in range(len(pids))}
        _fp_vec = (fp_point_loader.fp_vector(conn, award, season, carried, pids)
                   if (bo.RENORM_MODE == "union_jspeak" and ctx.jsfloor) else None)
        if _fp_vec is not None:
            _fpm = np.isfinite(_fp_vec)
            _fp_vec[_fpm] = _fp_vec[_fpm] * _rw[_fpm]
        _vsp, _q, cloud, alloc_mask = renorm_set.apply(
            samples, cloud, rf_mask, tradeable_now, _yesmap, pids,
            mode=bo.RENORM_MODE, award=award, market_floor=bo.RENORM_MARKET_FLOOR,
            fp_point=_fp_vec)
        samples.vote_share_pred[:] = _vsp
        samples.sizing_weights[:] = _q

    _eval_snap_base = None
    if record:
        for i, pid in enumerate(pids):
            model_eval.append(dict(
                award=award, snapshot_date=str(day), frac=frac, player_id=pid,
                name=names.get(pid, str(pid)), model_yes=float(samples.vote_share_pred[i]),
                sizing_weight=float(samples.sizing_weights[i]),
                market_yes=float(yes_mids.get(pid, float("nan"))),
                in_contender=bool(alloc_mask[i]), target_usd=0.0, raw_kelly_usd=0.0,
                f_conc=1.0, prev_position_usd=0.0, radj=float("nan"),
                gate_pass="", stage_raw=_stage_prefatigue[i],
                stage_postfatigue=_stage_postfatigue[i], stage_postelig=_stage_postelig[i]))
        _eval_snap_base = len(model_eval) - len(pids)

    cost_curves = []
    for i, pid in enumerate(pids):
        cand = candidates[pid]
        cost_curves.append({
            "yes": (lambda s, c=cand, d=day: c.cost_curve(d, "yes").effective_price_at(s)),
            "no": (lambda s, c=cand, d=day: c.cost_curve(d, "no").effective_price_at(s))})

    a_prev = np.zeros(len(pids))
    for i, pid in enumerate(pids):
        p = ledger._pos.get(pid)
        if p is not None:
            a_prev[i] = p.outlay_eff if p.side == "YES" else -p.outlay_eff

    _abk = budget
    if budget_asof:
        _pri = [d for d in sorted(budget_asof) if str(d) <= str(day)]
        _abk = float(budget_asof[_pri[-1]]) if _pri else budget
    res = solve_award_v2(samples, cost_curves, portfolio_usd=budget, award_budget=_abk, kelly_fraction=1.0,
                         award=award, tradeable_mask=alloc_mask, seed=bo.SEED,
                         central_weights=bo.CENTRAL_WEIGHTS, a_prev=a_prev,
                         turnover_default=(bo.TURNOVER_DEFAULT if turnover is None else turnover),
                         n_restarts=n_restarts,
                         guard_top_k=ctx.guard_top_k)
    raw = np.asarray(res.raw_allocation, float)
    if record and warn_spread and res.restart_spread > 0.02 and verbose:
        print(f"  [{day}] WARN sizer restart_spread={res.restart_spread:.3f} (non-convexity)")

    _growth = float(getattr(res, "exp_log_growth", float("nan")))
    trad_idx = [i for i in range(len(pids)) if alloc_mask[i] and abs(raw[i]) > 1e-6]
    out = dict(empty=(not trad_idx), samples=samples, pids=pids, yes_mids=yes_mids,
               region_state=_region_state, target_by_pid={}, pid_to_idx={},
               deployed=0.0, growth=_growth, edge_pack=[])
    if not trad_idx:
        return out

    radj_list, psig_list, kelly_targets, diag_rows = [], [], [], []
    _edge_by_pid = {}
    for i in trad_idx:
        pid = pids[i]
        side = "yes" if raw[i] > 0 else "no"
        leg_price = yes_mids[pid] if side == "yes" else 1.0 - yes_mids[pid]
        size_intended = abs(raw[i])
        entry_cost = candidates[pid].cost_curve(day, side).cost_frac_at(size_intended)
        hist = np.diff(np.asarray(hist_logit[pid], float)) if len(hist_logit[pid]) > 1 else np.zeros(1)
        _pw_pool = samples.pool[:, i] if ctx.pwin_kind == "pool" else None
        radj, eedge, cvar, psig = bo._composite(
            fe, cloud, i, leg_price, entry_cost, vol_model, frac, hist, side,
            central_pwin=float(samples.vote_share_pred[i]), pw_pool=_pw_pool)
        radj_list.append(radj); psig_list.append(psig); kelly_targets.append(raw[i])
        diag_rows.append(dict(pid=pid, side=side, radj=radj))
        _edge_by_pid[pid] = (float(eedge), float(cvar))

    _scale_params = ScaleParams(kelly_fraction=1.0, s_soft_no=bo.CONC_S_SOFT_NO,
                                s_soft_yes=bo.CONC_S_SOFT_YES)
    if os.environ.get("USE_REGION") == "1":
        _conf = int(os.environ.get("REGION_CONFIRM", "2"))
        _hyst = float(os.environ.get("REGION_HYST", "1.0"))
        _mtf = float(os.environ.get("REGION_MIN_TRADE", "0.0"))
        target_by_pid, pid_to_idx, _region_state = _region.region_target_by_pid(
            samples, pids, trad_idx, raw, radj_list, psig_list, kelly_targets,
            yes_mids, candidates, ledger, day, budget, ceiling, _scale_params,
            _region_state, snapshot_id=carried, confirm_snapshots=_conf,
            hysteresis_mult=_hyst, open_hurdle=bo.HURDLE, fill_form=bo.FILL_FORM,
            fill_k=bo.FILL_K, min_fill=bo.MIN_FILL, min_trade_frac=_mtf)
    else:
        final_alloc, _fd = sizer_fill.size_positions(
            np.asarray(kelly_targets, float), np.asarray(radj_list, float),
            np.asarray(psig_list, float), hurdle=bo.HURDLE, min_fill=bo.MIN_FILL,
            fill_form=bo.FILL_FORM, fill_kwargs={"k": bo.FILL_K, "ceiling": ceiling})
        final_alloc = np.asarray(final_alloc, float)
        sc = scale_allocation(final_alloc, budget, _scale_params,
                              price_dispersion=None, player_ids=[pids[i] for i in trad_idx])
        final_alloc = np.asarray(sc.scaled, float)
        pid_to_idx = {pids[i]: i for i in range(len(pids))}
        target_by_pid = {pids[i]: float(final_alloc[j]) for j, i in enumerate(trad_idx)}

    out["region_state"] = _region_state
    out["target_by_pid"] = target_by_pid
    out["pid_to_idx"] = pid_to_idx
    out["deployed"] = float(sum(abs(v) for v in target_by_pid.values()))
    _pack = []
    for _pid, _notion in target_by_pid.items():
        _ec = _edge_by_pid.get(_pid)
        if _ec is None:
            continue
        _pack.append((abs(float(_notion)), _ec[0], _ec[1]))
    out["edge_pack"] = _pack

    if record:
        raw_by_pid = {pids[i]: float(raw[i]) for i in trad_idx}
        radj_by_pid = {d["pid"]: float(d["radj"]) for d in diag_rows}
        for row in model_eval[_eval_snap_base:]:
            pid = row["player_id"]
            row["target_usd"] = float(target_by_pid.get(pid, 0.0))
            row["raw_kelly_usd"] = raw_by_pid.get(pid, 0.0)
            r_ = radj_by_pid.get(pid, float("nan"))
            row["radj"] = r_
            row["gate_pass"] = "" if r_ != r_ else bool(r_ > bo.HURDLE)
    return out


def step_award_day(ctx, day):
    out = _award_core(ctx, day, ctx.portfolio_usd, ctx.region_state, record=True)
    ledger = ctx.ledger
    samples = out["samples"]
    yes_mids = out["yes_mids"]
    if out["empty"]:
        bo.self_mark(ledger, day, yes_mids, samples)
        ledger.record_deployed(day, out["deployed"])
        ctx.region_state = out["region_state"]
        return
    names = ctx.names
    candidates = ctx.candidates
    verbose = ctx.verbose
    target_by_pid = out["target_by_pid"]
    pid_to_idx = out["pid_to_idx"]

    for pid, i in pid_to_idx.items():
        leg = ledger._pos.get(pid)
        if leg is None:
            continue
        pwin_win = float(samples.vote_share_pred[i])
        pwin_leg = pwin_win if leg.side == "YES" else 1.0 - pwin_win
        ledger.set_model_context(pid, fv_yes_terminal=pwin_win, pwin_leg_terminal=pwin_leg)

    act_pids = set(target_by_pid) | {pid for pid in list(ledger._pos.keys()) if pid in yes_mids}
    for pid in act_pids:
        tgt = target_by_pid.get(pid, 0.0)
        i = pid_to_idx.get(pid)
        fv = float(samples.vote_share_pred[i]) if i is not None else None
        bo._rebalance_to(ledger, candidates[pid], day, pid, tgt, yes_mids[pid],
                         fv_yes=fv, name=names.get(pid), verbose=verbose)

    bo.self_mark(ledger, day, yes_mids, samples)
    ledger.record_deployed(day, out["deployed"])
    ctx.region_state = out["region_state"]


def finalize_award_daily(ctx):
    award = ctx.award
    conn = ctx.conn
    denom_diag = ctx.denom_diag
    last_day = ctx.last_day
    ledger = ctx.ledger
    model_eval = ctx.model_eval
    season = ctx.season
    verbose = ctx.verbose
    if ctx.pwin_kind == "pool":
        from scripts.backtest.stat_leader.stat_producers import stat_true_leader
        winner = stat_true_leader(conn, award, season)
    else:
        winner = bo.A_true_winner(conn, award, season)
    _n_before = len(ledger.position_log)
    ledger.settle(last_day, winner_player_id=winner)
    if verbose:
      for r in ledger.position_log[_n_before:]:
        print(f"  [{last_day}] {r['name']} SETTLE {r['side']} "
              f"shares={r['shares_settled']:.0f} @ {r['close_leg']:.3f} "
              f"pnl=${r['realised_pnl']:.0f} ({r['verdict']})")
    for row in model_eval:
        row["is_winner"] = int(row["player_id"] == winner)
    ledger.denom_diag = denom_diag
    return ledger, model_eval


def run_award_daily(conn, award, season, budget, ceiling=bo.KELLY_FILL_CEILING,
                    use_stub=False, verbose=True, turnover=None, n_restarts=6,
                    warn_spread=False, budget_asof=None):
    ctx = prepare_award_daily(conn, award, season, budget, ceiling=ceiling,
                              use_stub=use_stub, verbose=verbose, turnover=turnover,
                              n_restarts=n_restarts, warn_spread=warn_spread,
                              budget_asof=budget_asof)
    for _day in ctx.days:
        step_award_day(ctx, _day)
    return finalize_award_daily(ctx)


def _award_worker(payload):
    award, season, budget, use_stub, _out = payload
    from scripts.common.db import connect as _connect
    _conn = _connect("data/awards.db")
    _L, _ev = run_award_daily(_conn, award, season, budget, use_stub=use_stub, verbose=False)
    return (award, _L.trade_log, _L.position_log, _L.book_summary(), _ev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--awards", nargs="+", default=["MVP", "DPOY", "ROTY"])
    ap.add_argument("--budget", type=float, default=1000.0)
    ap.add_argument("--out", default="out/final_test_daily")
    ap.add_argument("--stub-cost", action="store_true")
    args = ap.parse_args()
    from scripts.common.db import connect
    conn = connect("data/awards.db")
    os.makedirs(args.out, exist_ok=True)
    trade_rows, pos_rows, book_rows, model_eval = [], [], [], []
    _payloads = [(aw, args.season, args.budget, args.stub_cost, args.out) for aw in args.awards]
    _results = {}
    if os.environ.get("SERIAL") or len(args.awards) == 1:
        for _p in _payloads:
            _aw, _tl, _pl, _bs, _ev = _award_worker(_p)
            _results[_aw] = (_tl, _pl, _bs, _ev)
    else:
        with _PPE(max_workers=len(args.awards)) as _ex:
            for _aw, _tl, _pl, _bs, _ev in _ex.map(_award_worker, _payloads):
                _results[_aw] = (_tl, _pl, _bs, _ev)
    for award in args.awards:
        _tl, _pl, b, ev = _results[award]
        trade_rows += _tl
        pos_rows += _pl
        book_rows.append(b)
        model_eval += ev
        print(f"\n=== {award} {args.season} DAILY (budget ${args.budget:.0f}) ===")
        print(f"  {award}: PnL ${b['realised_pnl']:.0f} ({b.get('return_pct', 0):.1f}%) "
              f"win_rate={b.get('win_rate', float('nan'))} "
              f"cost_drag=${b.get('sum_cost_drag', float('nan')):.0f} "
              f"path_residual=${b.get('sum_path_residual', float('nan')):.0f} "
              f"n_txn={b.get('n_transactions', 0)}")
    bo._dump_csv(os.path.join(args.out, f"trade_log_{args.season}.csv"), trade_rows)
    bo._dump_csv(os.path.join(args.out, f"position_log_{args.season}.csv"), pos_rows)
    bo._dump_csv(os.path.join(args.out, f"model_eval_{args.season}.csv"), model_eval)
    with open(os.path.join(args.out, f"book_summary_{args.season}.json"), "w") as f:
        json.dump(book_rows, f, indent=2, default=str)
    print(f"\nPOOLED PnL ${sum(b['realised_pnl'] for b in book_rows):.0f}   -> {args.out}")


if __name__ == "__main__":
    main()
