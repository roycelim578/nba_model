"""Adapter between the orchestrator's per-snapshot data and notrade_region, plus a churn
comparison helper for the 2024 A/B.

Integration is a single swap in backtest_orchestrator.py. Where the current code does

    final_alloc, fill_diag = sizer_fill.size_positions(
        np.asarray(kelly_targets, float), np.asarray(radj_list, float),
        np.asarray(psig_list, float), hurdle=HURDLE, min_fill=MIN_FILL,
        fill_form=FILL_FORM, fill_kwargs={"k": FILL_K, "ceiling": ceiling})

replace with a call to region_targets below, passing the CURRENT signed holdings read from
the ledger, and then move each name to its returned target with NO second band in
_rebalance_to (delete the band test there; the region already owns inertia). The stateless
edge-floor zero-snap is gone: radj is passed only as an open-only admission filter.

Create reversal_state = {} ONCE per book before the snapshot loop (not per snapshot) and
pass it in each snapshot, reassigning it from the return. This carries the reversal-
persistence count across snapshots so a one-snapshot edge-sign flip that reverts never
trades; only a reversal that persists confirm_snapshots snapshots executes. Same-side adds
and trims are never delayed.

Confirm the ledger accessor (current_from_ledger) against live code; the mirror shows
ledger._pos[pid].side and .outlay_eff, which may have moved.

British English, no inline comments.
"""
from __future__ import annotations

import numpy as np

try:
    from scripts.strategy.trade_regions import notrade_region as ntr
except ImportError:
    import notrade_region as ntr


def current_from_ledger(ledger, pids):
    """Signed dollar notional currently held per pid: +outlay for a YES leg, -outlay for a
    NO leg, 0 if flat. Confirm attribute names against live ledger."""
    out = np.zeros(len(pids), dtype=float)
    for j, pid in enumerate(pids):
        pos = ledger._pos.get(int(pid))
        if pos is None:
            continue
        out[j] = pos.outlay_eff if pos.side == "YES" else -pos.outlay_eff
    return out


def region_targets(kelly_targets, current, cloud_pwin, market_prices, cost_frac_at_fns,
                   price_sigmas, budget, radj, pids, reversal_state, open_hurdle=0.05,
                   hysteresis_mult=1.0, fill_form="inverse", fill_k=3.2, fill_ceiling=1.0,
                   min_fill=0.05, confirm_snapshots=2, force_flat=None, snapshot_id=None):
    """Thin pass-through to notrade_region.size_positions_region, kept so the orchestrator
    imports one adapter. cost_frac_at_fns[i](size)->one-way cost fraction on name i's traded
    leg; build from candidates[pid].cost_curve(snap, side).cost_frac_at.

    reversal_state is a dict id->streak that MUST be created once per book (per season, per
    award) BEFORE the snapshot loop and passed in every snapshot, then reassigned from the
    return, so the persistence dimension sees the same name across snapshots even as the
    candidate ordering changes. pids keys that state stably.

    snapshot_id is the carried FEATURE snapshot for this call. In the daily book pass the
    carried_snap so reversal persistence is measured on the weekly fair-value clock, not per
    trading day. In the weekly book leave it None.

    Returns (new_positions, diagnostics, reversal_state)."""
    return ntr.size_positions_region(
        raw_targets=kelly_targets, current=current, cloud_pwin=cloud_pwin,
        market_prices=market_prices, cost_frac_at_fns=cost_frac_at_fns,
        price_sigmas=price_sigmas, budget=budget, radj=radj, open_hurdle=open_hurdle,
        hysteresis_mult=hysteresis_mult, fill_form=fill_form, fill_k=fill_k,
        fill_ceiling=fill_ceiling, min_fill=min_fill, force_flat=force_flat,
        pids=pids, reversal_state=reversal_state, confirm_snapshots=confirm_snapshots,
        snapshot_id=snapshot_id)


def region_target_by_pid(samples, pids, trad_idx, raw, radj_list, psig_list, kelly_targets,
                         yes_mids, candidates, ledger, snap, budget, ceiling, scale_params,
                         reversal_state, snapshot_id, confirm_snapshots=2, hysteresis_mult=1.0,
                         open_hurdle=0.05, fill_form="inverse", fill_k=3.0, min_fill=0.05,
                         band_opens=False, min_trade_frac=0.0):
    """Produce target_by_pid for the daily loop under the no-trade region, replacing the
    sizer_fill gate and the degenerate rebalance band while preserving fill and the
    concentration/tail shrink.

    Pipeline, matching the control path up to the intended target then swapping only the
    inertia:
      1. fill fraction on the traded-name raw Kelly (price variance), no radj hurdle gate
      2. scale_allocation (concentration + tail) to the intended target, exactly as control
      3. region over the UNION of traded and currently-held names: a held name the fresh
         solve dropped gets target zero and a region decision (trim an abandoned loser, hold
         a wide-band converging winner), not a blanket close or blanket hold
      4. NO-leg-aware curvature and snapshot_id=carried so persistence sits on the weekly
         fair-value clock

    Returns (target_by_pid, pid_to_idx, reversal_state)."""
    from scripts.strategy.sizing.size_scaling import scale_allocation
    from scripts.strategy.trade_regions import notrade_region as _ntr

    pid_to_idx = {pids[i]: i for i in range(len(pids))}

    fills = np.array([_ntr._fill_fraction(psig_list[j], form=fill_form, k=fill_k, ceiling=ceiling)
                      for j in range(len(trad_idx))], dtype=float)
    filled = np.array([0.0 if fills[j] < min_fill else kelly_targets[j] * fills[j]
                       for j in range(len(trad_idx))], dtype=float)
    sc = scale_allocation(filled, budget, scale_params, price_dispersion=None,
                          player_ids=[pids[i] for i in trad_idx])
    intended_trad = np.asarray(sc.scaled, float)

    trad_pid_set = {pids[i] for i in trad_idx}
    intended_by_pid = {pids[trad_idx[j]]: float(intended_trad[j]) for j in range(len(trad_idx))}
    radj_by_pid = {pids[trad_idx[j]]: float(radj_list[j]) for j in range(len(trad_idx))}

    held_pids = [int(p) for p in ledger._pos.keys() if int(p) in yes_mids and int(p) in pid_to_idx]
    union = [pids[i] for i in trad_idx] + [p for p in held_pids if p not in trad_pid_set]
    if not union:
        return {}, pid_to_idx, reversal_state

    u_raw, u_cur, u_pw, u_mp, u_cf, u_radj = [], [], [], [], [], []
    for pid in union:
        idx = pid_to_idx[pid]
        leg = ledger._pos.get(pid)
        cur = (leg.outlay_eff if leg.side == "YES" else -leg.outlay_eff) if leg is not None else 0.0
        tgt = intended_by_pid.get(pid, 0.0)
        side = "yes" if (tgt > 1e-9 or (abs(tgt) <= 1e-9 and cur >= 0)) else "no"
        u_raw.append(tgt)
        u_cur.append(cur)
        u_pw.append(float(samples.vote_share_pred[idx]))
        u_mp.append(float(yes_mids[pid]))
        u_cf.append((lambda s, _c=candidates[pid], _s=side: _c.cost_curve(snap, _s).cost_frac_at(s)))
        u_radj.append(radj_by_pid.get(pid, float("inf")))

    new, _diag, reversal_state = ntr.size_positions_region(
        np.asarray(u_raw, float), np.asarray(u_cur, float), np.asarray(u_pw, float),
        np.asarray(u_mp, float), u_cf, np.zeros(len(union), float), float(budget),
        radj=np.asarray(u_radj, float), open_hurdle=open_hurdle, hysteresis_mult=hysteresis_mult,
        min_fill=0.0, pids=union, reversal_state=reversal_state,
        confirm_snapshots=confirm_snapshots, snapshot_id=snapshot_id, apply_fill=False,
        band_opens=band_opens, min_trade_frac=min_trade_frac)

    target_by_pid = {union[k]: float(new[k]) for k in range(len(union))}
    return target_by_pid, pid_to_idx, reversal_state


def churn_summary(trades_df):
    """Given a trade ledger dataframe with columns including snap, pid, action, and a signed
    notional/shares change, return the churn metrics for the A/B table. path_residual is the
    running mark-to-market give-back; we surface transaction count and gross traded notional,
    which are the direct churn levers. PnL is read from the book elsewhere."""
    n_trades = int(len(trades_df))
    gross_notional = float(np.abs(trades_df.get("delta_usd", trades_df.get("shares", 0.0))).sum())
    per_name = trades_df.groupby("pid").size()
    return dict(transactions=n_trades, gross_traded_usd=gross_notional,
                names_touched=int(per_name.size),
                median_trades_per_name=float(per_name.median()) if per_name.size else 0.0)


def compare_books(label_a, summary_a, pnl_a, path_resid_a,
                  label_b, summary_b, pnl_b, path_resid_b):
    """Print the 2024 churn payoff table: current three-door mechanism versus the region.
    The payoff is fewer transactions and smaller path_residual at equal or better PnL."""
    rows = [
        ("transactions", summary_a["transactions"], summary_b["transactions"]),
        ("gross_traded_usd", round(summary_a["gross_traded_usd"], 1),
         round(summary_b["gross_traded_usd"], 1)),
        ("path_residual", round(path_resid_a, 1), round(path_resid_b, 1)),
        ("pnl", round(pnl_a, 1), round(pnl_b, 1)),
    ]
    w = max(len(label_a), len(label_b), 10)
    print(f"{'metric':<20}{label_a:>{w+2}}{label_b:>{w+2}}{'delta':>{w+2}}")
    for name, a, b in rows:
        print(f"{name:<20}{a:>{w+2}}{b:>{w+2}}{round(b - a, 1):>{w+2}}")
