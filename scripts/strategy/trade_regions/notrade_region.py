"""Unified transaction-cost no-trade region for the execution layer.

Replaces the three overlapping inertia doors (stateless edge-floor zero-snap in
sizer_fill.size_positions, the flat turnover toll in backtest_sizer_v2, and the
degenerate proportional band in backtest_orchestrator._rebalance_to) with one
state-aware rule.

The principle. Around the frictionless Kelly target a_star for a name, hold whatever
you currently hold while it sits inside a no-trade region, and when it falls outside,
trade only to the nearest boundary, never to a_star. Opening from flat, closing,
trimming and adding are all the same operation on different (current, target) pairs.
The sunk entry spread is never recharged to a held name, so a converging winner is
held rather than trimmed the instant its entry edge decays (the Castle defect).

The band width. The correct half-width is NOT the round-trip cost in dollars. It is the
round-trip cost fraction divided by the CURVATURE of expected log-growth at the optimum:

    w = c / k,   c = round-trip cost fraction,   k = -g''(a_star).

Derivation, single asset. With linear cost c*|a - a0| on top of log-growth g(a), the
subgradient optimality condition is |g'(a0)| <= c, so the no-trade region is
{ a : |g'(a)| <= c }. Linearising g' about the optimum (g'(a_star)=0) gives
|g''(a_star)| * |a - a_star| <= c, i.e. |a - a_star| <= c/k. A sharply peaked optimum
(confident bet, high k) has a narrow band and is tracked closely; a flat optimum
(marginal edge, low k) has a wide band and is left alone. This is exactly the desired
behaviour: a near-zero-EV position has low curvature and a target near zero, so the band
is wide and the position is held, because holding is free but trading is not.

Two-variance discipline is preserved. Outcome variance stays in the Kelly magnitude
(the raw target and the curvature k). Price variance stays in the dispersion-keyed fill
fraction, applied to scale the target BEFORE the region test. Transaction cost enters
only here, in the band width. No quantity is double counted.

British English throughout.
"""
from __future__ import annotations

import numpy as np
import os

_PORTFOLIO_SLACK = 0.0  # portfolio capital slack in [0,1], set by the shared-bankroll coordinator each day


def kelly_star_binary(p, b):
    """Frictionless Kelly fraction for a binary bet, win prob p at net decimal odds b.
    Returns the unconstrained optimum (may be negative, meaning the other side)."""
    p = float(p)
    b = float(b)
    if b <= 0:
        return 0.0
    return (p * (b + 1.0) - 1.0) / b


def loggrowth_deriv_binary(f, p, b):
    """First derivative of single-asset log-growth g(f) = p log(1+bf) + (1-p) log(1-f)."""
    f = float(f)
    denom_win = 1.0 + b * f
    denom_lose = 1.0 - f
    if denom_win <= 0 or denom_lose <= 0:
        return np.nan
    return p * b / denom_win - (1.0 - p) / denom_lose


def loggrowth_curvature_binary(f, p, b):
    """k = -g''(f) > 0 at the interior. Curvature of single-asset log-growth."""
    f = float(f)
    denom_win = 1.0 + b * f
    denom_lose = 1.0 - f
    if denom_win <= 0 or denom_lose <= 0:
        return np.inf
    return p * b * b / (denom_win * denom_win) + (1.0 - p) / (denom_lose * denom_lose)


def band_halfwidth(cost_frac, curvature, floor=1e-9):
    """No-trade half-width w = c / k in the same units as the position variable."""
    return float(cost_frac) / max(float(curvature), floor)


def project_to_region(target, current, halfwidth):
    """Trade-to-nearest-boundary. Hold current if it lies within halfwidth of target,
    otherwise move to the near edge of the region (target +/- halfwidth), never to
    target. Halfwidth of zero forces a full move to target (a collapsed region, used by
    force-flat lifecycle exits)."""
    target = float(target)
    current = float(current)
    w = max(float(halfwidth), 0.0)
    gap = current - target
    if abs(gap) <= w:
        return current
    return target + np.sign(gap) * w


def roundtrip_cost_fraction(cost_frac_at_fn, ref_notional, min_ref=1.0):
    """Round-trip cost fraction for a name, sourced from its real per-leg cost curve at a
    reference trade size. cost_frac_at_fn(size)->one-way fraction; round trip is twice the
    one-way fraction. ref_notional keys the size dependence (sqrt impact); we evaluate at
    the current position scale, floored at min_ref, because the cost of adjusting a
    position is on the order of its own size."""
    s = max(abs(float(ref_notional)), float(min_ref))
    one_way = float(cost_frac_at_fn(s))
    return 2.0 * one_way


def turnover_frac_from_curves(current, cost_curves, min_ref=1.0):
    """Per-name round-trip cost fraction vector for the A-path objective toll. current is
    signed dollars held; cost_curves[i] = {'yes': fn, 'no': fn} with fn(size)->one-way
    frac. The toll charges this fraction on |a - a_prev| in the log-wealth objective,
    replacing the flat turnover_default with a real, liquidity-aware, per-name figure."""
    n = len(cost_curves)
    out = np.zeros(n, dtype=float)
    cur = np.asarray(current, dtype=float)
    for i in range(n):
        side = "yes" if cur[i] >= 0 else "no"
        fn = cost_curves[i][side]
        out[i] = roundtrip_cost_fraction(lambda s, fn=fn: fn(s), abs(cur[i]), min_ref=min_ref)
    return out


def _fill_fraction(price_sigma, form="inverse", k=3.2, ceiling=1.0):
    s = float(price_sigma)
    if form == "inverse":
        f = 1.0 / (1.0 + k * s)
    elif form == "exp":
        f = float(np.exp(-k * s))
    elif form == "linear":
        f = 1.0 - k * s
    else:
        raise ValueError(f"unknown fill form {form!r}")
    return float(np.clip(ceiling * f, 0.0, ceiling))


def size_positions_region(
    raw_targets,
    current,
    cloud_pwin,
    market_prices,
    cost_frac_at_fns,
    price_sigmas,
    budget,
    radj=None,
    open_hurdle=0.05,
    hysteresis_mult=1.0,
    fill_form="inverse",
    fill_k=3.2,
    fill_ceiling=1.0,
    min_fill=0.05,
    force_flat=None,
    pids=None,
    reversal_state=None,
    confirm_snapshots=2,
    reversal_margin=1.0,
    snapshot_id=None,
    apply_fill=True,
    band_opens=False,
    min_trade_frac=0.0,
):
    """State-aware replacement for sizer_fill.size_positions.

    Composition, once each, no double counting:
      1. fill scale (price variance)      scaled = raw_target * fill_fraction(price_sigma)
      2. open-only risk admission (radj)  refuse to OPEN a flat name whose risk-adjusted
                                           edge is below open_hurdle; NEVER applied to a
                                           held name (this is what removes the early-trim)
      3. no-trade region (transaction cost) new = project_to_region(scaled, current, w)
                                           with w = hysteresis_mult * c / k
      4. force-flat lifecycle override      collapses w to 0 so the name trades fully to 0

    raw_targets      signed dollars per candidate from the frictionless coupled Kelly solve
    current          signed dollars currently held per candidate (0 == flat)
    cloud_pwin       central win probability per candidate (drives curvature)
    market_prices    per-candidate YES mid in (0,1) (drives net odds b)
    cost_frac_at_fns per candidate, a callable size->one-way cost fraction on the traded leg
    price_sigmas     price dispersion per candidate (fill key)
    budget           the capital frame (dollars) against which Kelly fractions f = a/budget
                     and hence curvature and band width are computed; must be positive
    radj             optional risk-adjusted edge per candidate (open-only admission)
    force_flat       optional bool mask; True collapses the region and forces the name flat
    pids             optional per-candidate ids to key reversal_state stably across snapshots
                     (candidate ordering can change snapshot to snapshot); defaults to index
    reversal_state   dict id->consecutive-reversal count, carried across snapshots by the
                     caller; mutated in place and returned. None starts empty
    confirm_snapshots a held side is only reversed (position crosses zero to the opposite
                     leg) after the flipped target has persisted this many consecutive
                     snapshots. Default 2: a one-snapshot flip that reverts never trades.
                     Set to 1 to recover the pure magnitude-only region
    reversal_margin  dollar margin below which an opposite-side target is treated as a
                     trim-to-flat, not a genuine reversal (avoids counting tiny stubs)
    snapshot_id      identity of the current FEATURE (fair-value) snapshot. When given, the
                     reversal streak advances only when this id changes, so persistence is
                     measured on the fair-value clock. In the daily book fair value is
                     carried weekly and only changes at feature-snapshot boundaries, so pass
                     the carried snapshot id here; an intra-week price-driven flip is then
                     held (never advances the streak) rather than acted on, and only a
                     fair-value reversal that persists confirm_snapshots feature snapshots
                     executes. Pass None (weekly orchestrator, one call per snapshot) to
                     advance every call.

    band_opens       when False (default), a flat name opens straight to the intended target,
                     gated only by the radj open admission, matching the control's entry
                     behaviour; the band and reversal persistence then govern only held
                     positions. Set True for the Davis-Norman open-smaller-than-target
                     behaviour, which suppresses small entries and changes entry decisions.
    apply_fill       when False, raw_targets is the already-shaped intended target (caller
                     has applied fill and concentration/tail scaling); the region then only
                     applies the band, persistence and open admission.

    The persistence dimension is scoped to REVERSALS only (the census churn driver: a large
    clean re-rank flips the sign, clears the cost band, and reverts next snapshot). Same-side
    opens, adds and trims are governed by the cost/curvature band alone and are not delayed.
    A lifecycle force_flat bypasses persistence entirely.

    Returns (new_positions, diagnostics, reversal_state)."""
    raw = np.asarray(raw_targets, dtype=float)
    cur = np.asarray(current, dtype=float)
    pwin = np.asarray(cloud_pwin, dtype=float)
    px = np.asarray(market_prices, dtype=float)
    sig = np.asarray(price_sigmas, dtype=float)
    n = raw.size
    B = max(float(budget), 1.0)
    radj = np.full(n, np.inf) if radj is None else np.asarray(radj, dtype=float)
    force_flat = np.zeros(n, dtype=bool) if force_flat is None else np.asarray(force_flat, dtype=bool)
    reversal_state = {} if reversal_state is None else reversal_state
    keys = list(pids) if pids is not None else list(range(n))
    _ASYM = os.environ.get("ASYM_TRIM") == "1"
    _ASYM_UP = float(os.environ.get("ASYM_TRIM_UP", "3.0"))
    _ASYM_FLAT = float(os.environ.get("ASYM_TRIM_FLAT", "2.0"))
    _ASYM_DOWN = float(os.environ.get("ASYM_TRIM_DOWN", "1.0"))
    _ASYM_EPS = float(os.environ.get("ASYM_FV_EPS", "0.01"))
    _SLACK_KAPPA = float(os.environ.get("SLACK_KAPPA", "2.0"))
    _EDGE_NOISE = float(os.environ.get("EDGE_NOISE", "0.02"))
    _PX_MIN = float(os.environ.get("PX_MIN", "0.05"))
    _PX_MAX = float(os.environ.get("PX_MAX", "0.95"))
    min_trade_frac = max(float(min_trade_frac), 0.0)
    _BANDMODE = os.environ.get("REGION_BAND_MODE", "curvature")  # _REFORM_BAND_MARKER
    _BAND_VOL_MULT = float(os.environ.get("REGION_BAND_VOL_MULT", "0.05"))
    _BAND_SIGMA_REF = float(os.environ.get("REGION_BAND_SIGMA_REF", "0.12"))

    out = np.zeros(n, dtype=float)
    diag = []
    for i in range(n):
        fill = _fill_fraction(sig[i], form=fill_form, k=fill_k, ceiling=fill_ceiling) if apply_fill else 1.0
        floored = apply_fill and (fill < min_fill)
        scaled = 0.0 if floored else raw[i] * fill

        is_flat = abs(cur[i]) < 1e-9
        if is_flat:
            _open_scale = float(np.clip(radj[i] / open_hurdle, 0.0, 1.0)) if np.isfinite(radj[i]) else 1.0
            scaled = scaled * _open_scale
            refused_open = (_open_scale <= 0.0)
        else:
            refused_open = False

        pr = float(np.clip(px[i], 1e-4, 1.0 - 1e-4))
        if (pr < _PX_MIN or pr > _PX_MAX) and abs(scaled) > abs(cur[i]):
            scaled = cur[i]  # outside tradeable price band: hold, never open or add
        if abs(scaled) > 1e-9:
            side_sign = 1.0 if scaled > 0 else -1.0
        elif not is_flat:
            side_sign = 1.0 if cur[i] > 0 else -1.0
        else:
            side_sign = 1.0
        if side_sign >= 0:
            p_side, b = float(pwin[i]), (1.0 - pr) / pr
        else:
            p_side, b = 1.0 - float(pwin[i]), pr / (1.0 - pr)
        f_mag = float(np.clip(abs(scaled) / B, 0.0, 0.999))
        k = loggrowth_curvature_binary(f_mag, p_side, b)
        c = roundtrip_cost_fraction(lambda s, i=i: cost_frac_at_fns[i](s),
                                    max(abs(cur[i]), abs(scaled)))
        if _BANDMODE == "vol":
            _w_price = _BAND_VOL_MULT * (float(sig[i]) / max(_BAND_SIGMA_REF, 1e-9))
            w_frac = hysteresis_mult * max(_w_price, float(c))
            w_usd = w_frac * B
        else:
            w_frac = hysteresis_mult * band_halfwidth(c, k)
            w_usd = w_frac * B
        if _ASYM:
            _ki = keys[i]
            if snapshot_id is not None and reversal_state.get((_ki, "_fvsnap")) != snapshot_id:
                _pv = reversal_state.get((_ki, "_pwin"))
                reversal_state[(_ki, "_fvdir")] = 0.0 if _pv is None else float(pwin[i]) - float(_pv)
                reversal_state[(_ki, "_pwin")] = float(pwin[i])
                reversal_state[(_ki, "_fvsnap")] = snapshot_id
            _fvd = float(reversal_state.get((_ki, "_fvdir"), 0.0)) * side_sign
            _is_trim = (abs(cur[i]) > 1e-9 and abs(scaled) < abs(cur[i])
                        and (np.sign(scaled) == np.sign(cur[i]) or abs(scaled) <= 1e-9))
            if _is_trim:
                if _fvd >= _ASYM_EPS:
                    w_usd = w_usd * _ASYM_UP
                elif _fvd > -_ASYM_EPS:
                    w_usd = w_usd * _ASYM_FLAT
                else:
                    w_usd = w_usd * _ASYM_DOWN

        if _PORTFOLIO_SLACK > 0.0 and abs(cur[i]) > 1e-9 and abs(scaled) < abs(cur[i]):
            _leg_px_side = pr if side_sign >= 0 else (1.0 - pr)
            _edge_side = p_side - _leg_px_side
            if _edge_side > -_EDGE_NOISE:
                w_usd = w_usd * (1.0 + _SLACK_KAPPA * float(_PORTFOLIO_SLACK))
        if force_flat[i]:
            new = project_to_region(0.0, cur[i], 0.0)
        elif is_flat and not band_opens:
            new = scaled
        else:
            new = project_to_region(scaled, cur[i], w_usd)

        key = keys[i]
        streak = int(reversal_state.get(key, 0))
        if snapshot_id is not None:
            fresh = reversal_state.get((key, "_snap")) != snapshot_id
            reversal_state[(key, "_snap")] = snapshot_id
        else:
            fresh = True
        held = abs(cur[i]) > 1e-9
        is_reversal = (not force_flat[i]) and held and \
            (np.sign(scaled) == -np.sign(cur[i])) and (abs(scaled) > reversal_margin)
        reversal_pending = False
        if force_flat[i]:
            streak = 0
        elif is_reversal:
            if fresh:
                streak += 1
            if streak < confirm_snapshots:
                new = cur[i]
                reversal_pending = True
            else:
                streak = 0
                new = scaled
        else:
            if fresh:
                streak = 0
        reversal_state[key] = streak

        reversal_committed = is_reversal and not reversal_pending
        floor_i = min_trade_frac * abs(cur[i])
        below_min = (floor_i > 0.0) and (not force_flat[i]) and \
            (not reversal_committed) and (0.0 < abs(new - cur[i]) < floor_i)
        if below_min:
            new = cur[i]

        out[i] = new
        diag.append(dict(
            cand=i, raw=float(raw[i]), fill=fill, scaled=float(scaled),
            current=float(cur[i]), curvature=float(k), cost_frac=float(c),
            band_usd=float(w_usd), refused_open=bool(refused_open),
            gated_by_fill=bool(floored), force_flat=bool(force_flat[i]),
            reversal_pending=bool(reversal_pending), reversal_streak=int(streak),
            new=float(new), traded=bool(abs(new - cur[i]) > 1e-9), below_min=bool(below_min)))
    return out, diag, reversal_state
