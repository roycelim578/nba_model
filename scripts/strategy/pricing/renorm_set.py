"""Contender-set renormalisation for the sized backtest (task 4).

The sizer keeps q (sizing_weights, mean_of_softmax) over the FULL field and only
pins out-of-set candidates to zero allocation. So probability mass on names the
model cannot back still lives in the wealth objective as field-wins states, where
every NO leg pays and every YES leg loses. That mass subsidises NO legs and
penalises the true winner's YES. This module lets the softmax normalise over a
chosen contender SET, which is more than odds-neutral: it changes the ABSOLUTE q
level the log-Kelly objective sizes against.

MODES (the set S over which the three model objects are renormalised):
  baseline      no renorm; caller keeps the full-field objects and S = rank_floor
                & tradeable (current behaviour).
  topn          S = rank_floor & tradeable. Pure sharpening: strips the dead tail,
                keeps the model's own contender view, no market coupling. KEEPS the
                longshot fade (a top-N name is tradeable whatever its market price).
  market        S = {market yes >= floor} (and scored by the model). Trusts the
                market's contender set entirely. SACRIFICES the longshot fade and
                the model-ahead-of-market riser.
  intersection  S = rank_floor & {market yes >= floor}. Tightest: only names the
                model AND market call live. Sacrifices the riser AND the longshot fade.
  union         S = (rank_floor | {market yes >= floor}) & tradeable. Keeps the
                riser (a top-N name the market has not caught) and the longshot fade.
  union_jspeak  S as union, PLUS the jspeak_floor favourite lift: the favourite's
                price is lifted toward the first-place model's point (fp_point),
                floored so FP may only add confidence and never hedge it down. The
                lift is applied to the SAME masked cloud, so vote_share_pred, the
                sizing weights and the per-draw pwin all inherit it coherently. If
                fp_point is None this degrades to plain union.

COHERENCE INVARIANT: the allocation set the caller uses is exactly S. You can never
trade mass you excluded from the denominator, so a name below the market floor in
market/intersection is neither counted nor traded, by construction.

EDGE CASES handled:
  - a market name the model never scored is not in pids, so it cannot enter S (no
    zero or NaN); it is simply dropped. The diagnostic counts these separately.
  - a pid with no finite cloud row is dropped from S (degenerate-score guard).
  - if the requested set is empty on a thin-market snapshot, S falls back to
    rank_floor & tradeable (a fallback the diagnostic counts).

All three objects are recomputed from ONE masked cloud so they cannot drift apart:
  vote_share_pred  softmax_of_mean over S (the edge object),
  sizing_weights   mean_of_softmax over S (the wealth object q),
  cloud            per-draw scores with out-of-S set to -inf, so forward_edge's
                   per_draw_pwin softmaxes over S too.

season = STARTING year. British English. No inline comments.
"""
from __future__ import annotations

import numpy as np

try:
    from scripts.strategy.sizing.soft_outcome import edge_weights, sizing_weights as _sizing_weights
except ImportError:  # pragma: no cover
    from soft_outcome import edge_weights, sizing_weights as _sizing_weights


NEG_INF = -1e9
VALID_MODES = ("baseline", "topn", "market", "intersection", "union", "union_jspeak")


def _market_mask(yes_by_pid: dict, pids: list, floor: float) -> np.ndarray:
    out = np.zeros(len(pids), dtype=bool)
    for i, pid in enumerate(pids):
        y = yes_by_pid.get(pid)
        out[i] = (y is not None) and (float(y) >= floor)
    return out


def resolve_set(mode: str, rf_mask: np.ndarray, tradeable_now: np.ndarray,
                market_mask: np.ndarray):
    """The contender set S for a mode, plus a flag for whether the fallback fired.
    Fallback (rank_floor & tradeable) protects thin-market snapshots where the
    requested set is empty."""
    rf = np.asarray(rf_mask, dtype=bool)
    tr = np.asarray(tradeable_now, dtype=bool)
    mk = np.asarray(market_mask, dtype=bool)
    if mode == "topn":
        S = rf & tr
    elif mode == "market":
        S = mk & tr
    elif mode == "intersection":
        S = rf & mk
    elif mode in ("union", "union_jspeak"):
        S = (rf | mk) & tr
    else:
        S = rf & tr
    fell_back = False
    if not S.any():
        S = rf & tr
        fell_back = True
    return S, fell_back


def apply(samples, cloud, rf_mask, tradeable_now, yes_by_pid, pids, *,
          mode, award, market_floor=0.02, fp_point=None):
    """Return (vote_share_pred, sizing_weights, masked_cloud, set_mask).

    cloud is the (ncand, K*M) score array (samples.sim as extracted in run_award).
    For baseline the inputs pass through unchanged and S = rank_floor & tradeable.
    The returned set_mask IS the allocation set (coherence invariant).
    fp_point (union_jspeak only): first-place point per candidate aligned to pids,
    NaN where the name was not scored by the FP fold.
    """
    if mode not in VALID_MODES:
        raise ValueError(f"unknown renorm mode {mode!r}; one of {VALID_MODES}")

    rf = np.asarray(rf_mask, dtype=bool)
    tr = np.asarray(tradeable_now, dtype=bool)
    cl = np.asarray(cloud, dtype=float)
    finite_row = np.isfinite(cl).any(axis=1)

    if mode == "baseline":
        return (np.asarray(samples.vote_share_pred, dtype=float),
                np.asarray(samples.sizing_weights, dtype=float),
                cl, rf & tr & finite_row)

    mk = _market_mask(yes_by_pid, pids, market_floor)
    tr_all = np.ones_like(tr)
    S_denom, _ = resolve_set(mode, rf, tr_all, mk)
    S_denom = S_denom & finite_row
    S = S_denom & tr

    masked = cl.copy()
    masked[~S_denom, :] = NEG_INF

    if mode == "union_jspeak":
        from scripts.strategy.pricing import jspeak_reshape
        fp = (np.asarray(fp_point, dtype=float) if fp_point is not None
              else np.full(len(pids), np.nan))
        masked, _p = jspeak_reshape.jspeak_reshape_masked(masked, S_denom, fp)

    draws = masked.T
    new_vsp = np.asarray(edge_weights(draws), dtype=float)
    new_q = np.asarray(_sizing_weights(draws), dtype=float)
    new_vsp[~S_denom] = 0.0
    new_q[~S_denom] = 0.0
    _pre = np.asarray(samples.vote_share_pred, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        _lift = np.where(_pre > 1e-6, new_vsp / _pre, 0.0)
    _hot = (new_vsp > 0.5) & (_lift > 3.0)
    if bool(_hot.any()):
        import logging
        _idx = int(np.argmax(np.where(_hot, new_vsp, 0.0)))
        logging.getLogger("renorm_set").warning(
            "mass-dump tripwire (%s): %d name(s) lifted >3x to share>0.5 "
            "(max %.3f from %.3f); check for an untradeable favourite this snapshot",
            award, int(_hot.sum()), float(new_vsp[_idx]), float(_pre[_idx]))
    return new_vsp, new_q, masked, S


def _selftest():
    rng = np.random.default_rng(0)
    ncand, draws = 12, 500
    cloud = rng.normal(size=(ncand, draws))
    pids = list(range(ncand))
    rf = np.zeros(ncand, dtype=bool)
    rf[:6] = True
    tr = np.ones(ncand, dtype=bool)
    yes = {i: (0.10 if i in (0, 1, 2, 9) else 0.005) for i in pids}

    class _S:
        vote_share_pred = np.full(ncand, 1.0 / ncand)
        sizing_weights = np.full(ncand, 1.0 / ncand)

    expect = {"baseline": None, "topn": 6, "market": 4, "intersection": 3,
              "union": 7, "union_jspeak": 7}
    for mode in VALID_MODES:
        fp = None
        if mode == "union_jspeak":
            fp = np.full(ncand, np.nan)
            fp[0] = 0.9  # a sharp first-place point on the favourite
        vsp, q, mc, S = apply(_S(), cloud, rf, tr, yes, pids,
                              mode=mode, award="MVP", market_floor=0.02, fp_point=fp)
        assert abs(vsp.sum() - 1.0) < 1e-9, (mode, vsp.sum())
        assert abs(q.sum() - 1.0) < 1e-9, (mode, q.sum())
        if mode != "baseline":
            assert np.allclose(vsp[~S], 0.0) and np.allclose(q[~S], 0.0), mode
            assert int(S.sum()) == expect[mode], (mode, int(S.sum()))
        print(f"  {mode}: |S|={int(S.sum())} vsp_sum={vsp.sum():.4f} q_sum={q.sum():.4f}")

    # union_jspeak must never hedge the favourite below plain union
    fp = np.full(ncand, np.nan); fp[0] = 0.95
    vu, _, _, Su = apply(_S(), cloud, rf, tr, yes, pids, mode="union",
                         award="MVP", market_floor=0.02)
    vj, _, _, Sj = apply(_S(), cloud, rf, tr, yes, pids, mode="union_jspeak",
                         award="MVP", market_floor=0.02, fp_point=fp)
    t = int(np.argmax(vu))
    assert vj[t] >= vu[t] - 1e-12, (vj[t], vu[t])
    print(f"  union_jspeak floor: favourite {vu[t]:.3f} -> {vj[t]:.3f} (lift, never below)")

    empty_yes = {i: 0.0 for i in pids}
    _, _, _, Sf = apply(_S(), cloud, rf, tr, empty_yes, pids,
                        mode="market", award="MVP", market_floor=0.02)
    assert int(Sf.sum()) == int((rf & tr).sum()), "empty-market fallback failed"
    _mp = np.full((ncand, draws), -2.0)
    _mp[0, :] = 3.0
    rf_mp = np.zeros(ncand, dtype=bool); rf_mp[:3] = True
    yes_mp = {i: (0.1 if i < 3 else 0.0) for i in pids}
    tr_full = np.ones(ncand, dtype=bool)
    _, q_full, _, _ = apply(_S(), _mp, rf_mp, tr_full, yes_mp, pids,
                            mode="union", award="MVP")
    tr_gap = tr_full.copy(); tr_gap[0] = False
    _, q_gap, _, S_gap = apply(_S(), _mp, rf_mp, tr_gap, yes_mp, pids,
                              mode="union", award="MVP")
    assert not S_gap[0], "untradeable favourite must leave the allocation set"
    assert q_gap[0] > 0.5, "favourite mass must be preserved, not dumped"
    assert q_gap[1] <= q_full[1] + 1e-9, (
        f"mass dumped onto survivor {q_full[1]:.3f}->{q_gap[1]:.3f}")
    print(f"  mass-preservation: survivor {q_full[1]:.3f}->{q_gap[1]:.3f}, "
          f"fav parked q={q_gap[0]:.3f}")
    print("renorm_set selftest OK")


if __name__ == "__main__":
    _selftest()
