"""Shared-helper library for the backtest engine (no run path of its own).

Holds the pieces the run path imports: A_build_samples (pinned-model scoring and the
joint-sample builder), the sample-cloud and composite helpers, self_mark, _rebalance_to
and _close_leg (position stepping), _dump_csv, and the execution constants. The daily
loop lives in backtest_orchestrator_daily; the entry point is backtest_singlepass.
Capital is split across books by config.BOOK_WEIGHTS (shrunk skill-weighting), which
replaced the old equal-fixed placeholder.
"""
from __future__ import annotations
import os
import argparse
import glob
import numpy as np

from scripts.strategy.cost import pm_fees as _pm_fees
from scripts.features.eligibility.eligibility import eligibility_factors as _elig_factors, reweight_vector as _elig_reweight
from scripts.strategy.pricing.injury_miss_model import load_distribution as _load_elig_dist
_ELIG_DIST = _load_elig_dist()



AWARD_BUDGET_DEFAULT = 1000.0
KELLY_FILL_CEILING = 1.0
HURDLE = 0.05
RENORM_MODE = "union_jspeak"
JSFLOOR_AWARDS = set(
    x for x in os.environ.get("JSFLOOR_AWARDS", "MVP,ROTY,DPOY").split(",") if x
)
RENORM_MARKET_FLOOR = 0.02
INCUMBENCY_DIVERGENCE = False
MIN_FILL = 0.05
FILL_FORM = "inverse"
FILL_K = 3.0
COUPLING_RHO = 0.0
IMPACT_C = 0.5
COST_MODE = os.environ.get("COST_MODE", "impact")
LIQUIDITY_REGIME = os.environ.get("COST_REGIME", "normal")
DEPTH_DECAY = 0.6
FEE_CATEGORY = "sports"
COST_PARAMS_PATH = "data/cost_params.json"
EFF_CAP = 0.99
CVAR_QLOW = 0.05
M_SAMPLES = 200
COMPOSITE_DRAWS = 8000
SEED = 20260701

CENTRAL_WEIGHTS = "vote_share_pred"
GUARD_TOP_K = {"MVP": 1, "ROTY": 1, "DPOY": 1}
FATIGUE_REWEIGHT = {"DPOY": True, "MVP": True, "ROTY": False}
INCUMBENCY_DIVERGENCE = False
CONC_S_SOFT_NO = 0.12
CONC_S_SOFT_YES = 0.30
TURNOVER_DEFAULT = 0.10
MIN_TICKET_USD = 15.0


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


# =====================================================================
# ADAPTERS / SEAMS  (interfaces verified against the repo except where noted)
# =====================================================================
def A_connect(db_path="data/awards.db"):
    from scripts.common import db
    return db.connect(db_path)


def A_load_tradeable(conn, award, season):
    """backtest_pricejoin.load_tradeable -> {snap: SnapshotFrame}. Each
    SnapshotFrame.candidates is a list of CandidateLeg (player_id, yes_exec_price,
    no_exec_price, ...). Verified."""
    from scripts.backtest.engine.backtest_pricejoin import load_tradeable
    return load_tradeable(conn, award, season)


def A_snap_yes_mids(tradeable, snap):
    """{player_id: yes_mid} for a snapshot, from SnapshotFrame.candidates
    (yes_exec_price is the D+1 sealed YES price). Verified."""
    frame = tradeable[snap]
    return {int(c.player_id): float(c.yes_exec_price) for c in frame.candidates}


def A_build_samples(conn, award, season, snaps):
    """backtest_samples.build_samples -> {snap: JointSamples}. Requires the
    one-line sim patch (see _cloud). final-model glob picks the latest K200; for
    the SEALED 2025 run pin the exact 86fba1d6-family hashes instead."""
    from scripts.backtest.engine.backtest_samples import build_samples
    from scripts.common import config
    import pandas as pd
    bundle = pd.read_pickle(config.OOF_BUNDLE[award])
    bundle = bundle[bundle["season"] <= config.OOF_SEASON_CAP].copy()
    fm = config.FINAL_MODEL[award]
    from scripts.common import samples_cache
    return samples_cache.load_or_build(
        award, season, snaps, fm, config.OOF_BUNDLE[award], config.OOF_SEASON_CAP,
        M_SAMPLES, SEED,
        builder=lambda: build_samples(conn, award, season, snaps, bundle, fm, M=M_SAMPLES, seed=SEED))


def A_vol_model():
    """forward_vol.ForwardVolModel default path is out/forward_vol/forward_vol.pkl."""
    from scripts.strategy.forward_estimates.forward_vol import ForwardVolModel
    return ForwardVolModel()


class _StubCurve:
    """MECHANISM-CHECK ONLY. Flat sqrt-impact against a single near_touch, no
    per-bucket scale, so it does NOT punish size on cheap longshots. Matches the
    v5 shape effective = price*(1+bowl)*(1+c*sqrt(size/near_touch)) but with the
    real per-bucket near_touch replaced by a constant. Numbers are not PnL."""
    def __init__(self, price, near_touch=97.0, bowl=0.02, c=IMPACT_C):
        self.px = float(min(max(price, 1e-4), 1 - 1e-4))
        self.nt, self.bowl, self.c = near_touch, bowl, c

    def effective_price_at(self, size_usd):
        return self.px * (1 + self.bowl) * (1 + self.c * (max(float(size_usd), 0.0) / self.nt) ** 0.5)

    def cost_frac_at(self, size_usd):
        return self.effective_price_at(size_usd) / self.px - 1.0

    def exit_price_at(self, size_usd):
        return max(0.0, self.px * (1.0 - self.cost_frac_at(size_usd)))


class _StubCandidate:
    """MECHANISM-CHECK ONLY stand-in for the real backtest_candidate layer.
    Exposes cost_curve(date, side).effective_price_at / .cost_frac_at and a no-op
    change_net. price_map: {(player_id, date): yes_mid}."""
    def __init__(self, player_id, price_map):
        self.player_id = int(player_id)
        self.price_map = price_map
        self.net = 0.0

    def cost_curve(self, date, side):
        y = self.price_map.get((self.player_id, str(date)))
        if y is None:
            y = 0.5
        return _StubCurve(y if side == "yes" else 1.0 - y)

    def change_net(self, date, delta_signed_usd):
        self.net += float(delta_signed_usd)


class _RealCurve:
    """Cost curve backed by the frozen v5 CostModel + pm_fees. effective_price =
    price * (1 + bowl_impact_frac + taker_fee_frac). The NO leg is priced at its
    own token price (1 - yes) through the SAME CostModel, so it lands in its own
    bucket, identical to backtest_candidate.ComplementaryPrices."""
    def __init__(self, cost_model, price, frac_to_res):
        self.cm = cost_model
        self.px = float(min(max(price, 1e-4), 1 - 1e-4))
        self.f2r = float(frac_to_res)

    def cost_frac_at(self, size_usd):
        base = self.cm.cost_frac(size_usd=max(float(size_usd), 0.0), price=self.px,
                                 mode=COST_MODE, liquidity_regime=LIQUIDITY_REGIME,
                                 frac_to_resolution=self.f2r, c=IMPACT_C,
                                 depth_decay=DEPTH_DECAY, leg="entry")
        fee = _pm_fees.taker_fee_frac(self.px, FEE_CATEGORY)
        return base + fee

    def effective_price_at(self, size_usd):
        return self.px * (1.0 + self.cost_frac_at(size_usd))

    def exit_cost_frac_at(self, size_usd):
        base = self.cm.cost_frac(size_usd=max(float(size_usd), 0.0), price=self.px,
                                 mode=COST_MODE, liquidity_regime=LIQUIDITY_REGIME,
                                 frac_to_resolution=self.f2r, c=IMPACT_C,
                                 depth_decay=DEPTH_DECAY, leg="exit")
        fee = _pm_fees.taker_fee_frac(self.px, FEE_CATEGORY)
        return base + fee

    def exit_price_at(self, size_usd):
        return max(0.0, self.px * (1.0 - self.exit_cost_frac_at(size_usd)))


class _RealCandidate:
    """Real-cost stand-in for backtest_candidate.Candidate: cost_curve(date,side)
    off the frozen CostModel, no-op change_net (trade_ledger is the position
    truth). frac_to_resolution = 1 - season-elapsed-frac (early markets thin ->
    higher impact)."""
    def __init__(self, player_id, price_map, snap_frac, cost_model):
        self.player_id = int(player_id)
        self.price_map = price_map
        self.snap_frac = snap_frac
        self.cm = cost_model
        self.net = 0.0

    def cost_curve(self, date, side):
        y = self.price_map.get((self.player_id, str(date)), 0.5)
        px = y if side == "yes" else 1.0 - y
        f2r = max(0.0, 1.0 - float(self.snap_frac.get(str(date), 0.0)))
        return _RealCurve(self.cm, px, f2r)

    def change_net(self, date, delta_signed_usd):
        self.net += float(delta_signed_usd)


def A_candidates(conn, award, season, player_ids, tradeable, snap_frac, use_stub=False):
    """Returns {player_id: candidate}. DEFAULT is the real frozen CostModel oracle
    (data/cost_params.json). --stub-cost forces the flat mechanism-check stub."""
    price_map = {}
    for snap, frame in tradeable.items():
        for c in frame.candidates:
            price_map[(int(c.player_id), str(snap))] = float(c.yes_exec_price)
    if use_stub:
        print("  [WARN] MECHANISM-CHECK stub cost (flat near_touch); NOT tradeable PnL.")
        return {int(pid): _StubCandidate(pid, price_map) for pid in player_ids}
    from scripts.strategy.cost.cost_model import CostModel
    cm = CostModel.load(COST_PARAMS_PATH)
    return {int(pid): _RealCandidate(pid, price_map, snap_frac, cm) for pid in player_ids}


def A_true_winner(conn, award, season):
    """Resolve the actual winner via award_voting.won_flag (verified schema)."""
    cur = conn.execute(
        "SELECT player_id FROM award_voting WHERE award=? AND season=? AND won_flag=1 "
        "LIMIT 1", (award, int(season)))
    row = cur.fetchone()
    if row is None:
        raise RuntimeError(f"A_true_winner: no won_flag row for {award} {season}")
    return int(row[0])


# =====================================================================
# COMPOSITE RISK  (forward_edge; consumes the eta-widened cloud)
# =====================================================================
def _cloud(samples):
    sim = getattr(samples, "sim", None)
    if sim is None:
        sim = getattr(samples, "cloud", None)
    if sim is None:
        raise RuntimeError(
            "build_samples does not expose the eta-widened cloud. Add one line in "
            "the snapshot loop of build_samples: pass sim=sim into JointSamples "
            "(sim already exists as the reshaped (ncand, K*M) array). Do NOT fall "
            "back to raw_scores (bootstrap-only understates early dispersion).")
    return np.asarray(sim)


def _composite(fe, cloud, cand_idx, leg_price, entry_cost, vol_model, frac, history, side,
               central_pwin=None):
    rng = np.random.default_rng((SEED, int(cand_idx), 0 if side == "yes" else 1, int(round(float(leg_price) * 1e6)), int(round(float(frac) * 1e6))))
    edges = fe.composite_edge_draws(
        cloud.T, cand_idx, current_price=leg_price, entry_cost=entry_cost,
        vol_model=vol_model, frac=frac, history=history, side=side,
        n_draws=COMPOSITE_DRAWS, coupling_rho=COUPLING_RHO, central_pwin=central_pwin, rng=rng)
    read = fe.read_off(edges, q_low=CVAR_QLOW)
    psig = fe.price_dispersion(leg_price, vol_model, frac, history, n_draws=COMPOSITE_DRAWS, rng=rng)
    return read["risk_adjusted_edge"], read["expected_edge"], read["cvar_downside"], float(psig)


# =====================================================================
# ONE AWARD, ONE SEASON
# =====================================================================
def self_mark(ledger, snap, yes_mids, samples):
    ledger.record_mark(snap, {int(p): yes_mids.get(int(p)) for p in samples.player_ids
                              if yes_mids.get(int(p)) is not None})


def _rebalance_to(ledger, cand, snap, pid, target_usd, yes_mid, fv_yes, name=None, verbose=False):
    """Move the candidate toward the signed target notional, reasoning in per-leg
    space (not signed-notional-delta space). target_usd > 0 wants a YES leg of that
    size, < 0 a NO leg, == 0 flat. If we hold the opposite side, or the target is
    flat while we hold a leg, close the current leg first. Then resize the target
    leg: add on a positive gap, sell down on a negative gap, gated (on same-side
    resizes only) by a no-trade band ~ round-trip cost. Fills are priced off the
    stub cost curve here; with the real engine this drives change_net and reads the
    executed fill back instead."""
    pos = ledger._pos.get(pid)
    cur_side = pos.side if pos is not None else None
    cur_notional = pos.outlay_eff if pos is not None else 0.0
    target_side = "YES" if target_usd > 1e-9 else ("NO" if target_usd < -1e-9 else None)
    target_notional = abs(target_usd)
    mid_of = lambda lc: (yes_mid if lc == "yes" else 1.0 - yes_mid)

    if target_side is None:
        if cur_side is not None:
            _close_leg(ledger, cand, snap, pid, yes_mid)
        return

    if target_notional < MIN_TICKET_USD:
        return

    if cur_side is not None and cur_side != target_side:
        _close_leg(ledger, cand, snap, pid, yes_mid)
        pos, cur_side, cur_notional = None, None, 0.0

    side_lc = "yes" if target_side == "YES" else "no"
    delta = target_notional - cur_notional
    band = max(2.0 * cand.cost_curve(snap, side_lc).cost_frac_at(max(abs(delta), 1.0)) * max(abs(delta), 1.0),
               MIN_TICKET_USD)
    if abs(delta) < band:
        return

    if delta > 0:
        eff = cand.cost_curve(snap, side_lc).effective_price_at(delta)
        if eff >= EFF_CAP:
            return
        shares_delta = delta / eff
        cand.change_net(snap, delta if side_lc == "yes" else -delta)
        ledger.record_trade(snap, pid, target_side, shares_delta, eff, mid_of(side_lc), fv_yes=fv_yes)
        if verbose:
            _act = "OPEN" if pos is None else "ADD"
            print(f"  [{snap}] {name or pid} {_act} {target_side} ${delta:.0f} eff={eff:.3f}")
    else:
        reduce_usd = -delta
        vwap = pos.vwap_eff if pos is not None else 1.0
        eff = cand.cost_curve(snap, side_lc).exit_price_at(reduce_usd)
        shares_delta = -(reduce_usd / max(vwap, 1e-6))
        cand.change_net(snap, -reduce_usd if side_lc == "yes" else reduce_usd)
        ledger.record_trade(snap, pid, target_side, shares_delta, eff, mid_of(side_lc))
        if verbose:
            try:
                _still = any((getattr(p, "player_id", None) or (p.get("player_id") if isinstance(p, dict) else p)) == pid for p in ledger.open_positions())
            except Exception:
                _still = True
            _act = "TRIM" if _still else "CLOSE"
            print(f"  [{snap}] {name or pid} {_act} {target_side} ${reduce_usd:.0f} exit_eff={eff:.3f}")


def _close_leg(ledger, cand, snap, pid, yes_mid):
    pos = ledger._pos.get(pid)
    if pos is None:
        return
    side = "yes" if pos.side == "YES" else "no"
    eff = cand.cost_curve(snap, side).exit_price_at(pos.outlay_eff)
    mid_leg = yes_mid if side == "yes" else 1.0 - yes_mid
    cand.change_net(snap, -pos.outlay_eff if side == "yes" else pos.outlay_eff)
    ledger.record_trade(snap, pid, pos.side, -pos.shares, eff, mid_leg, action="REBALANCE")


def _dump_csv(path, rows):
    import csv
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)