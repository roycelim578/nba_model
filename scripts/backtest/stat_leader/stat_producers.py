"""Stat-leader engine producers: the pricejoin_fn and samples_fn an AwardSpec row
binds for the pts / reb / ast books. Written to Chat C's pinned shapes so the
generalised single-pass engine consumes them exactly like the voter producers.

pricejoin_fn = stat_daily_grid(conn, book, season) -> (feature_snaps, daily) with
daily = {day: (carried_snap, {player_id: yes_exec_price})}. It reuses Chat B's
resolve_markets / _legs_for_market / _series for stat identity and prices, and the
voter daily-grid mechanics (carry, D+1, LOCF) unchanged. feature_snaps is the
season's context snapshot schedule (the same snapshot_grid x availability set that
builds the MC ctx), so every carried snap is one the MC also scores.

samples_fn = stat_samples(conn, book, season, feature_snaps) -> {snap: StatRow}.
It runs build_stat_samples over the resolved market set, adapts StatSamples to the
shape the engine reads (pool from pwin_pool, tradeable_mask from listed_ids), and
LOCF-fills any feature_snap that build_stat_samples skipped (leader-guard early
snaps), so samples_by_snap[carried] always resolves. Each feature_snap gets its own
copied arrays, since the engine assigns vote_share_pred in place per day.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import numpy as np

from scripts.backtest.engine.backtest_pricejoin_daily import (
    _parse, _first_at_or_after, _last_at_or_before, _carry)
from scripts.backtest.stat_leader import stat_pricejoin as PJ
from scripts.backtest.stat_leader.stat_samples import build_stat_samples

_SNAP_SCHEDULE_SQL = (
    "SELECT DISTINCT a.snapshot_date FROM stg_nba_availability_asof a "
    "JOIN snapshot_grid g ON g.season=a.season AND g.snapshot_date=a.snapshot_date "
    "WHERE a.season=? AND g.snapshot_kind IN ('weekly','ratings') AND a.team_games_asof>0 "
    "ORDER BY a.snapshot_date")


def stat_daily_grid(conn, book, season, exec_lag_days=1, forward_tol_days=3,
                    locf_cap_days=4, locf_min_price=0.01, locf_max_price=0.99,
                    aliases=None):
    cur = conn.cursor()
    feature_snaps = [r[0] for r in cur.execute(_SNAP_SCHEDULE_SQL, (season,)).fetchall()]
    if not feature_snaps:
        raise SystemExit(f"no snapshot schedule for {book} {season}")

    _recs, resolved, _unres, _routing = PJ.resolve_markets(conn, book, season, aliases=aliases)
    series = {}
    for mid, pid in resolved.items():
        yes_cid, _no_cid = PJ._legs_for_market(conn, mid)
        if yes_cid is None:
            continue
        series[int(pid)] = PJ._series(conn, yes_cid)

    price_days = set()
    for rows in series.values():
        for ts, _px in rows:
            price_days.add(_parse(ts).date().isoformat())
    first_feat = feature_snaps[0]
    grid_days = sorted(d for d in price_days if d >= first_feat)

    daily = {}
    for day in grid_days:
        carried = _carry(feature_snaps, day)
        if carried is None:
            continue
        exec_from = _parse(day + "T00:00:00+00:00") + timedelta(days=exec_lag_days)
        tol = exec_from + timedelta(days=forward_tol_days)
        prices = {}
        for pid, rows in series.items():
            px, _ts = _first_at_or_after(rows, exec_from, tol)
            if px is None:
                cpx, _cts = _last_at_or_before(
                    rows, exec_from, exec_from - timedelta(days=locf_cap_days))
                if cpx is not None and locf_min_price < cpx < locf_max_price:
                    px = cpx
            if px is not None:
                prices[pid] = px
        if prices:
            daily[day] = (carried, prices)
    return feature_snaps, dict(sorted(daily.items()))


@dataclass
class StatRow:
    date: str
    frac: float
    player_ids: list
    vote_share_pred: np.ndarray
    sizing_weights: np.ndarray
    pool: np.ndarray
    tradeable_mask: np.ndarray


def _adapt(ss):
    listed = set(int(p) for p in ss.listed_ids)
    mask = np.array([int(pid) in listed for pid in ss.player_ids], dtype=bool)
    return StatRow(
        date=ss.date, frac=ss.frac, player_ids=[int(p) for p in ss.player_ids],
        vote_share_pred=np.array(ss.vote_share_pred, dtype=float),
        sizing_weights=np.array(ss.sizing_weights, dtype=float),
        pool=np.asarray(ss.pwin_pool, dtype=float),
        tradeable_mask=mask)


def _fill_schedule(produced, feature_snaps):
    """Map every feature_snap to a StatRow: the produced sample at that snap, else
    the nearest earlier produced (LOCF), else the first produced (leading edge).
    Each snap gets its own copied arrays via _adapt, so per-day in-place writes to
    vote_share_pred do not alias across snaps."""
    prod_snaps = sorted(produced)
    if not prod_snaps:
        raise SystemExit("build_stat_samples produced no snapshots")
    prod_set = set(prod_snaps)
    out, last = {}, None
    for snap in feature_snaps:
        if snap in prod_set:
            last = produced[snap]
        if last is not None:
            out[snap] = _adapt(last)
    first = produced[prod_snaps[0]]
    for snap in feature_snaps:
        if snap not in out:
            out[snap] = _adapt(first)
    return out


def stat_samples(conn, book, season, feature_snaps, aliases=None):
    _recs, resolved, _unres, _routing = PJ.resolve_markets(conn, book, season, aliases=aliases)
    mpids = tuple(sorted({int(p) for p in resolved.values()}))
    produced, _report = build_stat_samples(conn, book, season, market_player_ids=mpids)
    return _fill_schedule(produced, list(feature_snaps))


def stat_true_leader(conn, book, season):
    """Terminal settlement winner for a stat-leader book: the qualified season
    leader, the same realised_eff argmax build_stat_samples labels y_lead against,
    so the ledger settles on the leg the pool was calibrated to. realised_eff
    applies the qualifier max(gp, ceil(0.70*ftg)) and returns the resolved-market
    player_id space, matching the ledger's candidate keys, so no crosswalk is
    introduced. book is lowered to the producer key. Read-only. Reached only from
    finalize_award_daily, after prepare_award_daily has asserted the season is not
    sealed for the book."""
    from scripts.modelling.stat_leader import mc as MC
    from scripts.features.stat_leader import nodes as N
    b = book.lower()
    _counts, finals, _pos, _firstyr = N._load(conn, [int(season)])
    ftg = MC._load_ftg(conn, int(season))
    eff = MC.realised_eff(finals, ftg, int(season), b)
    if not eff:
        raise RuntimeError(f"stat_true_leader: no qualified leader for {b} {season}")
    return int(max(eff, key=eff.get))
