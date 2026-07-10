"""Daily trade grid for the sized backtest / live path.

Decouples the three cadences the daily loop needs:
  - model score + fatigue: HELD from the most recent feature snapshot (both read
    feature_stats_asof, which exists only at the ~20 model_predictions snapshots).
  - price: DAILY, from pm_prices at the D+1 seal.
  - eligibility: DAILY (the caller passes the daily date to eligibility_factors,
    which reads injuries and game-logs as-of any date).

load_daily_grid returns, per tradeable day D in the season window:
    (carried_feature_snap, {player_id: yes_exec_price_at_D+1})
plus the ordered list of feature snapshots (for build_samples) and the carry map.
The D+1 seal is identical to load_tradeable: the first price stamped strictly at
or after D+1 00:00 UTC, within a forward tolerance.

British English. No inline comments.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta

MODEL_VERSION_LIKE = "%{award}%"


def _parse(ts):
    return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))


def _first_at_or_after(rows, cutoff_dt, tol_dt):
    for ts, px in rows:
        t = _parse(ts)
        if t >= cutoff_dt:
            return (float(px), ts) if t <= tol_dt else (None, None)
    return (None, None)


def _last_at_or_before(rows, cutoff_dt, floor_dt):
    for ts, px in reversed(rows):
        t = _parse(ts)
        if t <= cutoff_dt:
            return (float(px), ts) if t >= floor_dt else (None, None)
    return (None, None)


def _pairing(cur, award, season):
    markets = [r[0] for r in cur.execute(
        "SELECT market_id FROM pm_markets WHERE award=? AND season=?",
        (award, season)).fetchall()]
    if not markets:
        return {}
    qm = ",".join("?" * len(markets))
    crows = cur.execute(
        f"SELECT market_id, candidate_id, candidate_name, player_id, outcome, outcome_side "
        f"FROM pm_candidates WHERE market_id IN ({qm})", markets).fetchall()

    def side_of(oside, outcome):
        s = (oside or outcome or "").strip().upper()
        return "YES" if s == "YES" else ("NO" if s == "NO" else "?")

    pair = {}
    for _mid, cid, cname, pid, outcome, oside in crows:
        if pid is None:
            continue
        d = pair.setdefault(int(pid), {"name": cname})
        d[side_of(oside, outcome)] = cid
    return {p: d for p, d in pair.items() if "YES" in d and "NO" in d}


def _price_series(cur, pair):
    all_cids = [d["YES"] for d in pair.values()]
    if not all_cids:
        return {}
    qc = ",".join("?" * len(all_cids))
    prows = cur.execute(
        f"SELECT candidate_id, timestamp, hist_price FROM pm_prices "
        f"WHERE price_type='history_agg' AND candidate_id IN ({qc}) "
        f"ORDER BY candidate_id, timestamp", all_cids).fetchall()
    series = {}
    for cid, ts, px in prows:
        series.setdefault(cid, []).append((ts, px))
    return series


def _carry(feature_snaps, day):
    """Most recent feature snapshot on or before day; None if none precedes it."""
    prior = [s for s in feature_snaps if s <= day]
    return prior[-1] if prior else None


def load_daily_grid(conn, award, season, exec_lag_days=1, forward_tol_days=3, locf_cap_days=4, locf_min_price=0.01, locf_max_price=0.99):
    """Return (feature_snaps, daily) where daily is an ordered dict
    {day: (carried_feature_snap, {player_id: yes_exec_price})}. Days before the
    first feature snapshot are skipped (no model score to carry). Days with no
    valid D+1 price for any candidate are skipped."""
    cur = conn.cursor()
    mv_like = MODEL_VERSION_LIKE.format(award=award)
    feature_snaps = sorted({r[0] for r in cur.execute(
        "SELECT DISTINCT snapshot_date FROM model_predictions "
        "WHERE model_version LIKE ? AND season=?", (mv_like, season)).fetchall()})
    if not feature_snaps:
        raise SystemExit(f"no model_predictions for {award} {season}")

    pair = _pairing(cur, award, season)
    series = _price_series(cur, pair)

    price_days = set()
    for rows in series.values():
        for ts, _px in rows:
            price_days.add(_parse(ts).date().isoformat())
    first_feat = feature_snaps[0]
    grid_days = sorted(d for d in price_days if d >= first_feat)

    daily = {}
    _locf_n = 0
    for day in grid_days:
        carried = _carry(feature_snaps, day)
        if carried is None:
            continue
        cutoff = _parse(day + "T00:00:00+00:00")
        exec_from = cutoff + timedelta(days=exec_lag_days)
        tol = exec_from + timedelta(days=forward_tol_days)
        prices = {}
        for pid, d in pair.items():
            px, ts = _first_at_or_after(series.get(d["YES"], []), exec_from, tol)
            if px is None:
                _cpx, _cts = _last_at_or_before(
                    series.get(d["YES"], []), exec_from,
                    exec_from - timedelta(days=locf_cap_days))
                if _cpx is not None and locf_min_price < _cpx < locf_max_price:
                    px, ts = _cpx, _cts
                    _locf_n += 1
            if px is not None:
                prices[pid] = px
        if prices:
            daily[day] = (carried, prices)
    if _locf_n:
        import logging
        logging.getLogger("locf").info(
            "LOCF bridged %d (name,day) price cells for %s %s (cap=%dd)",
            _locf_n, award, season, locf_cap_days)
    return feature_snaps, dict(sorted(daily.items()))
