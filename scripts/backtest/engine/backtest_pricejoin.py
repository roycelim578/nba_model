"""
Price join for the sized backtest. Per (award, season), for each model snapshot D,
assemble the tradeable candidate set with:
  - model outputs (p_win, vote_share_pred, CIs) from model_predictions
  - YES and NO candidate_ids (pairing via pm_candidates.outcome_side, clean on
    award markets; fallback to outcome text)
  - the D+1 EXECUTION price per leg (NO synthesised as 1 - yes per v5), enforcing
    the no-lookahead seal: execution timestamp must be strictly after the snapshot's
    inclusive feature cutoff (features know snapshot-day games, price stamps 00:00 UTC,
    so first valid price is the one stamped >= D+1).

Untradeable snapshot-candidates (no price at/after D+1 within tolerance) DROP OUT;
they are not errors and must not reach forward to a much-later price (that is the
mirror lookahead). British English. Read-only wrt the DB.

Public:
  load_tradeable(conn, award, season, exec_lag_days=1, forward_tol_days=6) -> dict
    { snapshot_date: SnapshotFrame }
  SnapshotFrame: .date, .candidates (list of CandidateLeg)
  CandidateLeg: player_id, name, p_win, vote_share_pred, ci..., yes_cid, no_cid,
                yes_exec_price, no_exec_price, exec_timestamp
"""
from __future__ import annotations
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta

MODEL_VERSION_LIKE = "{award}|%HELDOUT2024"  # season 2024 held-out models

@dataclass
class CandidateLeg:
    player_id: int
    name: str
    p_win: float
    vote_share_pred: float
    p_win_ci_lo: float
    p_win_ci_hi: float
    yes_cid: str
    no_cid: str
    yes_exec_price: float
    no_exec_price: float
    exec_timestamp: str

@dataclass
class SnapshotFrame:
    date: str
    candidates: list = field(default_factory=list)

def _parse(ts):
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))

def _first_price_at_or_after(rows, cutoff_dt, tol_dt):
    """rows: list of (timestamp_str, hist_price) sorted asc. Return (price, ts) for
    the first row with timestamp >= cutoff_dt and <= tol_dt, else (None, None)."""
    for ts, px in rows:
        t = _parse(ts)
        if t >= cutoff_dt:
            if t <= tol_dt:
                return float(px), ts
            return None, None  # first available is beyond tolerance -> untradeable
    return None, None

def load_tradeable(conn, award, season, exec_lag_days=1, forward_tol_days=6):
    mv_like = MODEL_VERSION_LIKE.format(award=award)
    cur = conn.cursor()

    # 1. model snapshots + predictions for this award/season
    preds = cur.execute(
        "SELECT snapshot_date, player_id, p_win, vote_share_pred, p_win_ci_lo, p_win_ci_hi "
        "FROM model_predictions WHERE model_version LIKE ? AND season=? "
        "ORDER BY snapshot_date", (mv_like, season)).fetchall()
    if not preds:
        raise SystemExit(f"no model_predictions for {award} {season} ({mv_like})")

    # 2. pairing: market_id (this award/season) -> yes/no candidate_id + player_id
    markets = [r[0] for r in cur.execute(
        "SELECT market_id FROM pm_markets WHERE award=? AND season=?",
        (award, season)).fetchall()]
    if not markets:
        raise SystemExit(f"no pm_markets for {award} {season}")
    qm = ",".join("?" * len(markets))
    crows = cur.execute(
        f"SELECT market_id, candidate_id, candidate_name, player_id, outcome, outcome_side "
        f"FROM pm_candidates WHERE market_id IN ({qm})", markets).fetchall()

    # build per-player yes/no cid using outcome_side (fallback outcome text)
    def side_of(outcome_side, outcome):
        s = (outcome_side or outcome or "").strip().upper()
        return "YES" if s in ("YES",) else ("NO" if s in ("NO",) else "?")
    pair = {}  # player_id -> {'YES':cid,'NO':cid,'name':..}
    for market_id, cid, cname, pid, outcome, oside in crows:
        if pid is None:
            continue
        pid = int(pid)
        d = pair.setdefault(pid, {"name": cname})
        d[side_of(oside, outcome)] = cid
    # keep only players with both legs
    pair = {p: d for p, d in pair.items() if "YES" in d and "NO" in d}

    # 3. price series per candidate_id (history_agg), sorted asc
    all_cids = [d[s] for d in pair.values() for s in ("YES", "NO")]
    qc = ",".join("?" * len(all_cids))
    prows = cur.execute(
        f"SELECT candidate_id, timestamp, hist_price FROM pm_prices "
        f"WHERE price_type='history_agg' AND candidate_id IN ({qc}) "
        f"ORDER BY candidate_id, timestamp", all_cids).fetchall()
    series = {}
    for cid, ts, px in prows:
        series.setdefault(cid, []).append((ts, px))

    # 4. assemble per snapshot with D+1 seal
    frames = {}
    for snap, pid, p_win, vsp, ci_lo, ci_hi in preds:
        pid = int(pid)
        if pid not in pair:
            continue
        d = pair[pid]
        cutoff = _parse(snap + "T00:00:00+00:00")           # inclusive: features know day D
        exec_from = cutoff + timedelta(days=exec_lag_days)   # first valid exec >= D+1 00:00 UTC
        tol = exec_from + timedelta(days=forward_tol_days)
        yrows = series.get(d["YES"], [])
        if not yrows:
            continue
        ypx, yts = _first_price_at_or_after(yrows, exec_from, tol)
        if ypx is None:
            continue
        # SEAL assertion: chosen exec timestamp must be strictly after the cutoff
        assert _parse(yts) >= exec_from, f"seal violation {yts} < {exec_from}"
        no_px = 1.0 - ypx  # v5 ComplementaryPrices convention
        frames.setdefault(snap, SnapshotFrame(date=snap)).candidates.append(
            CandidateLeg(
                player_id=pid, name=d["name"], p_win=p_win, vote_share_pred=vsp,
                p_win_ci_lo=ci_lo, p_win_ci_hi=ci_hi,
                yes_cid=d["YES"], no_cid=d["NO"],
                yes_exec_price=ypx, no_exec_price=no_px, exec_timestamp=yts))
    return dict(sorted(frames.items()))

def _report(award="MVP", season=2024):
    conn = sqlite3.connect("data/awards.db")
    fr = load_tradeable(conn, award, season)
    print(f"{award} {season}: {len(fr)} tradeable snapshots")
    for date, f in fr.items():
        n = len(f.candidates)
        top = sorted(f.candidates, key=lambda c: -c.p_win)[:3]
        tops = ", ".join(f"{c.name.split()[-1]} p{c.p_win:.2f}@{c.yes_exec_price:.2f}" for c in top)
        print(f"  {date}: {n:>2} cands | exec {f.candidates[0].exec_timestamp[:10]} | {tops}")
    conn.close()

if __name__ == "__main__":
    _report()
