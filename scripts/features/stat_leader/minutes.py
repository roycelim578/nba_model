"""Stat-leader arm: measured remaining-minutes shrink (replaces the hand-picked
per-stat scalar, which was exogenous overfitting).

The availability node draws remaining minutes-per-game from a peer pool. Its own
calibration shows that pool is about right for the average rotation player
(coverage ~0.75 to 0.80), but too wide for the CONTENDERS: a star is pooled with
less-stable peers, so his remaining-minutes spread inherits their volatility. The
MC ablation confirmed this minutes width is the single largest contributor to
eff_value variance, flattening the P(lead) argmax and under-confidently spreading
mass onto the pack.

Fix: shrink each sampled remaining-mpg toward the player's own banked mpg by a
reliability weight, and set the one hyperparameter by coverage-matching, the same
disciplined method used for the volume fano and the Dirichlet widths, NOT by
fitting to the leaderboard outcome.

  w = K / (M + K),   rm_shrunk = banked_mpg + w * (rm - banked_mpg)

M is the player's banked minutes, so a heavy-minute star (large M) shrinks hard
toward his own stable pace while a thin-history player (small M, w -> 1) keeps the
full peer spread. K is fit so the remaining-minutes predictive covers realised
remaining mpg at 0.80, with each observation weighted by M so the target is the
minutes calibration OF THE CONTENDERS, not of the fringe majority.

Self-policing: if the contenders' minutes are not actually too wide, the grid
returns the no-shrink K (w -> 1) and nothing changes, so the shrink is only
applied when the marginal calibration earns it, never because it flattered a
downstream P(lead) number.
"""

from __future__ import annotations

import numpy as np

try:
    from scripts.common.db import connect  # noqa: F401
    from scripts.features.stat_leader import availability as A
except ImportError:  # pragma: no cover
    import availability as A  # type: ignore

K_GRID = (50.0, 100.0, 200.0, 400.0, 800.0, 1600.0, 3200.0, 6400.0, 1e12)
KG_GRID = (5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0, 640.0, 1e12)
TARGET_COV = 0.80
FIT_DRAWS = 200
NO_SHRINK = 1e12


def shrink_mpg(rm, banked_mpg, banked_min, K):
    """Contract sampled remaining-mpg toward banked mpg by w = K/(M+K). K large
    (NO_SHRINK) is a no-op; banked_min 0 leaves the peer draws untouched."""
    if K is None or banked_min <= 0:
        return rm
    w = K / (banked_min + K)
    return banked_mpg + w * (rm - banked_mpg)


def _pool_minutes(pools, rec, tcut, k, rng):
    b = A._bin(rec, tcut)
    pool = None
    for key in A._backoff_keys(b):
        p = pools.get(key)
        if p and len(p) >= A.MIN_POOL:
            pool = p; break
    if pool is None:
        pool = pools.get((None, None, None), [])
    if not pool:
        return None
    idx = rng.integers(0, len(pool), size=k)
    rm = np.array([pool[i][1] for i in idx if pool[i][1] is not None], dtype=float)
    return rm if len(rm) >= 20 else None


def _banked_lookup(conn, seasons):
    qs = ",".join("?" * len(seasons))
    out = {}
    for r in conn.execute(
        f"SELECT a.season, a.snapshot_date, a.nba_api_id, a.games_played_asof, b.mpg_std "
        f"FROM stg_nba_availability_asof a "
        f"JOIN snapshot_grid g ON g.season=a.season AND g.snapshot_date=a.snapshot_date "
        f"LEFT JOIN stg_nba_box_asof b ON b.nba_api_id=a.nba_api_id AND b.season=a.season "
        f"  AND b.snapshot_date=a.snapshot_date "
        f"WHERE a.season IN ({qs}) AND g.snapshot_kind IN ('weekly','ratings')", seasons):
        out[(r["season"], r["snapshot_date"], r["nba_api_id"])] = (r["mpg_std"], r["games_played_asof"])
    return out


def fit_shrink_k(conn, train_recs, pools, tcut, seasons):
    """Coverage-match K on the minutes marginal, weighted by banked minutes so the
    target is the contenders. Returns K (NO_SHRINK if the pool is already tight
    enough for high-minute players)."""
    rng = np.random.default_rng(31)
    banked = _banked_lookup(conn, seasons)
    samples = []
    for r in train_recs:
        if r["rem_mpg"] is None:
            continue
        bk = banked.get((r["season"], r["snap"], r["pid"]))
        if not bk or not bk[0] or not bk[1]:
            continue
        mpg_std, gp = float(bk[0]), float(bk[1])
        rm = _pool_minutes(pools, r, tcut, FIT_DRAWS, rng)
        if rm is None:
            continue
        samples.append((rm, mpg_std, mpg_std * gp, float(r["rem_mpg"])))
    if len(samples) < 50:
        return NO_SHRINK
    best_gap, best_K = float("inf"), NO_SHRINK
    for K in K_GRID:
        num = den = 0.0
        for rm, bmpg, M, realised in samples:
            sh = shrink_mpg(rm, bmpg, M, K)
            lo, hi = np.quantile(sh, [0.1, 0.9])
            num += M * (1.0 if lo <= realised <= hi else 0.0); den += M
        cov = num / den if den > 0 else 0.0
        gap = abs(cov - TARGET_COV)
        if gap < best_gap:
            best_gap, best_K = gap, K
    return float(best_K)


def shrink_frac(rf, avail_rate, banked_team_games, Kg):
    """Contract the sampled remaining-games fraction toward the player's banked
    availability rate by w = Kg/(N+Kg), N = banked team games. Kg large (NO_SHRINK)
    is a no-op; result clipped to [0, 1]."""
    if Kg is None or banked_team_games <= 0:
        return rf
    w = Kg / (banked_team_games + Kg)
    return np.clip(avail_rate + w * (rf - avail_rate), 0.0, 1.0)


def _pool_fracs(pools, rec, tcut, k, rng):
    b = A._bin(rec, tcut)
    pool = None
    for key in A._backoff_keys(b):
        p = pools.get(key)
        if p and len(p) >= A.MIN_POOL:
            pool = p; break
    if pool is None:
        pool = pools.get((None, None, None), [])
    if not pool:
        return None
    idx = rng.integers(0, len(pool), size=k)
    return np.array([pool[i][0] for i in idx], dtype=float)


def _team_games_lookup(conn, seasons):
    qs = ",".join("?" * len(seasons))
    out = {}
    for r in conn.execute(
        f"SELECT a.season, a.snapshot_date, a.nba_api_id, a.team_games_asof "
        f"FROM stg_nba_availability_asof a "
        f"JOIN snapshot_grid g ON g.season=a.season AND g.snapshot_date=a.snapshot_date "
        f"WHERE a.season IN ({qs}) AND g.snapshot_kind IN ('weekly','ratings')", seasons):
        out[(r["season"], r["snapshot_date"], r["nba_api_id"])] = r["team_games_asof"]
    return out


def fit_shrink_kg(conn, train_recs, pools, tcut, seasons):
    """Coverage-match Kg on the games marginal (remaining games played), weighted by
    banked team games so the target is players with the most availability history.
    Returns Kg (NO_SHRINK if the games pool is already tight enough)."""
    rng = np.random.default_rng(37)
    tg = _team_games_lookup(conn, seasons)
    samples = []
    for r in train_recs:
        N = tg.get((r["season"], r["snap"], r["pid"]))
        if not N or N <= 0:
            continue
        rf = _pool_fracs(pools, r, tcut, FIT_DRAWS, rng)
        if rf is None or len(rf) < 20:
            continue
        realised_played = r["rem_frac"] * r["rem_team"]
        samples.append((rf, float(r["avail_rate"]), float(N), float(r["rem_team"]), realised_played))
    if len(samples) < 50:
        return NO_SHRINK
    best_gap, best_Kg = float("inf"), NO_SHRINK
    for Kg in KG_GRID:
        num = den = 0.0
        for rf, avail, N, rem_team, realised in samples:
            played = shrink_frac(rf, avail, N, Kg) * rem_team
            lo, hi = np.quantile(played, [0.1, 0.9])
            num += N * (1.0 if lo <= realised <= hi else 0.0); den += N
        cov = num / den if den > 0 else 0.0
        gap = abs(cov - TARGET_COV)
        if gap < best_gap:
            best_gap, best_Kg = gap, Kg
    return float(best_Kg)
