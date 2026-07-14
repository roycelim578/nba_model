"""Stat-leader arm: hierarchical availability recentre (own history for the centre).

The availability node is a nonparametric peer-pool sampler: a player is projected
on the pooled outcome distribution of peers in his coarse state bin, so his own
durability enters only as bin membership. A genuine iron-man in the top tercile is
projected on the whole top-tercile pool, which carries the league injury tail, so
his remaining games and minutes are centred below his own rate. Pooled PIT ~0.5
hides this, because it averages the low-centred durable and the high-centred
fragile; the miss is conditional, exactly as the rate shrinkage was.

The fix keeps what the pool is FOR (the fat left tail of absences and the
games/minutes covariance, which are real for anyone) and fixes only the CENTRE:
each drawn (remaining_frac, remaining_mpg) pair is rescaled multiplicatively onto
a reliability-weighted target, so a durable player's whole predictive shifts up
while retaining the pool's shape. Durability moves the centre; the tail stays.

TARGET, own history two ways (the seasonality point):
  own_frac  = a * banked_avail_rate + (1 - a) * prior_frac
  own_mpg   = a * banked_mpg        + (1 - a) * prior_mpg
where banked_* is this season as-of, prior_* is the player's PRIOR-season realised
remaining from the same week-index onward (his own late-season pattern: ramp,
fade, load-management), and a = gp / (gp + REF_GP_PRIOR) tips toward this season's
banked as games accrue. Injury-adjusted: a prior remainder that was itself
truncated (prior_frac very low, i.e. a season-ending injury, not a pattern) is
dropped from the blend, and when the player is currently absent the pool's
absent-bin plus the depressed banked rate already carry it.

Then the recentre shrinks the pool draw toward the target by reliability in games:
  r = gp / (gp + REF_GP_POOL)
  rf <- rf * (r * own_frac + (1 - r) * pool_mean_frac) / pool_mean_frac,  clipped [0,1]
  rm <- rm * (r * own_mpg  + (1 - r) * pool_mean_mpg)  / pool_mean_mpg
So with few banked games it is the pool; with many it is own history; always the
pool's dispersion. Off (AVAIL_HIER False) it is never called and the draw is the
pool as before.

fit builds the prior-season lookup once; recentre is a pure per-draw transform.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np

try:
    from scripts.features.stat_leader import availability as A
except ImportError:  # pragma: no cover
    import availability as A  # type: ignore

log = logging.getLogger("stat_leader.avail_hier")

REF_GP_PRIOR = 25.0    # banked games at which this-season availability outweighs prior-season
REF_GP_POOL = 20.0     # banked games at which own history outweighs the peer pool
PRIOR_INJURY_FLOOR = 0.45   # prior remaining-frac below this reads as injury-truncated, dropped


def fit(conn, seasons):
    """Prior-season lookup: {(season, pid, week_index): (rem_frac, rem_mpg)} built
    from availability._load, so a current (season, pid, wi) reads (season-1, pid,
    wi). Rotation filter and banked-state formulas match the pool by construction."""
    look = {}
    for s in seasons:
        for r in A._load(conn, [s]):
            look[(r["season"], r["pid"], r["week_index"])] = (r["rem_frac"], r["rem_mpg"])
    return look


def _prior(look, season, pid, wi):
    v = look.get((season - 1, pid, wi))
    if v is None:
        return None, None
    fr, mp = v
    if fr is None or fr < PRIOR_INJURY_FLOOR:   # injury-truncated prior, not a pattern
        return None, mp
    return fr, mp


def recentre(rf, rm, rc, d, pid, season, look):
    """Rescale a pool draw (rf, rm arrays) onto the reliability-weighted own-history
    target. Returns (rf, rm). No-op-safe: degenerate inputs return the draw."""
    gp = d.get("gp_played_asof") or 0.0
    mn = d.get("min_asof") or 0.0
    if gp <= 0 or rf is None:
        return rf, rm
    banked_frac = rc.get("avail_rate")
    banked_mpg = (mn / gp) if gp else None
    wi = rc.get("week_index", -1)
    prior_frac, prior_mpg = _prior(look, season, pid, wi)

    a = gp / (gp + REF_GP_PRIOR)
    own_frac = banked_frac if prior_frac is None else (a * banked_frac + (1 - a) * prior_frac)
    if banked_mpg is None:
        own_mpg = prior_mpg
    elif prior_mpg is None:
        own_mpg = banked_mpg
    else:
        own_mpg = a * banked_mpg + (1 - a) * prior_mpg

    r = gp / (gp + REF_GP_POOL)
    pf = float(np.mean(rf))
    if own_frac is not None and pf > 0:
        target_f = r * own_frac + (1 - r) * pf
        rf = np.clip(rf * (target_f / pf), 0.0, 1.0)
    valid = rm[~np.isnan(rm)] if rm is not None else np.array([])
    if own_mpg is not None and len(valid):
        pm = float(np.mean(valid))
        if pm > 0:
            rm = rm * (r * own_mpg + (1 - r) * pm) / pm
    return rf, rm
