"""Stat-leader arm: prior cache (fit once, draw many).

The expensive part of every run is the prior fit inside MC.load_all (gamma and
cohort priors, availability pools, minutes shrinks, reb-env variance, the
avail-hier prior lookup). None of it depends on the toggle flags, which are all
applied at draw time, so the fit is a pure function of (season, fit-lookback,
data) and can be cached once and reused across every A/B config. The cache stores
only the fitted objects, not the bulky per-season counts, so a cached load stays
small and reloads the eval-season data fresh from the DB.

Layout: models/stat_leader/priors_s{season}_lb{lookback}_{fingerprint}.pkl
The fingerprint is a cheap count-and-max-date of the game logs over the fit range,
so a data pull invalidates the cache automatically; --refit forces a rebuild.

ensure() fits and saves if missing; load() returns a B dict equivalent to
MC.load_all's, assembled from the cached priors plus a fresh eval-season data
read, together with the avail-hier prior. The caller sets the MC globals.
"""

from __future__ import annotations

import logging
import os
import pickle

try:
    from scripts.features.stat_leader import nodes as N
    from scripts.features.stat_leader import avail_hier as AH
    from scripts.modelling.stat_leader import mc as MC
except ImportError:  # pragma: no cover
    import nodes as N  # type: ignore
    import avail_hier as AH  # type: ignore
    import mc as MC  # type: ignore

log = logging.getLogger("stat_leader.prior_cache")

CACHE_DIR = "models/stat_leader"
_PRIOR_KEYS = ("vpriors", "npriors", "pools", "tcut", "reb_env_var",
               "mpg_k", "games_k", "pos", "firstyr")


def _fingerprint(conn, fit_lo, eval_season):
    r = conn.execute(
        "SELECT COUNT(*) c, COALESCE(MAX(game_date),'') m FROM stg_nba_player_game_logs "
        "WHERE season BETWEEN ? AND ?", (fit_lo, eval_season)).fetchone()
    return f"{r['c']}_{r['m']}"


def _path(conn, season, lookback):
    fit_lo = season - lookback
    fp = _fingerprint(conn, fit_lo, season)
    return os.path.join(CACHE_DIR, f"priors_s{season}_lb{lookback}_{fp}.pkl")


def ensure(conn, season, lookback, refit=False):
    """Fit and save the priors for (season, lookback) if not already cached.
    Returns the cache path. Distinct seasons write distinct files, so this is
    safe to run in parallel across seasons."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = _path(conn, season, lookback)
    if os.path.exists(path) and not refit:
        return path
    B = MC.load_all(conn, season, lookback)
    avail_prior = AH.fit(conn, list(range(B["fit_lo"], B["fit_hi"] + 1)))
    blob = {k: B[k] for k in _PRIOR_KEYS}
    blob["avail_prior"] = avail_prior
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(blob, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    log.info("fitted+cached priors: %s", os.path.basename(path))
    return path


def load(conn, season, lookback, refit=False):
    """Return (B, avail_prior). B is assembled from cached priors plus a fresh
    eval-season data read, equivalent to MC.load_all for scoring purposes."""
    path = ensure(conn, season, lookback, refit)
    with open(path, "rb") as fh:
        blob = pickle.load(fh)
    counts, finals, _pos, _fy = N._load(conn, [season])
    ctx = MC._load_context(conn, season)
    ftg = MC._load_ftg(conn, season)
    B = dict(counts=counts, finals=finals, ctx=ctx, ftg=ftg,
             pos=blob["pos"], firstyr=blob["firstyr"],
             vpriors=blob["vpriors"], npriors=blob["npriors"],
             pools=blob["pools"], tcut=blob["tcut"],
             reb_env_var=blob["reb_env_var"], mpg_k=blob["mpg_k"],
             games_k=blob["games_k"])
    return B, blob["avail_prior"]
