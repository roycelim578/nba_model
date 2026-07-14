"""Stat-leader arm: hierarchical per-game fano (the variance analogue of the mean).

The mean rate posterior blends the player's own history toward a cohort toward the
league. The per-game overdispersion (fano) did not; it was a single global scalar
per node, coverage-matched to 0.80 on the whole training pool. The coverage
diagnostic showed the composed terminal predictive is under-dispersed for the
top-volume contenders (elite scorers, rebounders, creators) more than for the
pack, so one global fano under-serves exactly the players whose right tail decides
P(lead). This module gives the fano the same three-level structure as the mean.

LEVELS, on the per-game dispersion scale (realised per-game Var/mean):
  own      the player's own as-of per-game Var/mean (rates.py stores it as
           own_fano_<node>; reb and usage only, since potential assists have no
           per-game substrate, so AST creation keeps the global fano).
  cohort   mean own Var/mean within the player's mpg-tier x volume-tier cell.
           Volume-tier is the tercile of the node's per-minute rate, so the cell
           separates elite volume from the pack, which is the axis the defect
           lives on; position is deliberately not used here.
  league   mean own Var/mean across the pool.
Reliability own_games / (own_games + REF_GP) tips toward own history as games
bank, exactly as banked-minutes over kappa does for the mean.

LEVEL ANCHORING (the discipline that keeps aggregate coverage). The existing
global fano is already coverage-matched to 0.80; we keep it as the calibrated
LEVEL and let the hierarchy only redistribute dispersion around it by the ratio of
each player's blended realised dispersion to the league realised dispersion:
    fano_p = 1 + (base_fano - 1) * (blend / league_realised),
    blend  = w * own + (1 - w) * cohort.
The pool mean of blend / league_realised is ~1, so the aggregate coverage the
global fano was tuned to is preserved by construction; only the elite-versus-pack
split changes. No new free knob is coverage-matched here, so nothing is refit
circularly; REF_GP is the one principled placeholder (the games count at which an
own per-game Var/mean is decently estimated, cf REF_MIN for the mean). If the
elite-versus-pack coverage gap does not close enough, the refinement is a second
criterion on REF_GP (match top-tier coverage to pack coverage), not a global
rescale, and is deferred until the diagnostic asks for it.

Fit and fano_for are pure functions over the training pool; nothing is sealed or
season-specific. Imported by volume.fit_priors (stashes the fit) and by mc
(applies fano_for in the draw path when MC.HIER_FANO is on).
"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np

try:
    from scripts.features.stat_leader.nodes import MIN_MPG, MIN_COHORT
except ImportError:  # pragma: no cover
    from nodes import MIN_MPG, MIN_COHORT  # type: ignore

log = logging.getLogger("stat_leader.fano_hier")

REF_GP = 20.0   # games at which an own per-game Var/mean is decently estimated;
                # principled placeholder (cf volume.REF_MIN), not coverage-matched.
MIN_CELL = MIN_COHORT

# nodes that carry an own-history per-game dispersion (per-game substrate exists);
# ast_create has only an as-of cumulative potential-assist average, so it is out.
OWN_FANO_COL = {"reb": "own_fano_reb", "usage": "own_fano_usage"}
HIER_NODES = tuple(OWN_FANO_COL)

# node -> substrate count columns, kept local so this module never imports volume
# (volume imports this module; the dependency must run one way only).
_VOL_NUM = {"reb": ["reb"], "usage": ["used_fga", "used_ft_trip", "used_tov"]}


def _vol_count(d, node):
    vals = [d.get(c) for c in _VOL_NUM[node]]
    if any(v is None for v in vals):
        return None, None
    mn = d.get("min_asof")
    if not mn or mn <= 0:
        return None, None
    return float(sum(vals)), float(mn)


def _mpg_tier(mpg, mpg_cuts):
    return 0 if mpg < mpg_cuts[0] else (1 if mpg < mpg_cuts[1] else 2)


def _vol_tier(rate, cuts):
    return 0 if rate < cuts[0] else (1 if rate < cuts[1] else 2)


def _cell(node, rate, mpg, mpg_cuts, vol_cuts):
    """mpg-tier x volume-tier cell for the fano hierarchy, or None if the node has
    no volume-tier cuts (e.g. ast_create)."""
    vc = vol_cuts.get(node)
    if vc is None:
        return None
    return (_mpg_tier(mpg, mpg_cuts), _vol_tier(rate, vc))


def fit(counts, finals, pos, firstyr, base_fano, mpg_cuts):
    """Build the hierarchy from the training finals' as-of season dispersion
    (own_fano_<node> at the last snapshot ~ the player's full-season per-game
    Var/mean). Returns a dict consumed by fano_for; empty nodes (missing column)
    are simply absent, so fano_for backs off to base_fano for them."""
    rates = defaultdict(list)                       # node -> [per-min rate]
    own = defaultdict(list)                         # node -> [(rate, mpg, own_fano)]
    for (s, pid), d in finals.items():
        gp = d.get("gp_played_asof") or 0.0
        mn = d.get("min_asof") or 0.0
        if gp <= 0 or (mn / gp) < MIN_MPG:
            continue
        mpg = mn / gp
        for node in HIER_NODES:
            of = d.get(OWN_FANO_COL[node])
            vc, vm = _vol_count(d, node)
            if of is None or of <= 0 or vc is None or vm <= 0:
                continue
            rate = vc / vm
            rates[node].append(rate)
            own[node].append((rate, mpg, float(of)))

    vol_cuts, cohort, league = {}, {}, {}
    for node in HIER_NODES:
        R = rates[node]
        if len(R) < MIN_CELL:
            continue
        vol_cuts[node] = (float(np.quantile(R, 1 / 3)), float(np.quantile(R, 2 / 3)))
        agg = defaultdict(list)
        allv = []
        for rate, mpg, of in own[node]:
            cell = _cell(node, rate, mpg, mpg_cuts, vol_cuts)
            agg[cell].append(of)
            allv.append(of)
        league[node] = float(np.mean(allv)) if allv else 0.0
        cohort[node] = {c: float(np.mean(v)) for c, v in agg.items() if len(v) >= MIN_CELL}

    return {"vol_cuts": vol_cuts, "mpg_cuts": mpg_cuts, "cohort": cohort,
            "league": league, "ref_gp": REF_GP, "base_fano": dict(base_fano)}


def fano_for(hier, node, rate, mpg, own_fano, gp):
    """Per-player per-game fano. Backs off to the coverage-matched global fano when
    the node has no hierarchy (ast_create) or the league realised dispersion is
    degenerate, and floors at 1 (Poisson). own_fano None (early, or no substrate)
    uses the cohort level; the reliability weight then tips toward own as gp
    grows."""
    base = (hier.get("base_fano", {}) or {}).get(node, 1.0)
    league = hier.get("league", {}).get(node)
    if node not in HIER_NODES or league is None or league <= 0 or base <= 1.0:
        return max(1.0, base)
    cell = _cell(node, rate, mpg, hier["mpg_cuts"], hier["vol_cuts"])
    R_cell = hier["cohort"].get(node, {}).get(cell, league)
    if own_fano is not None and own_fano > 0 and gp and gp > 0:
        w = gp / (gp + hier["ref_gp"])
        blend = w * own_fano + (1.0 - w) * R_cell
    else:
        blend = R_cell
    return max(1.0, 1.0 + (base - 1.0) * (blend / league))
