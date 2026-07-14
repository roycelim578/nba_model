"""Stat-leader arm: v2 correlation, the heterogeneous shared draw.

Stage one (mu-sharpening) was a per-contender mean nudge and moved P(lead) by
nothing, because the nudge was a fraction of a percent. A shared factor applied
IDENTICALLY to every contender (like reb_env) also cannot move P(lead): a common
multiplier scales all contenders equally, so it cancels in the ranking and only
shifts absolute levels. The only shared structure that changes the argmax is a
HETEROGENEOUS one, where contenders co-move by how much they actually overlap.

So v2 draws a shared shock PER OPPONENT TEAM per replicate, and each contender's
remaining rate is multiplied by the average of the shocks of the opponents HE
plays. Two contenders who share opponents share those shocks and their gap
variance shrinks; contenders with disjoint remaining schedules stay independent.
That differential shrinkage is what sharpens the front-runner's P(lead), the
order-statistic lever, without a blanket factor.

The shock is a common per-game environment shock (opponent form, pace, whistle)
around 1, with SD calibrated NON-CIRCULARLY to reproduce the shared-share of the
comove co-movement that stage-one attribution attributed to schedule overlap, not
tuned to P(lead). A player facing n_g remaining games, m of them against team t,
gets shock weight m/n_g on team t.

Sober ceiling: the opponent-driven rate swing is small relative to the intrinsic
count variance, so even done correctly this is expected to be a modest P(lead)
move; it is built because it is the one remaining named structural lever, and off
(CORR2 False) the engine is bit-identical.

attach_weights fills each contender's opp_w = {team_id: weight}; fit_sd sets the
per-team shock SD from the measured comove shared-share.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np

try:
    from scripts.features.stat_leader import remaining_schedule as RS
except ImportError:  # pragma: no cover
    import remaining_schedule as RS  # type: ignore

log = logging.getLogger("stat_leader.corr2")

# default per-team shock SD if a data-driven value is unavailable; small, since the
# opponent environment moves a per-game rate only modestly.
DEFAULT_SD = 0.06
SD_CAP = 0.15


def fit_sd(shared_share=None):
    """Per-team shock SD. If the stage-one opp_z shared-share is supplied (the
    fraction of cross-contender co-movement attributable to schedule overlap), map
    it to a per-team SD that reproduces roughly that pairwise correlation for
    fully-overlapping contenders: for k shared opponents the induced correlation is
    ~ SD^2 / (SD^2 + idio), so a small shared-share implies a small SD. We take
    SD = sqrt(shared_share) * DEFAULT_SD scale, capped. Non-circular: it is set
    from comove, never from the P(lead) outcome."""
    if shared_share is None or shared_share <= 0:
        return DEFAULT_SD
    return float(min(SD_CAP, (shared_share ** 0.5) * 0.35))


def attach_weights(conn, counts, ctx, season):
    """Fill counts[(season, snap, pid)]['opp_w'] = {opp_team_id: weight} from the
    contender's as-of remaining schedule (weights sum to 1). Cached per (snap,
    team). Missing schedule leaves opp_w absent, so the shock no-ops for that
    contender."""
    full = RS.full_schedule(conn, season)
    if not full:
        return
    pteam = defaultdict(list)
    for r in conn.execute(
        "SELECT nba_api_id, game_date, team_id FROM stg_nba_player_game_logs "
        "WHERE season=? AND team_id IS NOT NULL ORDER BY nba_api_id, game_date", (season,)):
        pteam[r["nba_api_id"]].append((r["game_date"], r["team_id"]))

    import bisect

    def team_asof(pid, snap):
        seq = pteam.get(pid)
        if not seq:
            return None
        i = bisect.bisect_right([d for d, _ in seq], snap) - 1
        return seq[i][1] if i >= 0 else None

    wcache = {}

    def weights(team, snap):
        key = (snap, team)
        if key in wcache:
            return wcache[key]
        rem = RS.remaining_asof(full, team, snap)
        w = None
        if rem:
            cnt = defaultdict(int)
            for _, opp in rem:
                cnt[opp] += 1
            n = float(len(rem))
            w = {t: c / n for t, c in cnt.items()}
        wcache[key] = w
        return w

    for snap, players in ctx.items():
        for pid in players:
            d = counts.get((season, snap, pid))
            if not d:
                continue
            t = team_asof(pid, snap)
            w = weights(t, snap) if t is not None else None
            if w:
                d["opp_w"] = w


def draw_team_shocks(rng, teams, sd, k):
    """One shared mean-1 shock vector per opponent team for this snapshot/replicate
    set. Lognormal-ish via clipped normal to stay positive."""
    return {t: np.maximum(0.4, rng.normal(1.0, sd, size=k)) for t in teams}


def contender_mult(opp_w, team_shocks, k):
    """Weighted average of the shocks of a contender's remaining opponents; 1.0
    (no shock) when the contender has no schedule or none of his opponents were
    drawn (should not happen since teams come from the union)."""
    if not opp_w:
        return None
    acc = np.zeros(k)
    tot = 0.0
    for t, w in opp_w.items():
        s = team_shocks.get(t)
        if s is None:
            continue
        acc += w * s
        tot += w
    if tot <= 0:
        return None
    return acc / tot
