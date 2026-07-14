"""Stat-leader arm: shared rebounding-environment factor (REB only).

Rebounds are the one PRA book with a shared, finite resource: a night has a fixed
pool of available boards (missed shots across both teams), and the contenders
compete for the SAME pool. The MC draws each contender's remaining rebounds
independently, which manufactures leapfrogs that do not happen, spreads the
P(lead) argmax, and leaves the field jointly over-dispersed even though each
single-player marginal is coverage-matched. PTS and AST have no such shared pool
(usage caps scorers within a team; assist creation is self-generated), which is
why the under-separation is REB-specific.

Fix: a single shared multiplicative factor env ~ mean 1, drawn ONCE per replicate
and applied to every reb contender's rate, so a scarce-boards night lowers the
whole field together and preserves their ranking. The correlation lives entirely
in the shared draw; the per-contender marginal is essentially unchanged for a
small dispersion.

The dispersion is MEASURED, not chosen: the realised night-level co-movement of
rebounding, an intraclass correlation of weekly reb-per-minute increments among
the top rebounders (the fraction of increment variance explained by a shared
date effect). Weekly increments attenuate the true per-night co-movement, so this
is a conservative floor; it self-polices to zero when there is no shared signal.
Computed from the in-memory count panel, so it needs no extra query.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

MIN_GAME_MIN = 48.0   # a weekly increment must span at least ~a game of minutes
TOPK = 20             # measure the environment among the contender population
CLIP = 0.5


def fit_env_var(counts, finals, topk=TOPK, clip=CLIP):
    by_season = defaultdict(list)
    for (s, pid), d in finals.items():
        gp = d.get("gp_played_asof") or 0.0
        mn = d.get("min_asof") or 0.0
        reb = d.get("reb")
        if gp < 1 or mn <= 0 or reb is None:
            continue
        by_season[s].append((reb / mn, pid))
    top = {s: set(pid for _, pid in sorted(v, reverse=True)[:topk]) for s, v in by_season.items()}

    panel = defaultdict(list)
    for (s, snap, pid), d in counts.items():
        if pid not in top.get(s, ()):
            continue
        reb = d.get("reb"); mn = d.get("min_asof")
        if reb is None or mn is None:
            continue
        panel[(s, pid)].append((snap, float(reb), float(mn)))

    player_inc = defaultdict(list)
    for (s, pid), rows in panel.items():
        rows.sort()
        for a, b in zip(rows, rows[1:]):
            dmin = b[2] - a[2]; dreb = b[1] - a[1]
            if dmin >= MIN_GAME_MIN and dreb >= 0:
                player_inc[(s, pid)].append((b[0], dreb / dmin))

    allres = []; bydate = defaultdict(list)
    for (s, pid), seq in player_inc.items():
        if len(seq) < 3:
            continue
        mu = float(np.mean([r for _, r in seq]))
        for snap, r in seq:
            e = r - mu; allres.append(e); bydate[(s, snap)].append(e)
    if len(allres) < 100:
        return 0.0
    total_var = float(np.var(allres))
    if total_var <= 0:
        return 0.0
    grand = float(np.mean(allres)); num = den = 0.0
    for _, es in bydate.items():
        if len(es) < 3:
            continue
        num += len(es) * (float(np.mean(es)) - grand) ** 2; den += len(es)
    between = num / den if den > 0 else 0.0
    return float(min(max(between / total_var, 0.0), clip))
