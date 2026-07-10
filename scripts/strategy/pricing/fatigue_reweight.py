"""External performance-conditional fatigue reweight (MVP, DPOY).

A serve-time multiplier on the priced point for actual prior winners, driven by how far a
player's CURRENT on-court form has fallen below his own peak award-season form, escalated by
the number of prior awards. Declined multi-time winner is pulled down; a repeater still at his
own standard is untouched because his decline term is ~0. Applied after eligibility and the
shelved incumbency-divergence layer, before the rank floor; never touches the eta cloud;
enters no selection.

Form composite: mean of z-standardised (within season-snapshot group) core performance
features, sign-aligned so higher is always better. Raw levels are read from feature_stats_asof
and standardised here, so the composite is encoding-agnostic, cross-season comparable, and
matches diag_feature_separation exactly.

DPOY composite (validated by AUROC on the repeat-winner panel, composite 0.938):
  steals, blocks, individual defensive rating (negated), all-shots defended-FG% (negated).
  Dropped: team defensive rating (redundant with individual), individual-net-of-team (AUROC
  0.65, did not separate). All-shots defended-FG% is tracking-era only (~2014+); pre-2014 it
  is null and the decline is computed over the remaining shared features.

MVP composite: NOT YET VALIDATED. Run diag_feature_separation on MVP repeat winners
(Jokic, Giannis, LeBron, Duncan, ...) and confirm AUROC before enabling the layer for MVP.
The team_conf_rank sign was ambiguous, so team_win_pct (unambiguous, higher-better) is used.

Decline is computed over the INTERSECTION of features present in BOTH the current and the
baseline season, so a change in feature availability (e.g. tracking-era gap) never manufactures
spurious decline; the composite is renormalised (mean) over that shared set.

Penalty: decline = max(0, baseline_form - current_form) in z-units; multiplier =
clip(1 - K * decline * count**POWER, FLOOR, 1); count = prior awards of THIS award before this
season. Count active from 1; the decline gate keeps corroborated repeaters safe at any count.
As-of: prior award seasons are strictly < current; form read from the snapshot.

Baseline modes: 'best' (highest-form prior award season), 'recent' (most recent), 'median'
(the prior award season nearest the median of their forms, robust to a single freak peak),
'mean' (per-feature average across prior award seasons, lets a dominant season ratchet the bar
up). LOCKED DEFAULT is 'median' with K=0.10, POWER=1.3, FLOOR=0.20, chosen on a baseline-mode
sweep across both awards: median de-floods the long-career artefact (a high multi-season peak,
e.g. LeBron 2009-2013, does not manufacture a decade of false decline the way best/mean do)
while protecting won rows and still firing on genuine fatigue (Gobert 2024, Howard late years).
One mode for both awards by design; per-award tuning would be overfitting.

KNOWN LIMITATION (MVP): a declining but still-elite perennial contender (LeBron post-2013,
correctly backed by voters on GOAT-tier narrative) reads as fatigued on form and is cut, a false
positive no baseline mode or gate cures at serve time, because current form cannot distinguish an
over-backed decliner (Giannis 2020, pull correct) from a correctly-backed one (LeBron 2017, pull
wrong). DPOY does not have this (a declined DPOY winner falls out of contention entirely). Hence
DPOY ships; MVP enablement is gated on the on-vs-off book showing no bleed on LeBron-type cases.

British English. No inline comments.
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger("fatigue_reweight")

K = 0.10
POWER = 1.3
FLOOR = 0.20
# Second-win gate: MVP fires only from the 2nd award (first title defence is not
# real fatigue; it is the won-year false-positive region). DPOY unchanged at 1.
MIN_COUNT = {"MVP": 2, "DPOY": 2}
# Second-win gate: MVP fires only from the 2nd award (first title defence is not
# real fatigue; it is the won-year false-positive region). DPOY unchanged at 1.
MIN_COUNT = {"MVP": 2, "DPOY": 1}

DEFAULT_FEATURES = {
    "DPOY": [("box_spg_std", +1), ("box_bpg_std", +1),
             ("adv_def_rating", -1), ("dfn_dpct_overall_std", -1)],
    "MVP": [("adv_pie", +1), ("team_win_pct", +1), ("box_ts_pct_std", +1)],
}

_FZ_CACHE: dict = {}
_WIN_CACHE: dict = {}


def _zscore(vals):
    v = np.asarray(vals, dtype=float)
    f = np.isfinite(v)
    if f.sum() < 2:
        return np.full(len(v), np.nan)
    mu = v[f].mean()
    sd = v[f].std()
    if sd == 0:
        return np.zeros(len(v))
    z = (v - mu) / sd
    z[~f] = np.nan
    return z


def season_feature_z(conn, award, season, features, snap=None):
    """{player_id: {feature: z}} at the given snapshot (default late) for one season.
    Each feature is z-standardised within the snapshot's candidate group and sign-aligned.
    Reads raw levels from feature_stats_asof directly."""
    key = (award, int(season), tuple(features), snap)
    if key in _FZ_CACHE:
        return _FZ_CACHE[key]
    cols = [f for f, _s in features]
    rows = conn.execute(
        f"SELECT player_id, snapshot_date, {', '.join(cols)} FROM feature_stats_asof "
        f"WHERE award = ? AND season = ?", (award, int(season))).fetchall()
    if not rows:
        _FZ_CACHE[key] = {}
        return {}
    late = snap if snap is not None else sorted({r[1] for r in rows})[-1]
    grp = [r for r in rows if r[1] == late]
    if len(grp) < 2:
        _FZ_CACHE[key] = {}
        return {}
    pids = [int(r[0]) for r in grp]
    colidx = {c: 2 + i for i, c in enumerate(cols)}
    out = {pid: {} for pid in pids}
    for f, sign in features:
        z = sign * _zscore([grp[k][colidx[f]] for k in range(len(grp))])
        for k, pid in enumerate(pids):
            out[pid][f] = float(z[k])
    _FZ_CACHE[key] = out
    return out


def _scalar(fz):
    vals = [v for v in fz.values() if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def prior_award_seasons(conn, award, pid, season):
    """Seasons this player won THIS award strictly before `season` (leak-safe)."""
    key = (award, int(pid))
    if key not in _WIN_CACHE:
        rows = conn.execute(
            "SELECT season FROM award_voting WHERE award=? AND player_id=? AND won_flag=1",
            (award, int(pid))).fetchall()
        _WIN_CACHE[key] = sorted(int(r[0]) for r in rows)
    return [s for s in _WIN_CACHE[key] if s < int(season)]


def _baseline(conn, award, pid, season, features, mode):
    priors = prior_award_seasons(conn, award, pid, season)
    if not priors:
        return None, 0
    cands = []
    for s in priors:
        fz = season_feature_z(conn, award, s, features).get(int(pid))
        if fz:
            sc = _scalar(fz)
            if np.isfinite(sc):
                cands.append((s, fz, sc))
    if not cands:
        return None, len(priors)
    if mode == "recent":
        return max(cands, key=lambda x: x[0])[1], len(priors)
    if mode == "median":
        target = float(np.median([c[2] for c in cands]))
        return min(cands, key=lambda x: abs(x[2] - target))[1], len(priors)
    if mode == "mean":
        avg = {}
        for f, _s in features:
            vals = [c[1].get(f) for c in cands
                    if c[1].get(f) is not None and np.isfinite(c[1].get(f))]
            avg[f] = float(np.mean(vals)) if vals else float("nan")
        return avg, len(priors)
    return max(cands, key=lambda x: x[2])[1], len(priors)


def _shared_decline(cur_fz, base_fz, features):
    """Decline over features present (finite) in BOTH current and baseline; mean over that set."""
    shared = [f for f, _s in features
              if np.isfinite(cur_fz.get(f, np.nan)) and np.isfinite(base_fz.get(f, np.nan))]
    if not shared:
        return None
    cur = np.mean([cur_fz[f] for f in shared])
    base = np.mean([base_fz[f] for f in shared])
    return max(0.0, float(base - cur)), float(base), float(cur), len(shared)


def fatigue_multiplier(decline, count, k=K, power=POWER, floor=FLOOR):
    if count <= 0 or decline <= 0:
        return 1.0
    return float(np.clip(1.0 - k * decline * (count ** power), floor, 1.0))


def apply_fatigue(w, pids, conn, award, season, features=None, snap=None,
                  mode="median", k=K, power=POWER, floor=FLOOR, return_detail=False):
    """Multiply prior winners' priced pwin by their fatigue multiplier and renormalise."""
    w = np.asarray(w, dtype=float)
    if award not in ("MVP", "DPOY"):
        return (w.copy(), []) if return_detail else w.copy()
    features = features or DEFAULT_FEATURES[award]
    cur_all = season_feature_z(conn, award, season, features, snap=snap)

    w_adj = w.copy()
    detail = []
    for i, pid in enumerate(pids):
        base_fz, count = _baseline(conn, award, int(pid), season, features, mode)
        if base_fz is None or count < MIN_COUNT.get(award, 1):
            continue
        cur_fz = cur_all.get(int(pid))
        if not cur_fz:
            continue
        sd = _shared_decline(cur_fz, base_fz, features)
        if sd is None:
            continue
        decline, base_sc, cur_sc, n_shared = sd
        mult = fatigue_multiplier(decline, count, k, power, floor)
        w_adj[i] = w[i] * mult
        if return_detail:
            detail.append(dict(player_id=int(pid), count=count, baseline=round(base_sc, 3),
                               current=round(cur_sc, 3), decline=round(decline, 3),
                               mult=round(mult, 4), n_shared=n_shared))

    total = w_adj.sum()
    if total > 0:
        w_adj = w_adj / total
    return (w_adj, detail) if return_detail else w_adj
