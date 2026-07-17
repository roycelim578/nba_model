"""Anchored idempotent patcher: upper-only multiplicative winsor of the per-game
usage/reb rate and the per-snapshot-window potast rate in rates.py, against each
player's own EWMA as-of reference. A game's rate is left untouched unless it
exceeds K times the reference, in which case only the excess above the cap is
removed from what gets projected forward; the banked/realised total elsewhere is
never touched, this only affects the count that feeds the volume-node posterior.
Below-cap games, including poor ones, are never touched, upper tail only. Guarded
by a minimum-games count so the cap cannot fire before the reference reflects the
player's own history rather than the league seed. Inert by default: K reads from
env (VOL_WINSOR_K_USAGE / _K_REB / _K_POTAST) and is None unless set, so an unset
build reproduces stat_rate_counts_asof byte-identically.

  python3 patch_rates_winsor.py --path scripts/features/stat_leader/rates.py
  python3 patch_rates_winsor.py --path scripts/features/stat_leader/rates.py --apply
"""
import argparse
import shutil
import sys

EDITS = [
    ("import sys\nfrom collections import defaultdict",
     "import os\nimport sys\nfrom collections import defaultdict"),
    ("""FT_POSS_COEF = 0.44
""",
     """FT_POSS_COEF = 0.44


def _envf(name):
    v = os.environ.get(name)
    return float(v) if v not in (None, "") else None


K_USAGE = _envf("VOL_WINSOR_K_USAGE")
K_REB = _envf("VOL_WINSOR_K_REB")
K_POTAST = _envf("VOL_WINSOR_K_POTAST")
REF_SEED_MIN = float(os.environ.get("VOL_ROBUST_SEED_MIN", "48"))
REF_HL_MIN = float(os.environ.get("VOL_ROBUST_HL_MIN", "500"))
MIN_GAMES = int(os.environ.get("VOL_WINSOR_MIN_GAMES", "5"))


class _RunRef:
    \"\"\"EWMA of the (possibly capped) per-minute rate, seeded by a league rate
    with REF_SEED_MIN pseudo-minutes so it is not undefined on game one. Half-life
    REF_HL_MIN minutes: a stable anchor, not itself chasing recent form. Updated
    with the value actually used (raw below the cap, the cap itself when a game
    is clipped), so a single blowout cannot inflate its own reference, but a
    sustained genuine step-change still migrates the reference up over a few
    games, at which point the cap stops firing on it. The current game/window is
    never in its own reference.\"\"\"
    __slots__ = ("num", "den", "hl")

    def __init__(self, mu0, w0, hl):
        self.num = mu0 * w0
        self.den = w0
        self.hl = hl

    def mean(self):
        return self.num / self.den if self.den > 0 else 0.0

    def update(self, mn, r_used):
        dec = 0.5 ** (mn / self.hl) if self.hl and self.hl > 0 else 1.0
        self.num = self.num * dec + mn * r_used
        self.den = self.den * dec + mn


def _winsor_factor(count_g, mn, ref, k, gp, min_games):
    \"\"\"Scale factor for a count over a span of minutes (one game, or one
    snapshot window) so its implied rate is capped at k times the running
    reference, upper tail only; below the cap, or with k unset, or before
    min_games of the player's own history have accrued, the game passes through
    unchanged. Always updates the reference with the value actually used, so the
    reference matures from game one even while the cap itself is held off.\"\"\"
    if mn <= 0.0 or count_g <= 0.0:
        return 1.0
    r = count_g / mn
    if k is None or gp < min_games:
        ref.update(mn, r)
        return 1.0
    cap = k * ref.mean()
    if cap <= 0.0 or r <= cap:
        ref.update(mn, r)
        return 1.0
    ref.update(mn, cap)
    return cap / r
"""),
    ("""    potast = _load_potast(conn, season)

    out = []
""",
     """    potast = _load_potast(conn, season)

    _tot_min = _tot_used = _tot_reb = 0.0
    for _games in logs.values():
        for _g in _games:
            _mn = _g["minutes"] or 0.0
            if _mn <= 0:
                continue
            _tot_min += _mn
            _tot_used += (_g["fga"] or 0.0) + FT_POSS_COEF * (_g["fta"] or 0.0) + (_g["turnovers"] or 0.0)
            _tot_reb += _g["rebounds"] or 0.0
    mu0_usage = (_tot_used / _tot_min) if _tot_min > 0 else 0.0
    mu0_reb = (_tot_reb / _tot_min) if _tot_min > 0 else 0.0
    _tot_potast = _tot_potast_min = 0.0
    for _pid, _vals in potast.items():
        if not _vals:
            continue
        _last = _vals[max(_vals)]
        _pmin = sum((_g["minutes"] or 0.0) for _g in logs.get(_pid, []) if (_g["minutes"] or 0) > 0)
        if _pmin > 0:
            _tot_potast += _last; _tot_potast_min += _pmin
    mu0_potast = (_tot_potast / _tot_potast_min) if _tot_potast_min > 0 else 0.0

    out = []
"""),
    ("""        c = {k: 0.0 for k in COUNT_COLS}
""",
     """        c = {k: 0.0 for k in COUNT_COLS}
        ref_u = _RunRef(mu0_usage, REF_SEED_MIN, REF_HL_MIN)
        ref_r = _RunRef(mu0_reb, REF_SEED_MIN, REF_HL_MIN)
        ref_p = _RunRef(mu0_potast, REF_SEED_MIN, REF_HL_MIN)
        _prev_potast_raw = 0.0
        _prev_min_at_potast = 0.0
"""),
    ("""                    fg2a = fga - fg3a; fg2m = fgm - fg3m
                    c["used_fga"] += fga
                    c["used_ft_trip"] += FT_POSS_COEF * fta
                    c["used_tov"] += tov
                    c["fg3a"] += fg3a; c["fg3m"] += fg3m
                    c["fg2a"] += fg2a; c["fg2m"] += fg2m
                    c["fta"] += fta; c["ftm"] += g["ftm"] or 0.0
                    c["reb"] += g["rebounds"] or 0.0""",
     """                    fg2a = fga - fg3a; fg2m = fgm - fg3m
                    reb_g = g["rebounds"] or 0.0
                    used_g = fga + FT_POSS_COEF * fta + tov
                    f_u = _winsor_factor(used_g, mn, ref_u, K_USAGE, c["gp_played_asof"], MIN_GAMES)
                    f_r = _winsor_factor(reb_g, mn, ref_r, K_REB, c["gp_played_asof"], MIN_GAMES)
                    c["used_fga"] += fga * f_u
                    c["used_ft_trip"] += FT_POSS_COEF * fta * f_u
                    c["used_tov"] += tov * f_u
                    c["fg3a"] += fg3a; c["fg3m"] += fg3m
                    c["fg2a"] += fg2a; c["fg2m"] += fg2m
                    c["fta"] += fta; c["ftm"] += g["ftm"] or 0.0
                    c["reb"] += reb_g * f_r"""),
    ("""            c["potential_ast_asof"] = potast.get(pid, {}).get(snap, 0.0)""",
     """            _raw_now = potast.get(pid, {}).get(snap, 0.0)
            _win_min = c["min_asof"] - _prev_min_at_potast
            _win_cnt = _raw_now - _prev_potast_raw
            f_p = _winsor_factor(_win_cnt, _win_min, ref_p, K_POTAST, c["gp_played_asof"], MIN_GAMES)
            c["potential_ast_asof"] += _win_cnt * f_p
            _prev_potast_raw = _raw_now
            _prev_min_at_potast = c["min_asof"]"""),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    src = open(a.path, encoding="utf-8").read()
    already = sum(1 for _o, n in EDITS if n in src)
    if already == len(EDITS):
        print("all edits already present; nothing to do"); return
    if already:
        print(f"partial apply detected ({already}/{len(EDITS)}); aborting"); sys.exit(2)
    for old, _n in EDITS:
        c = src.count(old)
        if c != 1:
            print(f"anchor count {c} != 1 for:\n{old[:80]}..."); sys.exit(2)
    out = src
    for old, new in EDITS:
        out = out.replace(old, new)
    if not a.apply:
        print(f"dry-run OK: {len(EDITS)} anchors each matched once"); return
    shutil.copy(a.path, a.path + ".bak")
    open(a.path, "w", encoding="utf-8").write(out)
    print(f"applied {len(EDITS)} edits; backup at {a.path}.bak")


if __name__ == "__main__":
    main()
