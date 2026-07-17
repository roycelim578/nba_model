"""Anchored idempotent patcher: soft-root bounded-influence compression of the
per-game usage/reb volume counts and the per-snapshot-window potast (creation
volume) count in rates.py, toward each player's own EWMA as-of rate. Inert by
default: scales read from env (VOL_ROBUST_S_USAGE / _S_REB / _S_POTAST) and are
None unless set, so an unset build reproduces stat_rate_counts_asof
byte-identically. Half-life reads from VOL_ROBUST_HL_MIN, default 500.

usage and reb are compressed per game, exactly as banked. potast has no per-game
substrate (the ext table reports it as a season-to-date cumulative per snapshot,
not built from stg_nba_player_game_logs), so it is compressed at the snapshot-to-
snapshot window level: the implied rate over that window is compressed toward the
reference and the compressed window count is accumulated, which telescopes back
to the raw passthrough exactly when the scale is None.

  python3 patch_rates_robust.py --path scripts/features/stat_leader/rates.py
  python3 patch_rates_robust.py --path scripts/features/stat_leader/rates.py --apply
"""
import argparse
import shutil
import sys

EDITS = [
    ("import sys\nfrom collections import defaultdict",
     "import math\nimport os\nimport sys\nfrom collections import defaultdict"),
    ("""FT_POSS_COEF = 0.44
""",
     """FT_POSS_COEF = 0.44


def _envf(name):
    v = os.environ.get(name)
    return float(v) if v not in (None, "") else None


S_USAGE = _envf("VOL_ROBUST_S_USAGE")
S_REB = _envf("VOL_ROBUST_S_REB")
S_POTAST = _envf("VOL_ROBUST_S_POTAST")
REF_SEED_MIN = float(os.environ.get("VOL_ROBUST_SEED_MIN", "48"))
REF_HL_MIN = float(os.environ.get("VOL_ROBUST_HL_MIN", "500"))


def _compress(r, mu, s):
    \"\"\"Soft-root bounded influence, unit slope at mu: locally linear near the
    player's own reference (normal game-to-game variance and genuine drift pass
    through essentially unchanged) and growing like sqrt only in the tail, so a
    game's or window's rate is damped in proportion to how far it sits from the
    reference, with diminishing marginal weight. s is the scale at which the bend
    becomes material; s is None disables it (r unchanged, byte-identical).\"\"\"
    if s is None or r <= 0.0:
        return r
    d = r - mu
    return mu + math.copysign(s * (math.sqrt(1.0 + 2.0 * abs(d) / s) - 1.0), d)


class _RunRef:
    \"\"\"EWMA of the compressed per-minute rate, seeded by a league rate with
    REF_SEED_MIN pseudo-minutes. Half-life REF_HL_MIN minutes: long enough that
    the reference is a stable anchor (not itself chasing recent form, which
    would reintroduce the recency bias the compression is meant to remove), short
    enough to still be the player's own history rather than a lifetime constant.
    Updated with the already-compressed rate, so a spike cannot inflate its own
    reference. The current game/window is never in its own reference.\"\"\"
    __slots__ = ("num", "den", "hl")

    def __init__(self, mu0, w0, hl):
        self.num = mu0 * w0
        self.den = w0
        self.hl = hl

    def mean(self):
        return self.num / self.den if self.den > 0 else 0.0

    def update(self, mn, r_comp):
        dec = 0.5 ** (mn / self.hl) if self.hl and self.hl > 0 else 1.0
        self.num = self.num * dec + mn * r_comp
        self.den = self.den * dec + mn


def _robust_factor(count_g, mn, ref, s):
    \"\"\"Scale factor for a count over a span of minutes (one game, or one
    snapshot window) so its implied rate is compressed toward the running
    reference; updates the reference with the compressed rate. Returns 1.0 when
    s is None or the span is degenerate (inert).\"\"\"
    if mn <= 0.0 or count_g <= 0.0:
        return 1.0
    r = count_g / mn
    r_c = _compress(r, ref.mean(), s)
    ref.update(mn, r_c)
    return r_c / r
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
                    f_u = _robust_factor(used_g, mn, ref_u, S_USAGE)
                    f_r = _robust_factor(reb_g, mn, ref_r, S_REB)
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
            f_p = _robust_factor(_win_cnt, _win_min, ref_p, S_POTAST)
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
