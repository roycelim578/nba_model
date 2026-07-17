"""Lead test v2: does an as-of upstream predict the surge, beyond the stat itself?

Upgrades over v1, driven by three objections:
  (1) Same-measurement upstreams (deflections~steals, dreb~reb) can predict the
      surge through a shared event, not a genuine lead. Fix: control for the
      stat's OWN recent momentum (l10_vs_std), not just its banked level. r_lead
      is the partial correlation of the upstream with the surge controlling for
      BOTH banked level AND own recent trajectory. If a signal survives into
      r_lead it carries information the stat's own recent form does not; if it
      collapses from r_partial to r_lead it was autocorrelation / co-movement.
  (2) The bar is not 0.5. On a mostly-irreducible surge target pooled over
      thousands of obs, achievable r is low and n makes significance trivial.
      Use r_lead as a RANKING SCREEN (floor ~0.1); prove value in a later
      walk-forward re-score, not here.
  (3) win_elev is confounded (winners are elevated on everything). Read it only
      as corroboration when r_lead also fires.

Columns: suspect (shares measurement with the target), r_raw, r_partial (control
banked level), r_lead (control banked level + own momentum), win_elev, n.

surge = (final_stat - banked_stat)/(final_gp - banked_gp) - banked_stat/banked_gp

Run:
  uv run python3 -m scripts.modelling.stat_leader.lead_test --seasons 2016-2023
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

try:
    from scripts.common.db import connect
    from scripts.features.stat_leader import nodes as N
except ImportError:  # pragma: no cover
    from db import connect  # type: ignore
    import nodes as N  # type: ignore

STAT_FLOOR = {"pts": 1997, "reb": 1997, "ast": 2013, "stl": 1997, "blk": 1997}
SEALED = {"pts": {2025}, "reb": {2025}, "ast": {2025},
          "stl": {2024, 2025}, "blk": {2024, 2025}}
OWN_MOM = {"pts": "ppg_l10_vs_std", "reb": "rpg_l10_vs_std", "ast": "apg_l10_vs_std",
           "stl": "spg_l10_vs_std", "blk": "bpg_l10_vs_std"}


def _pts(d):
    return 2.0 * (d.get("fg2m") or 0.0) + 3.0 * (d.get("fg3m") or 0.0) + (d.get("ftm") or 0.0)


STAT = {"reb": lambda d: d.get("reb") or 0.0, "pts": _pts,
        "ast": lambda d: d.get("ast") or 0.0, "stl": lambda d: d.get("stl") or 0.0,
        "blk": lambda d: d.get("blk") or 0.0}

TAB = {"hustle": "stg_nba_hustle_asof", "defend": "stg_nba_defend_asof",
       "ext": "stg_nba_player_asof_ext", "adv": "stg_nba_player_advanced_asof",
       "box": "stg_nba_box_asof"}

# (alias, column, label, suspect?)  suspect = shares measurement with the target
CAND = {
    "pts": [("ext", "touches", "touches/g", False),
            ("ext", "front_ct_touches", "front-court touches/g", False),
            ("ext", "time_of_poss", "time of possession/g", False),
            ("ext", "avg_sec_per_touch", "seconds per touch", False),
            ("adv", "usg_pct", "usage %", True),
            ("adv", "pace", "pace (team)", False),
            ("ext", "poss", "possessions/g", False),
            ("ext", "pfd", "fouls drawn/g", False)],
    "reb": [("ext", "reb_pct", "REB% (avail grabbed)", True),
            ("ext", "oreb_pct", "OREB%", True),
            ("ext", "dreb_pct", "DREB%", True),
            ("ext", "pct_dreb", "share of team DREB", True),
            ("adv", "dreb", "dreb/g (adv)", True),
            ("ext", "pts_2nd_chance", "2nd-chance pts/g", True),
            ("hustle", "scrast_std", "screen assists/g", False),
            ("adv", "pace", "pace (team)", False),
            ("ext", "poss", "possessions/g", False)],
    "ast": [("ext", "potential_ast", "potential assists/g", False),
            ("ext", "touches", "touches/g", False),
            ("ext", "time_of_poss", "time of possession/g", False),
            ("ext", "secondary_ast", "secondary assists/g", False),
            ("ext", "front_ct_touches", "front-court touches/g", False),
            ("ext", "avg_drib_per_touch", "dribbles per touch", False),
            ("ext", "ast_pct", "AST% (teammate FGs)", True),
            ("ext", "blka", "own shots blocked/g", False),
            ("ext", "poss", "possessions/g", False),
            ("adv", "pace", "pace (team)", False),
            ("ext", "pfd", "fouls drawn/g", False)],
    "stl": [("hustle", "defl_std", "deflections/g", True),
            ("hustle", "defl_l10_vs_std", "deflection momentum", True),
            ("hustle", "dloose_std", "def loose balls/g", False),
            ("hustle", "oloose_std", "off loose balls/g", False),
            ("ext", "def_rim_freq", "rim-defence frequency", False),
            ("hustle", "cont3_std", "contested 3s/g", False),
            ("hustle", "conttot_std", "total contests/g", False),
            ("defend", "dfga_overall_std", "defended FGA/g", False),
            ("defend", "dfga_fg3_std", "defended FG3A/g", False),
            ("ext", "pts_fb", "fast-break pts/g", True),
            ("ext", "pts_off_tov", "pts off turnovers/g", True),
            ("ext", "pct_stl", "steal share of team", True)],
    "blk": [("ext", "def_rim_fga", "defended rim FGA", False),
            ("ext", "def_rim_pct", "opp rim FG% allowed", False),
            ("ext", "def_rim_freq", "rim-defence frequency", False),
            ("ext", "def_rim_fgm", "rim makes allowed", True),
            ("hustle", "cont2_std", "contested 2s/g", False),
            ("hustle", "conttot_std", "total contests/g", False),
            ("defend", "dfga_overall_std", "defended FGA/g", False),
            ("defend", "dfga_fg2_std", "defended FG2A/g", False),
            ("ext", "opp_pts_paint", "opp paint pts/g", False),
            ("ext", "opp_pts_2nd_chance", "opp 2nd-chance pts/g", False),
            ("adv", "pace", "pace (team)", False)],
}


def _load_up(conn, table, cols, seasons):
    qs = ",".join("?" * len(seasons))
    sel = ", ".join(f'"{c}"' for c in cols)
    out = {}
    for r in conn.execute(
            f'SELECT season, snapshot_date, nba_api_id, {sel} FROM "{table}" '
            f'WHERE season IN ({qs})', seasons):
        out[(r["season"], r["snapshot_date"], r["nba_api_id"])] = {c: r[c] for c in cols}
    return out


def _resid(y, X):
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return y - X @ beta


def _pcorr(u, s, ctrls):
    u, s = np.asarray(u, float), np.asarray(s, float)
    X = np.column_stack([np.asarray(c, float) for c in ctrls] + [np.ones(len(u))])
    ru, rs = _resid(u, X), _resid(s, X)
    if ru.std() < 1e-9 or rs.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(ru, rs)[0, 1])


def run(conn, book, seasons):
    sc = STAT[book]
    counts, finals, _, _ = N._load(conn, seasons)
    mom = _load_up(conn, TAB["box"], [OWN_MOM[book]], seasons)

    fin_pg, fin_gp = {}, {}
    for (s, pid), d in finals.items():
        gp = d.get("gp_played_asof") or 0.0
        if gp >= 1:
            fin_pg[(s, pid)] = sc(d) / gp
            fin_gp[(s, pid)] = gp
    winner = {}
    for s in seasons:
        pool = [(pid, fin_pg[(x, pid)]) for (x, pid) in fin_pg if x == s
                and fin_gp[(x, pid)] >= 0.5 * max((fin_gp[(y, p)] for (y, p) in fin_gp if y == s), default=1)]
        if pool:
            winner[s] = max(pool, key=lambda kv: kv[1])[0]

    by_snap = {}
    for (s, snap, pid), d in counts.items():
        gp = d.get("gp_played_asof") or 0.0
        if gp < 5 or (s, pid) not in fin_pg:
            continue
        rem = fin_gp[(s, pid)] - gp
        if rem < 10:
            continue
        banked = sc(d) / gp
        surge = (sc(finals[(s, pid)]) - sc(d)) / rem - banked
        mr = mom.get((s, snap, pid))
        omv = mr.get(OWN_MOM[book]) if mr else None
        if omv is None:
            continue
        by_snap.setdefault((s, snap), []).append((pid, banked, surge, float(omv)))

    contenders = {k: sorted(v, key=lambda t: t[1], reverse=True)[:25] for k, v in by_snap.items()}

    caches = {}
    print("\n" + "=" * 78)
    print(f"book={book}  seasons={seasons[0]}-{seasons[-1]}   (r_lead controls banked + own momentum)")
    print(f"  {'upstream':<24}{'susp':>5}{'r_raw':>7}{'r_part':>8}{'r_lead':>8}{'win_elev':>9}{'n':>7}")
    for alias, col, label, suspect in CAND[book]:
        if alias not in caches:
            allcols = sorted({c for a, c, _, _ in CAND[book] if a == alias})
            try:
                caches[alias] = _load_up(conn, TAB[alias], allcols, seasons)
            except Exception:
                caches[alias] = None
        up = caches[alias]
        if up is None:
            print(f"  {label:<24}{'':>5}{'(table/col missing)':>30}")
            continue
        U, S, Bk, Om, elev = [], [], [], [], []
        for (s, snap), rows in contenders.items():
            vals = []
            for pid, bk, surge, om in rows:
                rec = up.get((s, snap, pid))
                v = rec.get(col) if rec else None
                if v is not None:
                    vals.append((pid, float(v), bk, surge, om))
            if not vals:
                continue
            for pid, v, bk, surge, om in vals:
                U.append(v); S.append(surge); Bk.append(bk); Om.append(om)
            if winner.get(s) is not None and len(vals) >= 5:
                fv = np.array([v for _, v, _, _, _ in vals])
                sd = fv.std()
                for pid, v, bk, surge, om in vals:
                    if pid == winner[s] and sd > 1e-9:
                        elev.append((v - fv.mean()) / sd)
        if len(U) < 40:
            print(f"  {label:<24}{('*' if suspect else ''):>5}{'n<40':>23}{len(U):>15}")
            continue
        rr = float(np.corrcoef(U, S)[0, 1])
        rp = _pcorr(U, S, [Bk])
        rl = _pcorr(U, S, [Bk, Om])
        we = float(np.mean(elev)) if elev else float("nan")
        print(f"  {label:<24}{('*' if suspect else ''):>5}{rr:>+7.2f}{rp:>+8.2f}{rl:>+8.2f}{we:>+9.2f}{len(U):>7}")
    print("  (* = shares measurement with the target; trust r_lead, not r_raw, for these)")


def _parse(s):
    if "-" in s:
        lo, hi = s.split("-"); return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in s.split(",")]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/awards.db")
    ap.add_argument("--stat", default="all",
                    choices=["reb", "pts", "ast", "stl", "blk", "all"])
    ap.add_argument("--seasons", default="2016-2023")
    ap.add_argument("--allow-sealed", action="store_true")
    a = ap.parse_args(argv)
    books = ["pts", "reb", "ast", "stl", "blk"] if a.stat == "all" else [a.stat]
    seasons = _parse(a.seasons)
    conn = connect(a.db)
    for book in books:
        yrs = [y for y in seasons if y >= STAT_FLOOR[book]]
        bad = [y for y in yrs if y in SEALED[book]]
        if bad and not a.allow_sealed:
            print(f"REFUSE {book}: sealed {bad}."); continue
        if yrs:
            try:
                run(conn, book, yrs)
            except Exception as e:
                print(f"book={book}: ERROR {e!r}")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
