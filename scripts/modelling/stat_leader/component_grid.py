"""Volume-leg grid, position-normalised and per-minute (v4).

Volume drives mu variance, so this drops the conversion legs entirely and scans
predictors of the VOLUME leg only (plus reb/oreb/dreb direct rates).

Confounder removal, per Royce: the top factors were positionally confounded
(a centre rebounds and blocks more, a guard steals and assists more, by role and
by minutes, not by any lead). So every test now strips:
  - minutes LEVEL   : count-like predictors are put per-minute (/ mpg); already
                      relative features (percentages, *_l10_vs_*, ratings, pace,
                      pie, ratios, team_*) are left as-is
  - minutes TREND   : minutes momentum (mpg_l10_vs_std) added as a control
  - own level/trend : banked leg rate + own rate momentum, as before
  - position        : guard/wing/big dummies added as controls

r_lead = partial corr of the (per-minute) predictor with the volume surge given
[banked, own-mom, minutes-mom, position]; r_z = same with the predictor first
standardised within each snapshot. Team features are printed unfiltered per
target. '*' flags same-measurement / construction inputs.

Run:
  caffeinate -i uv run python3 -m scripts.modelling.stat_leader.component_grid \
      --seasons 2016-2023
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict

import numpy as np

try:
    from scripts.common.db import connect
    from scripts.features.stat_leader import nodes as N
except ImportError:  # pragma: no cover
    from db import connect  # type: ignore
    import nodes as N  # type: ignore

MIN_R = 0.10
MIN_N = 200
TOP = 24
ROLLING = {"stg_nba_box_asof", "stg_nba_hustle_asof", "stg_nba_defend_asof"}
SKIP_COLS = {"season", "snapshot_date", "nba_api_id", "team_id", "game_id",
             "pulled_at", "gp", "min", "gp_asof", "gp_played_asof", "n_games_asof",
             "plus_minus", "def_ws"}
SKIP_BASE = {"mpg"}
OWN_MOM = {"pts": "ppg_l10_vs_std", "ast": "apg_l10_vs_std", "stl": "spg_l10_vs_std",
           "blk": "bpg_l10_vs_std", "reb": "rpg_l10_vs_std"}
BOX_EXTRA = ["mpg_std", "mpg_l10_vs_std"]
TEAM_COLS = ["team_off_poss", "team_def_poss", "team_fga_pg", "team_fg_pct",
             "team_fg3a_pg", "team_fg3_pct"]
RATE_HINTS = ("pct", "_l10_vs", "avg_sec", "avg_drib", "ast_to", "pace", "pie",
              "rating", "freq", "efg")
SUSP = {
    "pts": ("ppg", "pts", "usg", "efg", "ts_pct", "fg_pct", "fg3_pct", "ft_pct", "pra", "fga", "fta"),
    "ast": ("apg", "ast", "potential_ast", "secondary_ast", "pra"),
    "stl": ("spg", "stl", "steal", "defl", "pra"),
    "blk": ("bpg", "_blk", "blk_", "block", "def_rim", "cont"),
    "reb": ("rpg", "reb", "oreb", "dreb", "pra"),
    "oreb": ("oreb", "reb", "rpg", "pra", "2nd_chance"),
    "dreb": ("dreb", "reb", "rpg", "pra"),
}


def _pts(d):
    return 2.0 * (d.get("fg2m") or 0.0) + 3.0 * (d.get("fg3m") or 0.0) + (d.get("ftm") or 0.0)


def _tsa(d):
    return (d.get("fg2a") or 0.0) + (d.get("fg3a") or 0.0) + 0.44 * (d.get("fta") or 0.0)


STAT_TOT = {"pts": _pts, "ast": lambda d: d.get("ast") or 0.0,
            "stl": lambda d: d.get("stl") or 0.0, "blk": lambda d: d.get("blk") or 0.0}


def _is_rate(col):
    if col.startswith("team_"):
        return True
    c = col.lower()
    return any(h in c for h in RATE_HINTS)


def _suspect(key, col):
    if col.startswith("team_"):
        return False
    book = key.split("-")[0]
    c = col.lower()
    return any(tok in c for tok in SUSP[book])


def _resid(y, X):
    return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]


def _pc(u, s, ctrls):
    u, s = np.asarray(u, float), np.asarray(s, float)
    cols = [np.asarray(c, float) for c in ctrls]
    cols = [c for c in cols if c.std() > 1e-12]  # drop constant controls (e.g. absent pos)
    X = np.column_stack(cols + [np.ones(len(u))])
    ru, rs = _resid(u, X), _resid(s, X)
    if ru.std() < 1e-9 or rs.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(ru, rs)[0, 1])


def _zwithin(u, keys):
    u = np.asarray(u, float)
    out = np.zeros_like(u)
    idx = defaultdict(list)
    for i, k in enumerate(keys):
        idx[k].append(i)
    for k, ii in idx.items():
        v = u[ii]; sd = v.std()
        out[ii] = (v - v.mean()) / sd if sd > 1e-9 else 0.0
    return out


def _cols(conn, t):
    return [r[1] for r in conn.execute(f'PRAGMA table_info("{t}")')]


def _load_cols(conn, t, cols, seasons):
    qs = ",".join("?" * len(seasons))
    sel = ", ".join(f'"{c}"' for c in cols)
    out = {}
    for r in conn.execute(
            f'SELECT season, snapshot_date, nba_api_id, {sel} FROM "{t}" '
            f'WHERE season IN ({qs})', seasons):
        out[(r["season"], r["snapshot_date"], r["nba_api_id"])] = {c: r[c] for c in cols}
    return out


def _candidate_cols(conn, t):
    cols = _cols(conn, t)
    picked = []
    if t in ROLLING:
        for b in sorted({c[:-4] for c in cols if c.endswith("_std")}):
            if b in SKIP_BASE:
                continue
            picked.append(b + "_std")
            if (b + "_l10_vs_std") in cols:
                picked.append(b + "_l10_vs_std")
    else:
        for c in cols:
            if c in SKIP_COLS or c in SKIP_BASE:
                continue
            if re.search(r"_(std|ema|l5|l10|l20|l30|l10_vs_l30|l10_vs_std)$", c):
                continue
            picked.append(c)
    return list(dict.fromkeys(picked))


def _build_team(conn, seasons):
    qs = ",".join("?" * len(seasons))
    fg = {}
    for r in conn.execute(
            f'SELECT season, snapshot_date, team_id, team_fgm_asof, team_fga_asof, '
            f'team_fg3m_asof, team_fg3a_asof, team_games_asof FROM stat_team_fg_asof '
            f'WHERE season IN ({qs})', seasons):
        g = r["team_games_asof"] or 0
        fga = r["team_fga_asof"] or 0.0
        f3a = r["team_fg3a_asof"] or 0.0
        if g < 1:
            continue
        fg[(r["season"], r["snapshot_date"], r["team_id"])] = {
            "team_fga_pg": fga / g,
            "team_fg_pct": (r["team_fgm_asof"] or 0.0) / fga if fga > 0 else None,
            "team_fg3a_pg": f3a / g,
            "team_fg3_pct": (r["team_fg3m_asof"] or 0.0) / f3a if f3a > 0 else None,
        }
    snaps = defaultdict(set)
    for (s, snap, tid) in fg:
        snaps[s].add(snap)
    tg = defaultdict(list)
    for r in conn.execute(
            f'SELECT season, team_id, game_date, off_poss, def_poss FROM stat_team_game '
            f'WHERE season IN ({qs})', seasons):
        tg[(r["season"], r["team_id"])].append((r["game_date"], r["off_poss"], r["def_poss"]))
    poss = {}
    for (s, tid), games in tg.items():
        games.sort(key=lambda x: x[0] or "")
        gi = cg = cgd = 0
        coff = cdef = 0.0
        for snap in sorted(snaps.get(s, [])):
            while gi < len(games) and games[gi][0] is not None and games[gi][0] <= snap:
                _, op, dp = games[gi]
                if op is not None:
                    coff += op; cg += 1
                if dp is not None:
                    cdef += dp; cgd += 1
                gi += 1
            poss[(s, snap, tid)] = (coff / cg if cg else None, cdef / cgd if cgd else None)
    out = {}
    for k in set(fg) | set(poss):
        d = dict(fg.get(k, {}))
        op, dp = poss.get(k, (None, None))
        d["team_off_poss"] = op; d["team_def_poss"] = dp
        for c in TEAM_COLS:
            d.setdefault(c, None)
        out[k] = d
    tmap = {}
    for r in conn.execute(
            f'SELECT season, nba_api_id, team_id, COUNT(*) c FROM stg_nba_player_game_logs '
            f'WHERE season IN ({qs}) GROUP BY season, nba_api_id, team_id', seasons):
        k = (r["season"], r["nba_api_id"])
        if k not in tmap or r["c"] > tmap[k][1]:
            tmap[k] = (r["team_id"], r["c"])
    return out, {k: v[0] for k, v in tmap.items()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/awards.db")
    ap.add_argument("--seasons", default="2016-2023")
    a = ap.parse_args(argv)
    lo, hi = (a.seasons.split("-") + [a.seasons])[:2]
    seasons = list(range(int(lo), int(hi) + 1))

    conn = connect(a.db)
    counts, finals, _, _ = N._load(conn, seasons)
    dreb = _load_cols(conn, "stg_nba_player_advanced_asof", ["dreb"], seasons)
    defl = _load_cols(conn, "stg_nba_hustle_asof", ["defl_std"], seasons)
    rim = _load_cols(conn, "stg_nba_player_asof_ext", ["def_rim_fga"], seasons)
    box = _load_cols(conn, "stg_nba_box_asof",
                     sorted(set(OWN_MOM.values()) | set(BOX_EXTRA)), seasons)
    pos_map = {r["nba_api_id"]: (r["position"] or "unk")
               for r in conn.execute("SELECT nba_api_id, position FROM player_position_map")}

    team_asof, tmap = _build_team(conn, seasons)
    team_data = {}
    for (s, snap, pid) in counts:
        tid = tmap.get((s, pid))
        ta = team_asof.get((s, snap, tid)) if tid is not None else None
        if ta:
            team_data[(s, snap, pid)] = ta

    last = {}
    for (s, snap, pid) in counts:
        k = (s, pid)
        if k not in last or snap > last[k]:
            last[k] = snap

    def opp_tot(book, s, snap, pid, d, gp):
        if book == "pts":
            return _tsa(d)
        if book == "ast":
            return d.get("potential_ast_asof")
        if book == "stl":
            r = defl.get((s, snap, pid))
            return None if not r or r.get("defl_std") is None else r["defl_std"] * gp
        if book == "blk":
            r = rim.get((s, snap, pid))
            return None if not r or r.get("def_rim_fga") is None else r["def_rim_fga"] * gp
        return None

    tables = ["stg_nba_box_asof", "stg_nba_player_advanced_asof",
              "stg_nba_player_asof_ext", "stg_nba_hustle_asof", "stg_nba_defend_asof"]
    catalog = {}
    for t in tables:
        try:
            cc = _candidate_cols(conn, t)
            catalog[t] = (cc, _load_cols(conn, t, cc, seasons))
        except Exception as e:
            print(f"skip {t}: {e!r}")
    catalog["__team__"] = (TEAM_COLS, team_data)
    tables = tables + ["__team__"]

    def contenders_by(rate_of):
        by = defaultdict(list)
        for (s, snap, pid), d in counts.items():
            gp = d.get("gp_played_asof") or 0.0
            if gp < 5 or (s, pid) not in last:
                continue
            r = rate_of(s, snap, pid, d, gp)
            if r is not None:
                by[(s, snap)].append((pid, r))
        keep = set()
        for key, lst in by.items():
            lst.sort(key=lambda t: t[1], reverse=True)
            for pid, _ in lst[:25]:
                keep.add((key[0], key[1], pid))
        return keep

    def _posdummies(P):
        cats = sorted(set(P))
        return [[1.0 if p == c else 0.0 for p in P] for c in cats[1:]]

    def scan(key, surge_of, contender_rate):
        keep = contenders_by(contender_rate)
        rowmap = {}
        for k in keep:
            r = surge_of(*k)
            if r is not None:
                rowmap[k] = r
        if not rowmap:
            print(f"\n{key}: no rows"); return
        n_extra = len(next(iter(rowmap.values()))) - 3
        results, team_rows, seen = [], [], set()
        for t in tables:
            if t not in catalog:
                continue
            cc, data = catalog[t]
            src = "team" if t == "__team__" else t.replace("stg_nba_", "").replace("_asof", "")
            for col in cc:
                if (col, src) in seen:
                    continue
                rate_feat = _is_rate(col)
                U, S, B, M, MM, P, K = [], [], [], [], [], [], []
                EX = [[] for _ in range(n_extra)]
                for k, row in rowmap.items():
                    rec = data.get(k)
                    v = rec.get(col) if rec else None
                    if v is None:
                        continue
                    bx = box.get(k)
                    mpg = bx.get("mpg_std") if bx else None
                    mmv = bx.get("mpg_l10_vs_std") if bx else None
                    if mpg is None or mmv is None or mpg <= 0:
                        continue
                    x = float(v) if rate_feat else float(v) / mpg
                    U.append(x); S.append(row[1]); B.append(row[0]); M.append(row[2])
                    MM.append(float(mmv)); P.append(pos_map.get(k[2], "unk"))
                    K.append((k[0], k[1]))
                    for j in range(n_extra):
                        EX[j].append(row[3 + j])
                if len(U) < MIN_N:
                    continue
                ctrls = [B, M, MM] + _posdummies(P) + EX
                rl = _pc(U, S, ctrls)
                rz = _pc(_zwithin(U, K), S, ctrls)
                a_rl = 0.0 if np.isnan(rl) else abs(rl)
                a_rz = 0.0 if np.isnan(rz) else abs(rz)
                if src == "team":
                    team_rows.append((col, rl, rz, len(U)))
                if max(a_rl, a_rz) < MIN_R:
                    continue
                seen.add((col, src))
                results.append((a_rz, rl, rz, col, src, _suspect(key, col), len(U), rate_feat))
        results.sort(key=lambda x: (-x[0], -(0.0 if np.isnan(x[1]) else abs(x[1]))))
        print("\n" + "=" * 74)
        print(f"{key}-vol   ({len(results)} factors; top {TOP}, |r_z| sorted; pm=per-minute)")
        print(f"  {'r_lead':>7}{'r_z':>8}  {'susp':>4} {'pm':>3}  {'factor':<22}{'source':<12}{'n':>6}")
        for a_rz, rl, rz, col, src, susp, n, rate_feat in results[:TOP]:
            pm = "" if rate_feat else "pm"
            print(f"  {rl:>+7.2f}{rz:>+8.2f}  {'*' if susp else '':>4} {pm:>3}  {col:<22}{src:<12}{n:>6}")
        order = {c: i for i, c in enumerate(TEAM_COLS)}
        team_rows.sort(key=lambda x: order.get(x[0], 99))
        print("  -- team features (unfiltered) --")
        for col, rl, rz, n in team_rows:
            print(f"  {rl:>+7.2f}{rz:>+8.2f}            {col:<22}{'team':<12}{n:>6}")

    def make_vol(book):
        sc = STAT_TOT[book]; mcol = OWN_MOM[book]

        def crate(s, snap, pid, d, gp):
            return sc(d) / gp

        def surge_of(s, snap, pid):
            d = counts.get((s, snap, pid)); fk = (s, pid)
            fsnap = last.get(fk); fd = finals.get(fk)
            if d is None or fd is None or fsnap is None:
                return None
            gp = d.get("gp_played_asof") or 0.0
            fgp = fd.get("gp_played_asof") or 0.0
            rem = fgp - gp
            if gp < 5 or rem < 10:
                return None
            ob = opp_tot(book, s, snap, pid, d, gp)
            of = opp_tot(book, s, fsnap, pid, fd, fgp)
            if ob is None or of is None or ob <= 0 or (of - ob) <= 0:
                return None
            bx = box.get((s, snap, pid)); mv = bx.get(mcol) if bx else None
            if mv is None:
                return None
            b = ob / gp
            return (b, (of - ob) / rem - b, float(mv))

        return surge_of, crate

    def make_direct(book):
        sc = STAT_TOT[book]; mcol = OWN_MOM[book]

        def crate(s, snap, pid, d, gp):
            return sc(d) / gp

        def surge_of(s, snap, pid):
            d = counts.get((s, snap, pid)); fk = (s, pid)
            fsnap = last.get(fk); fd = finals.get(fk)
            if d is None or fd is None or fsnap is None:
                return None
            gp = d.get("gp_played_asof") or 0.0
            fgp = fd.get("gp_played_asof") or 0.0
            rem = fgp - gp
            if gp < 5 or rem < 10:
                return None
            b = sc(d) / gp
            f = sc(fd) / fgp
            bx = box.get((s, snap, pid)); mv = bx.get(mcol) if bx else None
            if mv is None:
                return None
            return (b, (f * fgp - b * gp) / rem - b, float(mv))

        return surge_of, crate

    for book in ("stl", "blk", "ast", "pts"):
        so, cr = make_direct(book) if book == "blk" else make_vol(book)
        try:
            scan(book, so, cr)
        except Exception as e:
            print(f"{book}: ERROR {e!r}")

    def make_conv(book):
        sc = STAT_TOT[book]; mcol = OWN_MOM[book]

        def crate(s, snap, pid, d, gp):
            return sc(d) / gp

        def surge_of(s, snap, pid):
            d = counts.get((s, snap, pid)); fk = (s, pid)
            fsnap = last.get(fk); fd = finals.get(fk)
            if d is None or fd is None or fsnap is None:
                return None
            gp = d.get("gp_played_asof") or 0.0
            fgp = fd.get("gp_played_asof") or 0.0
            rem = fgp - gp
            if gp < 5 or rem < 10:
                return None
            ob = opp_tot(book, s, snap, pid, d, gp)
            of = opp_tot(book, s, fsnap, pid, fd, fgp)
            if ob is None or of is None or ob <= 0 or (of - ob) <= 0:
                return None
            sb, sf = sc(d), sc(fd)
            if (sf - sb) < 0:
                return None
            bx = box.get((s, snap, pid)); mv = bx.get(mcol) if bx else None
            if mv is None:
                return None
            b = sb / ob
            return (b, (sf - sb) / (of - ob) - b, float(mv))

        return surge_of, crate

    so, cr = make_conv("blk")
    try:
        scan("blk-conv", so, cr)
    except Exception as e:
        print(f"blk-conv: ERROR {e!r}")

    mcol = OWN_MOM["reb"]

    def reb_pure():
        def crate(s, snap, pid, d, gp):
            return (d.get("reb") or 0.0) / gp

        def surge_of(s, snap, pid):
            d = counts.get((s, snap, pid)); fk = (s, pid)
            fsnap = last.get(fk); fd = finals.get(fk)
            if d is None or fd is None or fsnap is None:
                return None
            gp = d.get("gp_played_asof") or 0.0
            fgp = fd.get("gp_played_asof") or 0.0
            rem = fgp - gp
            if gp < 5 or rem < 10:
                return None
            bt, ft = d.get("reb") or 0.0, fd.get("reb") or 0.0
            bx = box.get((s, snap, pid)); mv = bx.get(mcol) if bx else None
            if mv is None:
                return None
            b = bt / gp
            return (b, (ft - bt) / rem - b, float(mv))

        return surge_of, crate

    def reb_comp(comp):
        def crate(s, snap, pid, d, gp):
            dr = dreb.get((s, snap, pid))
            if not dr or dr.get("dreb") is None:
                return None
            reb = d.get("reb") or 0.0
            return (dr["dreb"] if comp == "dreb" else reb - dr["dreb"]) / gp

        def surge_of(s, snap, pid):
            d = counts.get((s, snap, pid)); fk = (s, pid)
            fsnap = last.get(fk); fd = finals.get(fk)
            if d is None or fd is None or fsnap is None:
                return None
            gp = d.get("gp_played_asof") or 0.0
            fgp = fd.get("gp_played_asof") or 0.0
            rem = fgp - gp
            if gp < 5 or rem < 10:
                return None
            drb = dreb.get((s, snap, pid)); drf = dreb.get((s, fsnap, pid))
            if not drb or not drf or drb.get("dreb") is None or drf.get("dreb") is None:
                return None
            rb, rf = d.get("reb") or 0.0, fd.get("reb") or 0.0
            if comp == "dreb":
                bt, ft, other = drb["dreb"], drf["dreb"], (rb - drb["dreb"]) / gp
            else:
                bt, ft, other = rb - drb["dreb"], rf - drf["dreb"], drb["dreb"] / gp
            bx = box.get((s, snap, pid)); mv = bx.get(mcol) if bx else None
            if mv is None:
                return None
            b = bt / gp
            return (b, (ft - bt) / rem - b, float(mv), other)

        return surge_of, crate

    for label, mk in (("reb", reb_pure), ("oreb", lambda: reb_comp("oreb")),
                      ("dreb", lambda: reb_comp("dreb"))):
        so, cr = mk()
        try:
            scan(label, so, cr)
        except Exception as e:
            print(f"{label}: ERROR {e!r}")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
