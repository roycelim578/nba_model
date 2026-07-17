"""Walk-forward A/B for the volume mu overlay (STL, BLK, AST).

The overlay is a bounded blend on the EXISTING shrunk banked volume, never a
replacement:  v_star = w*v_hat + (1-w)*v_banked
  v_banked  banked per-game volume rate (the current prior's forward volume)
  v_hat     ElasticNet prediction from the surviving drivers (per-minute,
            position-controlled), fit walk-forward on prior seasons only
  w         blend weight; reported at 0 (banked only), 0.5, and w* fitted on the
            training seasons, so you see whether the regression earns more than a
            coin-flip. If it never beats w=0, the overlay is inert and drops out.

Volume legs / drivers (def_rim_fga removed from STL; no rim path to steals):
  stl  deflections/g   <- cont3, dfga_fg3, dpct_overall, dloose
  blk  rim shots faced <- dfga_fg2, pfd
  ast  potential ast/g <- pace

Target: realised remaining per-game volume rate. Metric: RMSE, pooled over the
walk-forward eval seasons, at each w. Lower is better; w=0 is the incumbent.

Run:
  uv run python3 -m scripts.modelling.stat_leader.volume_overlay_ab --seasons 2016-2023
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict

import numpy as np

try:
    from sklearn.linear_model import ElasticNetCV
    from sklearn.model_selection import cross_val_predict
except ImportError:
    ElasticNetCV = None
try:
    from scripts.common.db import connect
    from scripts.features.stat_leader import nodes as N
except ImportError:  # pragma: no cover
    from db import connect  # type: ignore
    import nodes as N  # type: ignore

SEALED = {"stl": {2024, 2025}, "blk": {2024, 2025}, "ast": {2025}}
# book -> (volume col, table, key) and drivers [(table, col, is_rate)]
VOL = {"stl": ("defl_std", "stg_nba_hustle_asof"),
       "blk": ("def_rim_fga", "stg_nba_player_asof_ext"),
       "ast": ("potential_ast_asof", "counts")}
DRV = {"stl": [("stg_nba_hustle_asof", "cont3_std", False),
               ("stg_nba_defend_asof", "dfga_fg3_std", False),
               ("stg_nba_defend_asof", "dpct_overall_std", True),
               ("stg_nba_hustle_asof", "dloose_std", False)],
       "blk": [("stg_nba_defend_asof", "dfga_fg2_std", False),
               ("stg_nba_player_asof_ext", "pfd", False)],
       "ast": [("stg_nba_player_advanced_asof", "pace", True),
               ("__team__", "team_fg3_pct", True)]}


def _team_fg3(conn, seasons):
    qs = ",".join("?" * len(seasons))
    fg = {}
    for r in conn.execute(f'SELECT season, snapshot_date, team_id, team_fg3m_asof, '
                          f'team_fg3a_asof FROM stat_team_fg_asof WHERE season IN ({qs})', seasons):
        a = r["team_fg3a_asof"] or 0.0
        if a > 0:
            fg[(r["season"], r["snapshot_date"], r["team_id"])] = {
                "team_fg3_pct": (r["team_fg3m_asof"] or 0.0) / a}
    tmap = {}
    for r in conn.execute(f'SELECT season, nba_api_id, team_id, COUNT(*) c FROM '
                          f'stg_nba_player_game_logs WHERE season IN ({qs}) '
                          f'GROUP BY season, nba_api_id, team_id', seasons):
        k = (r["season"], r["nba_api_id"])
        if k not in tmap or r["c"] > tmap[k][1]:
            tmap[k] = (r["team_id"], r["c"])
    return fg, {k: v[0] for k, v in tmap.items()}


def _load(conn, t, cols, seasons):
    qs = ",".join("?" * len(seasons))
    sel = ", ".join(f'"{c}"' for c in cols)
    out = {}
    for r in conn.execute(f'SELECT season, snapshot_date, nba_api_id, {sel} FROM "{t}" '
                          f'WHERE season IN ({qs})', seasons):
        out[(r["season"], r["snapshot_date"], r["nba_api_id"])] = {c: r[c] for c in cols}
    return out


def _vol_pg(book, d, aux, key, gp):
    col = VOL[book][0]
    if book == "ast":
        v = d.get("potential_ast_asof")
        return None if v is None else v / gp
    rec = aux.get(key)
    v = rec.get(col) if rec else None
    return None if v is None else float(v)  # already per-game (_std / def_rim_fga per game)


def _zwithin(x, keys):
    x = np.asarray(x, float); out = np.zeros_like(x)
    idx = defaultdict(list)
    for i, k in enumerate(keys):
        idx[k].append(i)
    for k, ii in idx.items():
        v = x[ii]; sd = v.std(); out[ii] = (v - v.mean()) / sd if sd > 1e-9 else 0.0
    return out


def run(conn, book, seasons):
    counts, finals, _, _ = N._load(conn, seasons)
    box = _load(conn, "stg_nba_box_asof", ["mpg_std"], seasons)
    pos = {r["nba_api_id"]: (r["position"] or "unk")
           for r in conn.execute("SELECT nba_api_id, position FROM player_position_map")}
    volt = VOL[book][1]
    aux = {} if volt == "counts" else _load(conn, volt, [VOL[book][0]], seasons)
    drv_tabs = {t for t, _, _ in DRV[book]}
    dcache = {t: _load(conn, t, [c for tt, c, _ in DRV[book] if tt == t], seasons)
              for t in drv_tabs if t != "__team__"}
    if "__team__" in drv_tabs:
        fgby, tmap2 = _team_fg3(conn, seasons)
        td = {}
        for (s, snap, pid) in counts:
            tid = tmap2.get((s, pid))
            rec = fgby.get((s, snap, tid)) if tid is not None else None
            if rec:
                td[(s, snap, pid)] = rec
        dcache["__team__"] = td
    last = {}
    for (s, snap, pid) in counts:
        k = (s, pid)
        if k not in last or snap > last[k]:
            last[k] = snap

    recs = []  # (season, snap, pid, y, v_banked, [driver vals], pos, mpg)
    bysnap = defaultdict(list)
    for (s, snap, pid), d in counts.items():
        gp = d.get("gp_played_asof") or 0.0
        fk = (s, pid)
        if gp < 5 or fk not in last:
            continue
        fd = finals.get(fk); fgp = (fd.get("gp_played_asof") or 0.0) if fd else 0.0
        rem = fgp - gp
        if rem < 10:
            continue
        vb = _vol_pg(book, d, aux, (s, snap, pid), gp)
        fsnap = last[fk]
        vf = _vol_pg(book, fd, aux, (s, fsnap, pid), fgp)
        if vb is None or vf is None:
            continue
        y = (vf * fgp - vb * gp) / rem
        mpg = (box.get((s, snap, pid)) or {}).get("mpg_std")
        if not mpg or mpg <= 0:
            continue
        dv = []
        ok = True
        for t, c, is_rate in DRV[book]:
            rec = dcache[t].get((s, snap, pid)); v = rec.get(c) if rec else None
            if v is None:
                ok = False; break
            dv.append(float(v) if is_rate else float(v) / mpg)
        if not ok:
            continue
        recs.append([s, snap, pid, y, vb, dv, pos.get(pid, "unk")])
        bysnap[(s, snap)].append((pid, vb))
    top = set()
    for key, lst in bysnap.items():
        lst.sort(key=lambda t: t[1], reverse=True)
        for pid, _ in lst[:25]:
            top.add((key[0], key[1], pid))
    recs = [r for r in recs if (r[0], r[1], r[2]) in top]
    if len(recs) < 200 or ElasticNetCV is None:
        print(f"{book}: insufficient rows or sklearn missing (n={len(recs)})"); return

    evals = sorted({r[0] for r in recs})
    W = [0.0, 0.5, None]  # None = fitted w*
    err = {w: [] for w in ("0", "0.5", "w*")}
    wstars = []
    for Y in evals:
        tr = [r for r in recs if r[0] < Y]
        ev = [r for r in recs if r[0] == Y]
        if len(tr) < 150 or not ev:
            continue
        cats = sorted({r[6] for r in tr})
        def feats(rows):
            X = []
            for r in rows:
                dum = [1.0 if r[6] == c else 0.0 for c in cats[1:]]
                X.append(r[5] + dum)
            return np.array(X, float)
        Xtr, ytr = feats(tr), np.array([r[3] for r in tr])
        Xtr = _std_cols(Xtr)
        m = ElasticNetCV(l1_ratio=[.2, .5, .8], cv=4, max_iter=5000)
        m.fit(Xtr, ytr)
        vhat_tr = cross_val_predict(m, Xtr, ytr, cv=4)
        vb_tr = np.array([r[4] for r in tr])
        ws = np.linspace(0, 1, 21)
        wstar = ws[np.argmin([np.mean(((w * vhat_tr + (1 - w) * vb_tr) - ytr) ** 2) for w in ws])]
        wstars.append(wstar)
        Xev = _std_cols(feats(ev), ref=Xtr_ref(Xtr))
        vhat_ev = m.predict(Xev)
        vb_ev = np.array([r[4] for r in ev]); yev = np.array([r[3] for r in ev])
        for lbl, w in (("0", 0.0), ("0.5", 0.5), ("w*", wstar)):
            err[lbl].extend(((w * vhat_ev + (1 - w) * vb_ev) - yev) ** 2)
    print(f"\n{book}-vol overlay  (walk-forward eval {evals[1] if len(evals)>1 else evals[0]}-{evals[-1]})")
    for lbl in ("0", "0.5", "w*"):
        e = err[lbl]
        if e:
            print(f"  w={lbl:<3} RMSE {np.sqrt(np.mean(e)):.4f}   n={len(e)}")
    if wstars:
        print(f"  fitted w* per season: {[round(x,2) for x in wstars]}")


_MU = {}
def _std_cols(X, ref=None):
    if ref is None:
        mu = X.mean(0); sd = X.std(0); sd[sd < 1e-9] = 1.0
        _MU["mu"], _MU["sd"] = mu, sd
        return (X - mu) / sd
    return (X - _MU["mu"]) / _MU["sd"]
def Xtr_ref(X):
    return X


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/awards.db")
    ap.add_argument("--seasons", default="2016-2023")
    ap.add_argument("--stat", default="all", choices=["stl", "blk", "ast", "all"])
    a = ap.parse_args(argv)
    lo, hi = (a.seasons.split("-") + [a.seasons])[:2]
    seasons = list(range(int(lo), int(hi) + 1))
    books = ["stl", "blk", "ast"] if a.stat == "all" else [a.stat]
    conn = connect(a.db)
    for book in books:
        yrs = [y for y in seasons if y not in SEALED[book]]
        if yrs != seasons:
            print(f"{book}: dev seasons {yrs} (sealed dropped)")
        try:
            run(conn, book, yrs)
        except Exception as e:
            print(f"{book}: ERROR {e!r}")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
