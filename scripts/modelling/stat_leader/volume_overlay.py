"""Production volume mu-overlay: fit, persist, apply.

Turns the walk-forward A/B (volume_overlay_ab.py) into a reusable engine. Per book
with a registered overlay it fits, on training seasons only, a shrunk ElasticNet of
the volume-leg forward per-game rate on standardised drivers plus position dummies,
and stores the coefficients, the standardisation, the fitted blend weight w*, and the
out-of-sample residual SD s_hat. At inference `apply` returns the bounded blend
    v_star = w* * v_hat + (1 - w*) * v_banked
and s_hat, so the mc.py volume leg can use v_star as its forward mean and inflate the
draw width by s_hat. Degrades to the incumbent at w*=0. Overlay sits ON TOP of the
existing shrunk (cohort + own-history) banked volume; it never replaces it.

Volume legs / drivers (per the shipped decisions; def_rim_fga removed from STL):
  stl  deflections/g       <- cont3, dfga_fg3, dpct_overall, dloose
  blk  rim shots faced/g   <- dfga_fg2, pfd
  ast  potential assists/g <- pace, team_fg3_pct

Caveats baked in: def_rim_fga and potential_ast_asof are CUMULATIVE (divide by gp for
a per-game rate); the "_std" hustle/defend columns are already per-game.

Seals respected: stl/blk hold out 2024+2025, ast holds out 2025; fits never see them.

Run:
  uv run python3 -m scripts.modelling.stat_leader.volume_overlay --stat all --fit --seasons 2016-2023
  uv run python3 -m scripts.modelling.stat_leader.volume_overlay --stat stl --seasons 2016-2023   # report only
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from collections import defaultdict

import numpy as np

try:
    from sklearn.linear_model import ElasticNetCV
    from sklearn.model_selection import cross_val_predict
except ImportError:  # pragma: no cover
    ElasticNetCV = None
try:
    from scripts.common.db import connect
    from scripts.features.stat_leader import nodes as N
except ImportError:  # pragma: no cover
    from db import connect  # type: ignore
    import nodes as N  # type: ignore

ARTEFACT_DIR = "models/stat_leader/overlay"
SEALED = {"stl": {2024, 2025}, "blk": {2024, 2025}, "ast": {2025}}

# book -> (volume col, source table). "counts" means the nodes substrate.
VOL = {"stl": ("defl_std", "stg_nba_hustle_asof"),
       "blk": ("blk", "counts"),
       "ast": ("potential_ast_asof", "counts")}
# book -> [(table, col, is_rate)]; is_rate stays as-is, else divided by mpg (per-minute).
# BLK: the two-part rim-FGA decomposition lost on the P(lead) scorecard, so BLK stays a
# DIRECT block-rate model and the overlay tilts that direct volume, driver = pfd only.
DRV = {"stl": [("stg_nba_hustle_asof", "cont3_std", False),
               ("stg_nba_defend_asof", "dfga_fg3_std", False),
               ("stg_nba_defend_asof", "dpct_overall_std", True),
               ("stg_nba_hustle_asof", "dloose_std", False)],
       "blk": [("stg_nba_player_asof_ext", "pfd", False)],
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
    """Forward volume as a per-game rate. Substrate ("counts") cols are cumulative
    season-to-date so divide by gp; table cols ("_std") are already per-game."""
    col, src = VOL[book]
    if src == "counts":
        v = d.get(col)
        return (v / gp) if (v is not None and gp) else None
    rec = aux.get(key)
    v = rec.get(col) if rec else None
    return None if v is None else float(v)


def _driver_vec(book, s, snap, pid, dcache, mpg):
    """Standardised-per-game driver vector for one contender, or None if incomplete."""
    dv = []
    for t, c, is_rate in DRV[book]:
        rec = dcache[t].get((s, snap, pid))
        v = rec.get(c) if rec else None
        if v is None:
            return None
        dv.append(float(v) if is_rate else float(v) / mpg)
    return dv


def _driver_cache(conn, book, seasons, counts):
    tabs = {t for t, _, _ in DRV[book]}
    dc = {t: _load(conn, t, [c for tt, c, _ in DRV[book] if tt == t], seasons)
          for t in tabs if t != "__team__"}
    if "__team__" in tabs:
        fgby, tmap = _team_fg3(conn, seasons)
        td = {}
        for (s, snap, pid) in counts:
            tid = tmap.get((s, pid))
            rec = fgby.get((s, snap, tid)) if tid is not None else None
            if rec:
                td[(s, snap, pid)] = rec
        dc["__team__"] = td
    return dc


def build_records(conn, book, seasons):
    """(season, snap, pid, y_forward_vol_rate, v_banked, driver_vec, pos) rows,
    top-25 by banked volume per snapshot, gp>=5, remaining>=10. Mirrors the A/B."""
    counts, finals, _, _ = N._load(conn, seasons)
    box = _load(conn, "stg_nba_box_asof", ["mpg_std"], seasons)
    pos = {r["nba_api_id"]: (r["position"] or "unk")
           for r in conn.execute("SELECT nba_api_id, position FROM player_position_map")}
    volt = VOL[book][1]
    aux = {} if volt == "counts" else _load(conn, volt, [VOL[book][0]], seasons)
    dcache = _driver_cache(conn, book, seasons, counts)
    last = {}
    for (s, snap, pid) in counts:
        k = (s, pid)
        if k not in last or snap > last[k]:
            last[k] = snap
    recs, bysnap = [], defaultdict(list)
    for (s, snap, pid), d in counts.items():
        gp = d.get("gp_played_asof") or 0.0
        fk = (s, pid)
        if gp < 5 or fk not in last:
            continue
        fd = finals.get(fk)
        fgp = (fd.get("gp_played_asof") or 0.0) if fd else 0.0
        rem = fgp - gp
        if rem < 10:
            continue
        vb = _vol_pg(book, d, aux, (s, snap, pid), gp)
        vf = _vol_pg(book, fd, aux, (s, last[fk], pid), fgp)
        if vb is None or vf is None:
            continue
        y = (vf * fgp - vb * gp) / rem
        mpg = (box.get((s, snap, pid)) or {}).get("mpg_std")
        if not mpg or mpg <= 0:
            continue
        dv = _driver_vec(book, s, snap, pid, dcache, mpg)
        if dv is None:
            continue
        recs.append([s, snap, pid, y, vb, dv, pos.get(pid, "unk")])
        bysnap[(s, snap)].append((pid, vb))
    top = set()
    for key, lst in bysnap.items():
        lst.sort(key=lambda t: t[1], reverse=True)
        for pid, _ in lst[:25]:
            top.add((key[0], key[1], pid))
    return [r for r in recs if (r[0], r[1], r[2]) in top]


def _design(recs, cats):
    X = []
    for r in recs:
        dum = [1.0 if r[6] == c else 0.0 for c in cats[1:]]
        X.append(r[5] + dum)
    return np.asarray(X, float)


def fit(conn, book, eval_season, lookback=10):
    """Walk-forward fit on [eval_season-lookback, eval_season-1], sealed seasons dropped.
    Returns the artefact dict, or None if insufficient data / sklearn missing."""
    if ElasticNetCV is None:
        print("sklearn missing"); return None
    fit_lo = eval_season - lookback
    train_seasons = [y for y in range(fit_lo, eval_season)
                     if y not in SEALED.get(book, set())]
    recs = build_records(conn, book, train_seasons)
    if len(recs) < 150:
        print(f"{book} s{eval_season}: insufficient train rows n={len(recs)}"); return None
    cats = sorted({r[6] for r in recs})
    X = _design(recs, cats)
    y = np.asarray([r[3] for r in recs], float)
    vb = np.asarray([r[4] for r in recs], float)
    mu = X.mean(0); sd = X.std(0); sd[sd < 1e-9] = 1.0
    Xs = (X - mu) / sd
    m = ElasticNetCV(l1_ratio=[.2, .5, .8], cv=4, max_iter=5000)
    m.fit(Xs, y)
    vhat_oos = cross_val_predict(m, Xs, y, cv=4)
    ws = np.linspace(0, 1, 21)
    wstar = float(ws[np.argmin([np.mean(((w * vhat_oos + (1 - w) * vb) - y) ** 2) for w in ws])])
    s_hat = float(np.std(y - vhat_oos))
    art = {"book": book, "eval_season": eval_season, "lookback": lookback,
           "coef": m.coef_.astype(float), "intercept": float(m.intercept_),
           "mu": mu, "sd": sd, "cats": cats, "wstar": wstar, "s_hat": s_hat,
           "drv": list(DRV[book]), "n_train": len(recs)}
    return art


def artefact_path(book, eval_season):
    return os.path.join(ARTEFACT_DIR, f"overlay_{book}_s{eval_season}.pkl")


def persist(art):
    os.makedirs(ARTEFACT_DIR, exist_ok=True)
    p = artefact_path(art["book"], art["eval_season"])
    tmp = f"{p}.tmp{os.getpid()}"
    with open(tmp, "wb") as fh:
        pickle.dump(art, fh)
    os.replace(tmp, p)
    return p


def load(book, eval_season):
    with open(artefact_path(book, eval_season), "rb") as fh:
        return pickle.load(fh)


def apply(art, drivers, mpg, pos, v_banked):
    """One contender at inference. drivers: {col: raw_value}. Returns (v_star, s_hat).

    v_star = w* * v_hat + (1 - w*) * v_banked, clamped non-negative. Falls back to
    v_banked with zero widening if any driver is missing, so a data gap degrades to
    the incumbent rather than erroring."""
    feat = []
    for t, c, is_rate in art["drv"]:
        v = drivers.get(c)
        if v is None or (not is_rate and (not mpg or mpg <= 0)):
            return float(v_banked), 0.0
        feat.append(float(v) if is_rate else float(v) / mpg)
    dum = [1.0 if pos == c else 0.0 for c in art["cats"][1:]]
    x = (np.asarray(feat + dum, float) - art["mu"]) / art["sd"]
    v_hat = float(x @ art["coef"] + art["intercept"])
    w = art["wstar"]
    v_star = w * v_hat + (1.0 - w) * float(v_banked)
    return max(v_star, 0.0), art["s_hat"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/awards.db")
    ap.add_argument("--seasons", default="2016-2023")
    ap.add_argument("--stat", default="all", choices=["stl", "blk", "ast", "all"])
    ap.add_argument("--fit", action="store_true", help="fit and persist per eval-season")
    ap.add_argument("--lookback", type=int, default=10)
    a = ap.parse_args(argv)
    lo, hi = (a.seasons.split("-") + [a.seasons])[:2]
    seasons = list(range(int(lo), int(hi) + 1))
    books = ["stl", "blk", "ast"] if a.stat == "all" else [a.stat]
    conn = connect(a.db)
    for book in books:
        evals = [y for y in seasons if y not in SEALED.get(book, set())]
        for Y in evals:
            art = fit(conn, book, Y, a.lookback)
            if art is None:
                continue
            msg = (f"{book} s{Y}: w*={art['wstar']:.3f} s_hat={art['s_hat']:.4f} "
                   f"n_train={art['n_train']} coef_nz={int((art['coef'] != 0).sum())}")
            if a.fit:
                persist(art); msg += f" -> {artefact_path(book, Y)}"
            print(msg)
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
