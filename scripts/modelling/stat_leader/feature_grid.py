"""Grid search: lead correlation of every available factor against every target.

For each target (pts, reb, oreb, dreb, ast, stl, blk) it computes r_lead, the
partial correlation of the candidate with the surge controlling for the target's
banked level AND its own recent momentum, over contender-snapshots. It scans
every column in box_asof (level + momentum forms), advanced_asof, asof_ext,
hustle_asof (level + momentum), defend_asof (level + momentum); availability and
minutes are excluded. It keeps rows with abs(r_lead) >= MIN_R, flags
same-measurement candidates, and sorts by abs(r_lead) descending so the list is
inspectable rather than guessed.

A large NEGATIVE r_lead is as usable as a positive one (a driver with negative
sign), which is why the filter is on the absolute value. The flag and the
eventual out-of-sample re-score, not this screen, decide what is real; with ~100
columns against 7 targets some passers will be chance or leakage.

surge = (final_stat - banked_stat)/(final_gp - banked_gp) - banked_stat/banked_gp
oreb/dreb reconstructed from stat_rate_counts.reb and advanced_asof.dreb; their
own-momentum control falls back to total-rebound momentum (no split rolling col).

Run:
  caffeinate -i uv run python3 -m scripts.modelling.stat_leader.feature_grid \
      --seasons 2016-2023
"""
from __future__ import annotations

import argparse
import re
import sys

import numpy as np

try:
    from scripts.common.db import connect
    from scripts.features.stat_leader import nodes as N
except ImportError:  # pragma: no cover
    from db import connect  # type: ignore
    import nodes as N  # type: ignore

MIN_R = 0.10
MIN_N = 200
ROLLING = {"stg_nba_box_asof", "stg_nba_hustle_asof", "stg_nba_defend_asof"}
FLAT = {"stg_nba_player_asof_ext", "stg_nba_player_advanced_asof"}
SKIP_COLS = {"season", "snapshot_date", "nba_api_id", "team_id", "game_id",
             "pulled_at", "gp", "min", "gp_asof", "gp_played_asof", "n_games_asof",
             "plus_minus", "def_ws"}
SKIP_BASE = {"mpg"}  # availability/minutes, excluded per instruction
OWN_MOM = {"pts": "ppg_l10_vs_std", "reb": "rpg_l10_vs_std", "oreb": "rpg_l10_vs_std",
           "dreb": "rpg_l10_vs_std", "ast": "apg_l10_vs_std", "stl": "spg_l10_vs_std",
           "blk": "bpg_l10_vs_std"}
SUSPECT = {
    "pts": ("ppg", "points", "_pts", "pts_", "usg", "efg", "ts_pct", "fg_pct", "fg3_pct", "ft_pct", "pra"),
    "reb": ("rpg", "reb", "oreb", "dreb", "pra", "2nd_chance"),
    "oreb": ("oreb", "reb", "rpg", "pra", "2nd_chance"),
    "dreb": ("dreb", "reb", "rpg", "pra"),
    "ast": ("apg", "ast", "assist", "pra"),
    "stl": ("spg", "stl", "steal", "defl", "_stl"),
    "blk": ("bpg", "_blk", "blk_", "block"),
}


def _pts(d):
    return 2.0 * (d.get("fg2m") or 0.0) + 3.0 * (d.get("fg3m") or 0.0) + (d.get("ftm") or 0.0)


BASE_STAT = {"reb": lambda d: d.get("reb") or 0.0, "pts": _pts,
             "ast": lambda d: d.get("ast") or 0.0, "stl": lambda d: d.get("stl") or 0.0,
             "blk": lambda d: d.get("blk") or 0.0}


def _suspect(target, col):
    c = col.lower()
    return any(tok in c for tok in SUSPECT[target])


def _resid(y, X):
    return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]


def _rlead(u, s, b, m):
    u, s, b, m = (np.asarray(x, float) for x in (u, s, b, m))
    X = np.column_stack([b, m, np.ones(len(u))])
    ru, rs = _resid(u, X), _resid(s, X)
    if ru.std() < 1e-9 or rs.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(ru, rs)[0, 1])


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
        bases = sorted({c[:-4] for c in cols if c.endswith("_std")})
        for b in bases:
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
    return picked


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/awards.db")
    ap.add_argument("--seasons", default="2016-2023")
    a = ap.parse_args(argv)
    lo, hi = (a.seasons.split("-") + [a.seasons])[:2]
    seasons = list(range(int(lo), int(hi) + 1))

    conn = connect(a.db)
    counts, finals, _, _ = N._load(conn, seasons)

    dreb = {}
    for r in conn.execute(
            f'SELECT season, snapshot_date, nba_api_id, dreb FROM stg_nba_player_advanced_asof '
            f'WHERE season IN ({",".join("?"*len(seasons))})', seasons):
        if r["dreb"] is not None:
            dreb[(r["season"], r["snapshot_date"], r["nba_api_id"])] = float(r["dreb"])
    mom = _load_cols(conn, "stg_nba_box_asof", sorted(set(OWN_MOM.values())), seasons)

    last = {}
    for (s, snap, pid) in counts:
        k = (s, pid)
        if k not in last or snap > last[k]:
            last[k] = snap

    def stat_pg(d, target):
        gp = d.get("gp_played_asof") or 0.0
        if gp < 1:
            return None
        if target in BASE_STAT:
            return BASE_STAT[target](d) / gp
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

    targets = ["pts", "reb", "oreb", "dreb", "ast", "stl", "blk"]
    for target in targets:
        rows = {}
        mcol = OWN_MOM[target]
        for (s, snap, pid), d in counts.items():
            gp = d.get("gp_played_asof") or 0.0
            fk = (s, pid)
            if gp < 5 or fk not in last:
                continue
            fsnap = last[fk]
            fd = finals.get(fk)
            fgp = (fd.get("gp_played_asof") or 0.0) if fd else 0.0
            rem = fgp - gp
            if rem < 10:
                continue
            if target in ("oreb", "dreb"):
                dr = dreb.get((s, snap, pid)); drf = dreb.get((s, fsnap, pid))
                if dr is None or drf is None:
                    continue
                reb_bt = d.get("reb") or 0.0
                reb_ft = fd.get("reb") or 0.0
                bt = dr if target == "dreb" else reb_bt - dr
                ft = drf if target == "dreb" else reb_ft - drf
                b = bt / gp
                surge = (ft - bt) / rem - b
            else:
                b = stat_pg(d, target)
                f = stat_pg(fd, target)
                if b is None or f is None:
                    continue
                surge = (f * fgp - b * gp) / rem - b
            mr = mom.get((s, snap, pid))
            mv = mr.get(mcol) if mr else None
            if mv is None:
                continue
            rows[(s, snap, pid)] = (b, surge, float(mv))

        by_snap = {}
        for (s, snap, pid), v in rows.items():
            by_snap.setdefault((s, snap), []).append((pid, v))
        keep = set()
        for key, lst in by_snap.items():
            lst.sort(key=lambda t: t[1][0], reverse=True)
            for pid, _ in lst[:25]:
                keep.add((key[0], key[1], pid))

        results = []
        for t in tables:
            if t not in catalog:
                continue
            cc, data = catalog[t]
            for col in cc:
                U, S, B, M = [], [], [], []
                for k in keep:
                    rec = data.get(k)
                    v = rec.get(col) if rec else None
                    if v is None:
                        continue
                    b, surge, mv = rows[k]
                    U.append(float(v)); S.append(surge); B.append(b); M.append(mv)
                if len(U) < MIN_N:
                    continue
                r = _rlead(U, S, B, M)
                if np.isnan(r) or abs(r) < MIN_R:
                    continue
                results.append((r, col, t.replace("stg_nba_", "").replace("_asof", ""),
                                _suspect(target, col), len(U)))
        results.sort(key=lambda x: -abs(x[0]))
        print("\n" + "=" * 68)
        print(f"TARGET {target}   ({len(results)} factors with |r_lead| >= {MIN_R})")
        print(f"  {'r_lead':>7}  {'susp':>4}  {'factor':<26}{'source':<14}{'n':>6}")
        for r, col, src, susp, n in results:
            print(f"  {r:>+7.2f}  {'*' if susp else '':>4}  {col:<26}{src:<14}{n:>6}")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
