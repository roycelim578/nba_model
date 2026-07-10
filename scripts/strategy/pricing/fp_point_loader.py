"""First-place point loader for the jspeak_floor binding.

Reads the FP arm's OUT-OF-SAMPLE predictions the jspeak experiment already wrote,
out/fp_experiment/oof_<award>_first_place_share.csv, and serves the FP softmax_of_mean
point per (snapshot, player) so the backtest needs no fresh training. The 2024 rows in
those bundles are the walk-forward test fold at K=200 (HELD_OUT={2025}), which is exactly
the out-of-sample first-place point jspeak_floor was validated against.

The CSV carries one raw ensemble-mean score per row (stage1_score) plus the group_key
"<award>|<season>|<snapshot>". The FP point is the grouped softmax of stage1_score within
group_key (softmax_of_mean over the admitted set). Rows key on nba_api_id and the snapshot
parsed out of group_key (its third field, canonical ISO), which avoids the display-munged
snapshot_date column. player_id is mapped to nba_api_id through the players table.

Any name the FP fold did not score returns NaN, which the jspeak lift treats as no-lift.

British English. No inline comments.
"""
from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

log = logging.getLogger("fp_point_loader")

FP_DIR = "models/fp"
_FP_CACHE = {}
_PLAYERS_CACHE = {}


def _players_map(conn):
    key = id(conn)
    if key not in _PLAYERS_CACHE:
        rows = conn.execute(
            "SELECT player_id, nba_api_id FROM players WHERE nba_api_id IS NOT NULL").fetchall()
        _PLAYERS_CACHE[key] = {int(r[0]): int(r[1]) for r in rows}
    return _PLAYERS_CACHE[key]


def _softmax(x):
    x = np.asarray(x, dtype=float)
    e = np.exp(x - x.max())
    return e / e.sum()


def _load(award, season, fp_dir=FP_DIR):
    key = (award, int(season), fp_dir)
    if key in _FP_CACHE:
        return _FP_CACHE[key]
    path = os.path.join(fp_dir, f"oof_{award}_first_place_share.csv")
    if not os.path.exists(path):
        raise SystemExit(
            f"FP OOF bundle not found: {path}. Either run the jspeak experiment's "
            f"diag_fp_vs_voteshare for {award}, or train_fp_fold, before the backtest.")
    df = pd.read_csv(path)
    df = df[df["season"].astype(int) == int(season)].copy()
    if df.empty:
        raise SystemExit(
            f"{path} has no season={season} rows. The 2024 fold must exist in the FP "
            f"bundle (HELD_OUT should be {{2025}}, leaving 2024 a scored fold).")
    if "stage1_score" not in df.columns or "group_key" not in df.columns:
        raise SystemExit(f"{path} missing stage1_score/group_key; unexpected schema.")
    df["snap"] = df["group_key"].str.split("|").str[2]
    df["fp_point"] = df.groupby("group_key")["stage1_score"].transform(
        lambda s: _softmax(s.to_numpy(dtype=float)))
    out = {(int(r.nba_api_id), str(r.snap)): float(r.fp_point) for r in df.itertuples()}
    _FP_CACHE[key] = out
    log.info("[fp] %s %s: loaded %d rows over %d snapshots from %s",
             award, season, len(out), df["snap"].nunique(), os.path.basename(path))
    return out


def fp_vector(conn, award, season, snap, pids, fp_dir=FP_DIR):
    """FP point per candidate aligned to pids (NaN where the FP fold did not score it)."""
    fp = _load(award, season, fp_dir)
    pmap = _players_map(conn)
    snap = str(snap)
    out = np.full(len(pids), np.nan, dtype=float)
    for i, pid in enumerate(pids):
        aid = pmap.get(int(pid))
        if aid is None:
            continue
        v = fp.get((aid, snap))
        if v is not None:
            out[i] = v
    return out


def coverage(conn, award, season, snap, pids, fp_dir=FP_DIR):
    v = fp_vector(conn, award, season, snap, pids, fp_dir)
    return int(np.isfinite(v).sum()), len(pids)
