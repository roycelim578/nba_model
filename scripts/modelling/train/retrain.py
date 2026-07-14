"""Monotone-constraint A/B retrain (MVP, DPOY) + jspeak_floor component persistence (ROTY).

Every fit persists its full booster ensemble in the canonical shape (d["boosters"] with
booster_strings + importances, plus a .pkl.manifest.json sidecar carrying the feature
order, the monotone vector, the label, seeds and K), so any model here reconstructs and
can be promoted straight into the backtest loader without refitting.

Arms per award:
  MVP, DPOY   baseline = kept features, label_vote_share, no constraint
              monotone = kept + carry_years_repeat, label_vote_share, hard non-increasing
                         (advanced) constraint on carry_years_repeat only
              fp       = kept + carry_years_repeat, label_first_place_share, SAME constraint.
                         The deployed quote is jspeak_floor(VS_point, FP_point); if only VS is
                         incumbency-corrected, a fatigued incumbent's high fp_top lifts him back
                         through jspeak_floor and undoes the fix, so fp carries the constraint too.
  ROTY        vs = kept, label_vote_share ; fp = kept, label_first_place_share
              (no years_repeat: fatigue is moot for ROTY. These two are the jspeak_floor
               components the backtest read from OOF CSVs without persisting boosters.)

Walk-forward: earliest voting season is 1996 and the minimum train span is 5 seasons, so
the first scored fold is 2001. Per-fold K = 50, except the 2024 fold at 200 (--k / --k-2024),
matching how the deployed model scores 2024. A deployable FINAL per arm is also fit on all
non-sealed seasons at K=200. Season 2025 is SEALED: never trained on, scored, or persisted.

PRECONDITION: apply_years_repeat_feature.py applied and feature_stats_asof rebuilt (so
carry_years_repeat exists); apply_fp_label.py applied (label_first_place_share served) for
the ROTY fp arm.

RUN from repo root (overnight):
  caffeinate -i uv run python -m scripts.modelling.train.retrain --k 50 --k-2024 200 --no-progress

British English. No inline comments.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scripts.common import config
    from scripts.common.db import connect
    from scripts.features.feature_loader import load_design_matrix
    from scripts.modelling.train.pl_trainer import (
        read_selection, factorise_groups, fit_and_return_boosters,
        TrainConfig, DEFAULT_TREE_PARAMS)
    from scripts.modelling.train.pl_objective import make_pl_objective, grouped_softmax
except ImportError:  # pragma: no cover
    sys.path.insert(0, "src")
    from scripts.common import config  # type: ignore
    from scripts.common.db import connect  # type: ignore
    from scripts.features.feature_loader import load_design_matrix  # type: ignore
    from scripts.modelling.train.pl_trainer import (  # type: ignore
        read_selection, factorise_groups, fit_and_return_boosters,
        TrainConfig, DEFAULT_TREE_PARAMS)
    from scripts.modelling.train.pl_objective import make_pl_objective, grouped_softmax  # type: ignore

log = logging.getLogger("retrain_monotone")

SEALED = {2025}
MIN_TRAIN = 5
CONSTRAINED = "carry_years_repeat"
LABEL_COLS = frozenset({"label_vote_share", "label_won_flag", "label_rank",
                        "label_first_place_share", "label_first_place_votes"})
POS_FORCE = ["pos_is_guard", "pos_is_wing", "pos_is_big"]  # force-included; old selection dropped them when the position map was empty
NUM_BOOST = {"MVP": 250, "DPOY": 250, "ROTY": 100, "6MOTY": 100}
SEED = 17
OUT = Path("models/folds")
N_JOBS = 6

AWARD_ARMS = {
    # MVP/DPOY: baseline vs monotone is the incumbency A/B (label_vote_share). The fp arm
    # (first-place-share) is the OTHER half of the deployed jspeak_floor quote and MUST carry
    # the same years_repeat + monotone constraint, or a fatigued incumbent's high fp_top would
    # lift him back up through jspeak_floor and undo the VS fix at serve time.
    "MVP": [("baseline", "label_vote_share", False), ("monotone", "label_vote_share", True),
            ("fp", "label_first_place_share", True)],
    "DPOY": [("baseline", "label_vote_share", False), ("monotone", "label_vote_share", True),
             ("fp", "label_first_place_share", True)],
    # ROTY: fatigue moot, so vs + fp are both unconstrained; they are the ROTY jspeak_floor
    # components the backtest previously read from OOF CSVs without persisting boosters.
    "ROTY": [("vs", "label_vote_share", False), ("fp", "label_first_place_share", False)],
    "6MOTY": [("vs", "label_vote_share", False), ("fp", "label_first_place_share", False)],
}


def _sealed_for(award):
    """Award-aware sealed-season set for training. The permanent one-shot
    2025 exclusion (SEALED) unioned with the award's held-out registry entry,
    so a new award holds out both its dev and test seasons while MVP/DPOY/ROTY
    keep 2024 in and exclude only 2025 exactly as the bare {2025} did. Reads
    config.SEAL_REGISTRY, defaulting to config._DEFAULT_SEAL for an
    unregistered award."""
    return SEALED | set(config.SEAL_REGISTRY.get(award, config._DEFAULT_SEAL))


def _assert_label_free(feat_cols, label_col, award, arm):
    """Guard against label leakage. The feature matrix must contain no label
    column, and in particular not the arm's own label_col. Raises before any
    fit so a mis-minted selection cannot silently train on the answer."""
    leaked = sorted(set(feat_cols) & (LABEL_COLS | {label_col}))
    if leaked:
        raise SystemExit(
            f"[{award}/{arm}] label leakage: feature set contains label "
            f"column(s) {leaked}; re-mint the selection with these excluded "
            f"before training.")


def _frame(conn, award):
    rows = load_design_matrix(conn, award, seasons=None)
    if not rows:
        raise SystemExit(f"no design-matrix rows for {award}")
    return pd.DataFrame(rows)


def _tree_params(feat_cols, add_yr):
    tp = dict(DEFAULT_TREE_PARAMS)
    if add_yr:
        tp["monotone_constraints"] = [-1 if f == CONSTRAINED else 0 for f in feat_cols]
        tp["monotone_constraints_method"] = "advanced"
    return tp


def _predict(dep, X):
    import lightgbm as lgb
    ens = np.empty((dep["k_bootstrap"], X.shape[0]), dtype=float)
    for i, s in enumerate(dep["booster_strings"]):
        ens[i] = lgb.Booster(model_str=s).predict(X)
    return ens


def _persist(dep, feat_cols, award, arm, label_col, add_yr, train_seasons, tag, sel_id):
    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / f"{award}_{arm}_{tag}"
    with open(f"{stem}.pkl", "wb") as f:
        pickle.dump({"boosters": dep, "award": award, "arm": arm,
                     "label_col": label_col, "tag": tag, "selection_id": sel_id}, f)
    manifest = {
        "award": award, "arm": arm, "label_col": label_col, "tag": tag,
        "features": list(feat_cols),
        "monotone_constraints": dep["tree_params"].get("monotone_constraints"),
        "monotone_constraints_method": dep["tree_params"].get("monotone_constraints_method"),
        "k_bootstrap": dep["k_bootstrap"], "num_boost_round": dep["num_boost_round"],
        "seeds": dep["seeds"], "seed_base": SEED, "hessian_floor": dep["hessian_floor"],
        "train_seasons": [int(s) for s in train_seasons], "selection_id": sel_id,
        "years_repeat_constrained": bool(add_yr),
    }
    with open(f"{stem}.pkl.manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return f"{stem}.pkl"


def _cfg(k, award, tree):
    return TrainConfig(k_bootstrap=k, num_boost_round=NUM_BOOST[award], n_jobs=N_JOBS,
                       tree_params=tree, seed=SEED, objective_factory=make_pl_objective)


def _fit(df, feat_cols, label_col, train_seasons, k, award, tree):
    tr = df[df["season"].isin(train_seasons)]
    if label_col not in tr.columns:
        raise SystemExit(f"{label_col} absent from load_design_matrix output; "
                         f"apply_fp_label.py must be applied for the fp arm.")
    Xtr = tr[feat_cols].to_numpy(float)
    ytr = tr[label_col].to_numpy(float)
    gtr = factorise_groups(tr)
    return fit_and_return_boosters(Xtr, ytr, gtr, _cfg(k, award, tree),
                                   np.random.default_rng(SEED), k=k)


def _latest_selection_id(conn, award):
    """Most recently persisted feature_selection for this award (the fresh re-select
    on the Wave 1 basis). No hardcoded ids, so this auto-tracks a new re-selection."""
    r = conn.execute(
        "SELECT selection_id FROM feature_selection WHERE award=? ORDER BY rowid DESC LIMIT 1",
        (award,)).fetchone()
    if not r:
        raise SystemExit(f"no feature_selection row for {award}; run feature_regularise --persist first")
    return r[0]


def _prepare(conn, award):
    """Load and cache one award's design matrix, selection and season list once,
    so the two-wave scheduler can revisit the award (deployable arms, then baseline)
    without reloading or recomputing the frame."""
    sel_id = _latest_selection_id(conn, award)
    kept = list(read_selection(conn, sel_id)["kept"])
    df = _frame(conn, award)
    present = set(df.columns)
    for c in POS_FORCE:
        if c in present and c not in kept:
            kept.append(c)
    sealed = _sealed_for(award)
    all_seasons = sorted(int(s) for s in df["season"].unique() if s not in sealed)
    leaked = sealed & set(all_seasons)
    assert not leaked, (
        f"seal breach: {award} training seasons contain sealed {sorted(leaked)}")
    missing = [c for c in kept if c not in df.columns]
    if missing:
        live_dl = sum(1 for c in df.columns if c.endswith("_delta_leader"))
        live_rk = sum(1 for c in df.columns if c.endswith("_rank"))
        raise SystemExit(
            f"[{award}] selection {sel_id}: {len(missing)} kept columns are absent from "
            f"the live design matrix (e.g. {missing[:5]}). The live loader emits {live_dl} "
            f"_delta_leader and {live_rk} _rank columns. If those are ~0 the SELECTION is "
            f"stale (pre-Wave 1): re-run  uv run python -m src.data.feature_regularise "
            f"--award {award} --train-seasons {'1997-2023' if award == 'ROTY' else '1996-2023'} "
            f"--persist  then retrain. If they are large the LOADER is pre-Wave 1: re-apply "
            f"apply_wave1_features.py, re-materialise, re-select, retrain.")
    log.info("[%s] prepared: seasons %d..%d, %d features, sel_id=%s", award,
             min(all_seasons), max(all_seasons), len(kept), sel_id)
    return {"award": award, "sel_id": sel_id, "kept": kept, "df": df,
            "all_seasons": all_seasons}


def _run_arm(ctx, arm, label_col, add_yr, since, k, k2024):
    """Fit and persist every walk-forward fold plus the deployable final for one
    arm, returning the per-season final-snapshot frames for the built-in report."""
    award, df, kept = ctx["award"], ctx["df"], ctx["kept"]
    all_seasons, sel_id = ctx["all_seasons"], ctx["sel_id"]
    diag_seasons = [s for s in all_seasons if s >= since]
    feat_cols = kept + [CONSTRAINED] if add_yr else list(kept)
    _assert_label_free(feat_cols, label_col, award, arm)
    tree = _tree_params(feat_cols, add_yr)
    fin = {}
    for T in diag_seasons:
        train_seasons = [s for s in all_seasons if s < T]
        if len(train_seasons) < MIN_TRAIN:
            continue
        kf = k2024 if T == 2024 else k
        dep = _fit(df, feat_cols, label_col, train_seasons, kf, award, tree)
        _persist(dep, feat_cols, award, arm, label_col, add_yr, train_seasons, str(T), sel_id)
        te = df[df["season"] == T]
        ens = _predict(dep, te[feat_cols].to_numpy(float))
        pred = grouped_softmax(ens.mean(axis=0), factorise_groups(te))
        te = te.assign(pred=np.asarray(pred, float))
        last = te["snapshot_date"].max()
        fin[T] = te[te["snapshot_date"] == last].reset_index(drop=True)
        log.info("[%s/%s] fold %d (K=%d) persisted", award, arm, T, kf)
    depF = _fit(df, feat_cols, label_col, all_seasons, k2024, award, tree)
    path = _persist(depF, feat_cols, award, arm, label_col, add_yr, all_seasons, "final", sel_id)
    log.info("[%s/%s] FINAL (K=%d, seasons<=%d) -> %s", award, arm, k2024, max(all_seasons), path)
    return fin


def _rank_of(fin, pid):
    if pid is None or fin.empty or pid not in set(fin["player_id"]):
        return None
    return int((fin["pred"] > float(fin.loc[fin["player_id"] == pid, "pred"].iloc[0])).sum()) + 1


def _report(award, fin_by_arm, seasons, names, winners):
    a0, a1 = AWARD_ARMS[award][0][0], AWARD_ARMS[award][1][0]
    if a0 not in fin_by_arm or a1 not in fin_by_arm:
        log.info("[%s] built-in A/B report skipped (needs arms %s and %s)", award, a0, a1)
        return
    print(f"\n================ {award}: {a0} vs {a1} (true winner rank at final snapshot) ================")
    yhdr = "yrs_rep" if award != "ROTY" else ""
    print(f"{'season':>7}{'winner':<24}{yhdr:>8}{a0[:8]+'_rk':>11}{a1[:8]+'_rk':>11}"
          f"  {a0[:10]+'_top1':<22}{a1[:10]+'_top1':<22}")
    for T in seasons:
        if T not in fin_by_arm[a0] or T not in fin_by_arm[a1]:
            continue
        w = winners.get((award, T))
        wname = names.get(w, str(w))[:22]
        f0, f1 = fin_by_arm[a0][T], fin_by_arm[a1][T]
        yr = ""
        src = f1 if CONSTRAINED in f1.columns else (f0 if CONSTRAINED in f0.columns else None)
        if src is not None and w in set(src["player_id"]):
            yr = f"{float(src.loc[src['player_id'] == w, CONSTRAINED].iloc[0]):.0f}"
        r0, r1 = _rank_of(f0, w), _rank_of(f1, w)
        t0 = names.get(int(f0.loc[f0['pred'].idxmax(), 'player_id']), '?')[:20] if not f0.empty else '?'
        t1 = names.get(int(f1.loc[f1['pred'].idxmax(), 'player_id']), '?')[:20] if not f1.empty else '?'
        print(f"{T:>7}{wname:<24}{yr:>8}{str(r0):>11}{str(r1):>11}  {t0:<22}{t1:<22}")
    print("  (rank 1 = model top pick is the true winner; lower better)")


def _all_winners(conn):
    out = {}
    for aw, s, pid in conn.execute(
            "SELECT award, season, player_id FROM award_voting WHERE won_flag=1").fetchall():
        out[(aw, int(s))] = int(pid)
    return out


def _names(conn):
    for col in ("name", "full_name", "display_name", "player_name"):
        try:
            return {int(r[0]): str(r[1]) for r in
                    conn.execute(f"SELECT player_id, {col} FROM players").fetchall()}
        except Exception:
            continue
    return {}


def main(argv=None):
    global N_JOBS
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/awards.db")
    ap.add_argument("--awards", nargs="+", default=["MVP", "DPOY", "ROTY"])
    ap.add_argument("--since", type=int, default=2001)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--k-2024", type=int, default=200, dest="k2024")
    ap.add_argument("--n-jobs", type=int, default=6)
    ap.add_argument("--no-progress", action="store_true")
    ap.add_argument("--defer-baseline", dest="defer_baseline", action="store_true", default=True,
                    help="run deployable arms (monotone/fp/vs) across all awards first and "
                         "baseline arms last, so diagnostics can start after WAVE A. Default on.")
    ap.add_argument("--no-defer-baseline", dest="defer_baseline", action="store_false")
    args = ap.parse_args(argv)
    if args.no_progress:
        os.environ["TQDM_DISABLE"] = "1"
    N_JOBS = args.n_jobs
    conn = connect(args.db)
    names = _names(conn)
    winners = _all_winners(conn)

    contexts = {aw: _prepare(conn, aw) for aw in args.awards}

    def _prio(arm):
        return 1 if (args.defer_baseline and arm == "baseline") else 0
    jobs = []
    for aw in args.awards:
        for (arm, label_col, add_yr) in AWARD_ARMS[aw]:
            jobs.append((_prio(arm), aw, arm, label_col, add_yr))
    jobs.sort(key=lambda j: j[0])

    fin_by_award = {aw: {} for aw in args.awards}
    wave_b = False
    for p, aw, arm, label_col, add_yr in jobs:
        if p == 1 and not wave_b:
            log.info("================ WAVE A COMPLETE: deployable arms persisted "
                     "(monotone/fp/vs). Run diag_object_suite now for the early combined-impact "
                     "read. Starting baseline arms (the isolated A/B). ================")
            wave_b = True
        fin_by_award[aw][arm] = _run_arm(contexts[aw], arm, label_col, add_yr,
                                         args.since, args.k, args.k2024)

    for aw in args.awards:
        seasons = [s for s in contexts[aw]["all_seasons"] if s >= args.since]
        _report(aw, fin_by_award[aw], seasons, names, winners)
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
