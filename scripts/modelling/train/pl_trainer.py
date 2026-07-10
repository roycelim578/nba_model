"""LightGBM Plackett-Luce trainer scaffold (Phase 2 close / Phase 3 model).

Owns everything AROUND the custom objective: read a named feature selection,
assemble the design matrix, build walk-forward folds grouped by season, factorise
group_key -> integer group_ids in row order, run a grouped bootstrap ensemble,
and score (Brier on P(win), vote-share RMSE, CI coverage, top-k) against
benchmarks. The custom Plackett-Luce objective is INJECTED via make_pl_objective
(parallel chat owns it; until it lands we inject a stub so the whole scaffold is
testable end-to-end).

INVARIANTS (inherited, do not break):
  - Season = STARTING year. group_key = "award|season|snapshot".
  - Walk-forward CV grouped by season: train <= T, predict T+1, rolling. A season
    is NEVER split across train/test.
  - 2024 and 2025 are held out ENTIRELY (never in any train or validation fold);
    reserved for the final trading-strategy test.
  - The admitted set is the softmax group; the label is zero-padded, within-group
    normalised vote_share, never renormalised over vote-getters only.
  - The trainer reads a NAMED selection_id; it never re-runs feature selection.

OBJECTIVE SEAM (frozen, see HANDOFF_pl_objective_parallel_chat.md):
  make_pl_objective(group_ids: np.ndarray) -> fobj(predt, dtrain) -> (grad, hess)
  group_ids is integer-coded, row-aligned to the training matrix, grouped by
  equality (not assumed sorted/contiguous). The trainer builds group_ids by
  factorising the fold's group_key column in row order and passes the closure as
  fobj to lgb.train.

Run from project root (once the real objective + lightgbm are in place):
  uv run python -m scripts.modelling.train.pl_trainer --award MVP --selection-id <id> [--k 200]
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scripts.common.db import connect
    from scripts.features.feature_loader import load_design_matrix
    from scripts.modelling.train.pl_objective import make_pl_objective, make_pl_eval, grouped_softmax
    import json as _json
except ImportError:  # pragma: no cover
    from db import connect  # type: ignore
    from feature_loader import load_design_matrix  # type: ignore
    from pl_objective import make_pl_objective, make_pl_eval, grouped_softmax  # type: ignore
    import json as _json

log = logging.getLogger("pl_trainer")

# Held out entirely for the final trading-strategy test. Never in any fold.
HELD_OUT_SEASONS = frozenset({2025})  # 2024 burned to dev; 2025 the sole sealed one-shot

# Conservative tree settings (PHASE2_CONTRACT). Leaf-wise overfit mitigation on a
# ~28-season dataset. Tunable later via walk-forward CV; these are the defaults.
DEFAULT_TREE_PARAMS = {
    "num_leaves": 31,
    "min_data_in_leaf": 20,
    "max_depth": 8,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,   # decorrelate trees across the bootstrap ensemble
    "bagging_fraction": 1.0,   # grouped bootstrap handled OUTSIDE lgb (whole groups)
    "num_threads": 1,          # ONE thread per booster: the process pool owns the cores
    "verbosity": -1,
}


# ---------------------------------------------------------------------------
# Selection reader
# ---------------------------------------------------------------------------

def read_selection(conn, selection_id: str) -> dict:
    """Read a persisted feature selection. Returns award, method, kept feature list."""
    row = conn.execute(
        "SELECT award, method, n_kept, kept_features_json, params_json "
        "FROM feature_selection WHERE selection_id = ?", (selection_id,),
    ).fetchone()
    if row is None:
        raise SystemExit(f"no feature_selection row for selection_id={selection_id!r}")
    return {
        "selection_id": selection_id,
        "award": row["award"],
        "method": row["method"],
        "kept": _json.loads(row["kept_features_json"]),
        "params": _json.loads(row["params_json"]),
    }


# ---------------------------------------------------------------------------
# Design-matrix assembly
# ---------------------------------------------------------------------------

# Non-feature columns the matrix carries alongside features (keys/label/group).
_KEEP_META = ["player_id", "nba_api_id", "season", "snapshot_date", "award",
              "group_key", "week_index", "label_vote_share", "label_won_flag",
              "label_first_place_share", "label_first_place_votes"]


def assemble_matrix(conn, award, kept_features, seasons=None,
                    model_version=None, pwin_key=None, placebo_narrative=False):
    rows = load_design_matrix(conn, award, seasons=seasons,
                              model_version=model_version, pwin_key=pwin_key,
                              placebo_narrative=placebo_narrative)
    if not rows:
        raise SystemExit(f"no design-matrix rows for award={award}")
    df = pd.DataFrame(rows)
    missing = [c for c in kept_features if c not in df.columns]
    if missing:
        raise SystemExit(f"selection references {len(missing)} columns not in the "
                         f"design matrix (loader/selection drift): {missing[:10]}")
    meta = [c for c in _KEEP_META if c in df.columns]
    cols = meta + [c for c in kept_features if c not in meta]
    return df[cols].copy()


# ---------------------------------------------------------------------------
# Group-id factorisation (the seam input)
# ---------------------------------------------------------------------------

def factorise_groups(df: pd.DataFrame) -> np.ndarray:
    """Integer-code group_key in ROW ORDER. Rows sharing a group_key get the same
    int; ids need not be contiguous or sorted (the objective groups by equality).
    Returns an int array length len(df), aligned to df's row order."""
    codes, _ = pd.factorize(df["group_key"], sort=False)
    return codes.astype(np.int64)


# ---------------------------------------------------------------------------
# Walk-forward folds (grouped by season; held-out seasons excluded)
# ---------------------------------------------------------------------------

@dataclass
class Fold:
    test_season: int
    train_seasons: list[int]
    train_idx: np.ndarray  # row positions into the assembled df
    test_idx: np.ndarray


def build_folds(df: pd.DataFrame, min_train_seasons: int = 5) -> list[Fold]:
    """Walk-forward: for each eligible test season T (ascending), train on ALL
    eligible seasons < T. Eligible = present in df AND not held out. A season is
    never split. Requires >= min_train_seasons of history before the first test
    fold (early folds with too little history are skipped, not run thin)."""
    seasons = sorted(s for s in df["season"].unique() if s not in HELD_OUT_SEASONS)
    folds: list[Fold] = []
    for i, T in enumerate(seasons):
        train_seasons = seasons[:i]  # strictly earlier eligible seasons
        if len(train_seasons) < min_train_seasons:
            continue
        train_mask = df["season"].isin(train_seasons).to_numpy()
        test_mask = (df["season"] == T).to_numpy()
        folds.append(Fold(
            test_season=int(T),
            train_seasons=[int(s) for s in train_seasons],
            train_idx=np.flatnonzero(train_mask),
            test_idx=np.flatnonzero(test_mask),
        ))
    return folds


# ---------------------------------------------------------------------------
# Grouped bootstrap (resample WHOLE groups, not rows)
# ---------------------------------------------------------------------------

def grouped_bootstrap_indices(group_ids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One bootstrap resample at the GROUP level: sample groups with replacement,
    return the concatenated row indices of the chosen groups (preserving PL group
    structure within each chosen group). Row indices are positions into the array
    that group_ids indexes."""
    uniq = np.unique(group_ids)
    # map each group id -> its member row positions, once
    members: dict[int, np.ndarray] = {g: np.flatnonzero(group_ids == g) for g in uniq}
    chosen = rng.choice(uniq, size=len(uniq), replace=True)
    return np.concatenate([members[g] for g in chosen])


# ---------------------------------------------------------------------------
# Objective injection (stub now; real make_pl_objective lands from parallel chat)
# ---------------------------------------------------------------------------

def stub_make_pl_objective(group_ids: np.ndarray):
    """STUB standing in for the parallel chat's make_pl_objective. Implements the
    SAME grouped-softmax cross-entropy math so the scaffold is exercised honestly
    end-to-end, but is NOT the validated artefact. Swap for the real import when
    it lands. Signature is the frozen seam: returns fobj(predt, dtrain)->(g,h)."""
    def fobj(predt: np.ndarray, dtrain):
        y = dtrain.get_label()
        grad = np.empty_like(predt, dtype=np.float64)
        hess = np.empty_like(predt, dtype=np.float64)
        for g in np.unique(group_ids):
            m = np.flatnonzero(group_ids == g)
            s = predt[m]
            s = s - s.max()  # stability
            e = np.exp(s)
            p = e / e.sum()
            grad[m] = p - y[m]
            hess[m] = p * (1.0 - p)
        return grad, hess
    return fobj


# ---------------------------------------------------------------------------
# Metrics (computed on a fold's test predictions)
# ---------------------------------------------------------------------------

def _group_softmax(scores: np.ndarray, group_ids: np.ndarray) -> np.ndarray:
    out = np.empty_like(scores, dtype=np.float64)
    for g in np.unique(group_ids):
        m = np.flatnonzero(group_ids == g)
        s = scores[m] - scores[m].max()
        e = np.exp(s)
        out[m] = e / e.sum()
    return out


def pwin_from_ensemble(ensemble_scores: np.ndarray, group_ids: np.ndarray) -> np.ndarray:
    """ensemble_scores: (K, n_rows) raw scores from K bootstrap models. P(win) per
    row = fraction of replicates in which the row is its group's argmax share."""
    K = ensemble_scores.shape[0]
    wins = np.zeros(ensemble_scores.shape[1], dtype=np.float64)
    for k in range(K):
        p = _group_softmax(ensemble_scores[k], group_ids)
        for g in np.unique(group_ids):
            m = np.flatnonzero(group_ids == g)
            wins[m[np.argmax(p[m])]] += 1.0
    return wins / K


def brier_pwin(pwin: np.ndarray, won_flag: np.ndarray) -> float:
    return float(np.mean((pwin - won_flag) ** 2))


def vote_share_rmse(pred_share: np.ndarray, true_share: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred_share - true_share) ** 2)))


def uniform_benchmark_brier(group_ids: np.ndarray, won_flag: np.ndarray) -> float:
    """Uniform-over-candidates P(win) = 1/group_size; the sanity floor."""
    pwin = np.empty(len(group_ids), dtype=np.float64)
    for g in np.unique(group_ids):
        m = np.flatnonzero(group_ids == g)
        pwin[m] = 1.0 / len(m)
    return brier_pwin(pwin, won_flag)


def winner_topk_accuracy(pwin: np.ndarray, won_flag: np.ndarray,
                         group_ids: np.ndarray, k: int = 1) -> float:
    """Fraction of groups whose TRUE winner is in the model's top-k by P(win).
    This is the metric with real signal at large group sizes: Brier-on-P(win) is
    swamped by the ~25 true-zeros per group, so a model that merely predicts
    'everyone near 0' scores well on Brier but badly here. Groups with no flagged
    winner (held-out label edge cases) are skipped."""
    hits = tot = 0
    for g in np.unique(group_ids):
        m = np.flatnonzero(group_ids == g)
        if won_flag[m].sum() == 0:
            continue  # no winner label in this group; not scoreable
        tot += 1
        true_winner_local = int(np.argmax(won_flag[m]))
        topk_local = set(np.argsort(pwin[m])[::-1][:k].tolist())
        if true_winner_local in topk_local:
            hits += 1
    return hits / tot if tot else float("nan")


def _bucket_week(wi: float) -> str:
    """Snapshot-position bucket for in-season tracking (does the model get the
    race right early, or only converge late?)."""
    if wi is None or (isinstance(wi, float) and np.isnan(wi)):
        return "unknown"
    if wi < 0:
        return "preseason"
    if wi <= 6:
        return "early"
    if wi <= 15:
        return "mid"
    return "late"


def accuracy_by_position(pwin, won_flag, group_ids, week_by_group: dict, k: int) -> dict:
    """top-k accuracy split by snapshot-position bucket. week_by_group maps a
    group's integer id -> its week_index. Answers: does the model identify the
    eventual winner from early snapshots, or only late?"""
    buckets: dict[str, list[int]] = {}
    for g in np.unique(group_ids):
        m = np.flatnonzero(group_ids == g)
        if won_flag[m].sum() == 0:
            continue
        b = _bucket_week(week_by_group.get(int(g)))
        true_w = int(np.argmax(won_flag[m]))
        topk = set(np.argsort(pwin[m])[::-1][:k].tolist())
        buckets.setdefault(b, []).append(1 if true_w in topk else 0)
    return {b: round(float(np.mean(v)), 3) for b, v in buckets.items()}


@dataclass
class FoldResult:
    test_season: int
    n_test_groups: int
    n_test_rows: int
    brier_pwin: float
    brier_uniform: float
    vote_share_rmse: float
    top1_acc: float
    top3_acc: float


# ---------------------------------------------------------------------------
# Orchestration (kept thin; real lgb.train wired when objective + lightgbm land)
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    k_bootstrap: int = 200          # dev default; 2000 for the final reported run
    num_boost_round: int = 250      # ~99% of grouped-CE gain on the large-data folds
    tree_params: dict = field(default_factory=lambda: dict(DEFAULT_TREE_PARAMS))
    seed: int = 17
    n_jobs: int = 8                 # process pool over the K bootstrap fits
    objective_factory: callable = make_pl_objective  # validated parallel-chat artefact
    model_version: str | None = None   # NLI model_version; set => merge narrative + bridge cols
    pwin_key: str | None = None         # divergence model-favourite source (None => stat proxy)
    placebo_narrative: bool = False     # scramble narrative cols (feature-count null test)
    label_col: str = "label_vote_share"  # PL target; set label_first_place_share for the first-place arm


def _bootstrap_fit_predict(args) -> tuple[np.ndarray, np.ndarray]:
    """One worker job: resample whole groups, fit a booster, predict test scores.
    Top-level (picklable) for ProcessPoolExecutor. Each worker fits single-threaded
    (num_threads=1 in tree_params) so the process pool, not lgb, owns the cores.
    Returns (test_scores, gain_importances) for this replicate."""
    Xtr, ytr, gtr, Xte, tree_params, num_boost_round, seed = args
    rng = np.random.default_rng(seed)
    bidx = grouped_bootstrap_indices(gtr, rng)
    b_gids = gtr[bidx]
    import lightgbm as lgb
    fobj = make_pl_objective(b_gids)
    params = dict(tree_params)
    params["objective"] = fobj
    params["seed"] = seed
    dtrain = lgb.Dataset(Xtr[bidx], label=ytr[bidx], free_raw_data=False)
    booster = lgb.train(params, dtrain, num_boost_round=num_boost_round)
    imp = booster.feature_importance(importance_type="gain")
    return booster.predict(Xte), imp


def _fit_ensemble_parallel(Xtr, ytr, gtr, Xte, cfg, rng,
                           progress_desc: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Fit the K-booster bootstrap ensemble across n_jobs processes; return
    ((K, n_test) raw scores, (K, n_features) gain importances). Serial fallback
    when n_jobs == 1.

    Progress: a tqdm bar advances per COMPLETED bootstrap fit (via as_completed),
    so the bar moves continuously within a fold rather than only at fold ends.
    Each future carries its replicate index k, so out-of-order completion still
    scatters results to the correct row (identical numbers to ex.map). progress_desc
    labels the bar. No-ops if tqdm absent."""
    try:
        from tqdm.auto import tqdm
    except ImportError:  # pragma: no cover
        def tqdm(it=None, **_):
            return it if it is not None else []
    seeds = [cfg.seed + k for k in range(cfg.k_bootstrap)]
    jobs = [(Xtr, ytr, gtr, Xte, cfg.tree_params, cfg.num_boost_round, s) for s in seeds]
    ens = np.empty((cfg.k_bootstrap, Xte.shape[0]), dtype=np.float64)
    imps = np.empty((cfg.k_bootstrap, Xtr.shape[1]), dtype=np.float64)
    if cfg.n_jobs == 1:
        for k in tqdm(range(cfg.k_bootstrap), desc=progress_desc,
                      total=cfg.k_bootstrap, leave=False, unit="fit"):
            ens[k], imps[k] = _bootstrap_fit_predict(jobs[k])
        return ens, imps
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=cfg.n_jobs) as ex:
        futs = {ex.submit(_bootstrap_fit_predict, jobs[k]): k
                for k in range(cfg.k_bootstrap)}
        for fut in tqdm(as_completed(futs), total=cfg.k_bootstrap,
                        desc=progress_desc, leave=False, unit="fit"):
            k = futs[fut]
            scores, imp = fut.result()
            ens[k] = scores
            imps[k] = imp
    return ens, imps


def _bootstrap_fit_return_booster(args) -> tuple[int, str, np.ndarray]:
    """One worker job for the DEPLOYABLE ensemble: resample whole groups, fit a
    booster with the custom PL objective, and RETURN the serialised booster
    (model_to_string) rather than discarding it after predicting. Mirrors
    _bootstrap_fit_predict exactly (same grouped bootstrap, same seed handling,
    same objective wiring) so the persisted ensemble is identical in construction
    to the scored one; the only difference is we keep the model.

    Returns (k, booster_string, gain_importances). Booster is serialised to a
    string so it pickles cleanly across the process-pool boundary and across
    LightGBM versions (raw Booster objects are fragile to pickle)."""
    k, Xtr, ytr, gtr, tree_params, num_boost_round, seed = args
    rng = np.random.default_rng(seed)
    bidx = grouped_bootstrap_indices(gtr, rng)
    b_gids = gtr[bidx]
    import lightgbm as lgb
    fobj = make_pl_objective(b_gids)
    params = dict(tree_params)
    params["objective"] = fobj
    params["seed"] = seed
    dtrain = lgb.Dataset(Xtr[bidx], label=ytr[bidx], free_raw_data=False)
    booster = lgb.train(params, dtrain, num_boost_round=num_boost_round)
    imp = booster.feature_importance(importance_type="gain")
    return k, booster.model_to_string(), imp


def fit_and_return_boosters(X, y, group_ids, cfg, rng, k: int | None = None):
    """Fit the K-booster grouped-bootstrap ensemble on the FULL provided frame
    (no walk-forward split; the caller is responsible for excluding held-out
    seasons) and RETURN the persisted boosters plus everything needed to predict
    and to interpret sigma later.

    Construction is identical to _fit_ensemble_parallel: seeds are cfg.seed + k
    (prefix-stable, so a K'-prefix reproduces a standalone K'-run), whole-group
    bootstrap, custom PL objective in params['objective'], one thread per booster
    with the process pool owning the cores.

    Returns a dict deployable artefact:
      booster_strings : list[str] length K, lgb model_to_string() per replicate
      importances     : (K, n_features) gain importances
      seeds           : list[int] the per-replicate seeds (reproducibility)
      k_bootstrap     : K
      num_boost_round : rounds used
      tree_params     : the tree params (excluding the objective closure, which
                        is not serialisable and not needed at predict time; a
                        custom-objective booster predicts RAW MARGINS, which the
                        caller softmaxes within group)
      hessian_floor   : recorded for sigma interpretation (read off the objective
                        module default; the booster itself does not need it to
                        predict, but sigma provenance does)
    Predict, later: for each booster_string,
        lgb.Booster(model_str=s).predict(X_new) -> raw scores (K, n_rows);
        grouped_softmax over the mean is the point estimate; the K-spread is sigma.
    """
    try:
        from tqdm.auto import tqdm
    except ImportError:  # pragma: no cover
        def tqdm(it=None, **_):
            return it if it is not None else []
    K = int(k if k is not None else cfg.k_bootstrap)
    seeds = [cfg.seed + i for i in range(K)]
    jobs = [(i, X, y, group_ids, cfg.tree_params, cfg.num_boost_round, seeds[i])
            for i in range(K)]
    booster_strings: list[str | None] = [None] * K
    imps = np.empty((K, X.shape[1]), dtype=np.float64)

    if cfg.n_jobs == 1:
        for i in tqdm(range(K), desc="FINAL fit", total=K, leave=False,
                      unit="fit"):
            ki, bs, imp = _bootstrap_fit_return_booster(jobs[i])
            booster_strings[ki] = bs
            imps[ki] = imp
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=cfg.n_jobs) as ex:
            futs = {ex.submit(_bootstrap_fit_return_booster, jobs[i]): i
                    for i in range(K)}
            for fut in tqdm(as_completed(futs), total=K, desc="FINAL fit",
                            leave=False, unit="fit"):
                ki, bs, imp = fut.result()
                booster_strings[ki] = bs
                imps[ki] = imp

    # hessian floor provenance (default from the objective module; not required
    # to predict, but recorded so sigma is interpretable downstream)
    try:
        from scripts.modelling.train.pl_objective import DEFAULT_HESSIAN_FLOOR as _hfloor  # type: ignore
    except Exception:
        try:
            from pl_objective import DEFAULT_HESSIAN_FLOOR as _hfloor  # type: ignore
        except Exception:
            _hfloor = 1e-3

    return {
        "booster_strings": booster_strings,
        "importances": imps,
        "seeds": seeds,
        "k_bootstrap": K,
        "num_boost_round": cfg.num_boost_round,
        "tree_params": dict(cfg.tree_params),
        "hessian_floor": _hfloor,
    }


def _label_sum_diagnostic(df: pd.DataFrame, group_ids: np.ndarray,
                          label_col: str = "label_vote_share") -> dict:
    """One-time sanity: per admitted group, the label should sum to ~1 (every
    consequential vote-getter is admitted; only stray low-share votes may fall
    outside, summing slightly below 1). Flags any group summing pathologically
    low (e.g. a filter/join problem), which would miscalibrate the softmax target.
    Descriptive, non-gating."""
    y = df[label_col].to_numpy(dtype=float)
    sums = []
    for g in np.unique(group_ids):
        sums.append(float(y[group_ids == g].sum()))
    sums = np.array(sums)
    return {
        "n_groups": int(len(sums)),
        "label_sum_mean": round(float(sums.mean()), 4),
        "label_sum_min": round(float(sums.min()), 4),
        "label_sum_p05": round(float(np.percentile(sums, 5)), 4),
        "n_groups_sum_below_0.5": int((sums < 0.5).sum()),
        "n_groups_sum_zero": int((sums == 0.0).sum()),
    }


def _fit_one(X: np.ndarray, y: np.ndarray, group_ids: np.ndarray,
             cfg: TrainConfig, seed: int):
    """Fit one LightGBM booster with the custom PL objective. lightgbm>=4: the
    objective goes in params['objective'] (the pre-4.0 fobj= kwarg is gone)."""
    import lightgbm as lgb
    fobj = cfg.objective_factory(group_ids)
    params = dict(cfg.tree_params)
    params["objective"] = fobj          # 4.x form, confirmed against 4.6
    params["seed"] = seed
    dtrain = lgb.Dataset(X, label=y, free_raw_data=False)
    booster = lgb.train(params, dtrain, num_boost_round=cfg.num_boost_round)
    return booster


def run_award(conn, selection_id: str, cfg: TrainConfig | None = None) -> dict:
    """End-to-end walk-forward run for one award's named selection.

    Per fold: K grouped-bootstrap resamples of the TRAIN rows, fit a booster on
    each with the custom PL objective, predict the test fold's raw scores, collect
    the K-score ensemble, derive P(win) via grouped Monte Carlo, and score Brier /
    RMSE against the uniform benchmark. 2024/2025 are never in any fold."""
    cfg = cfg or TrainConfig()
    sel = read_selection(conn, selection_id)
    df = assemble_matrix(conn, sel["award"], sel["kept"],
                         model_version=cfg.model_version, pwin_key=cfg.pwin_key,
                         placebo_narrative=cfg.placebo_narrative)
    feat_cols = [c for c in sel["kept"]]
    # Match the two label arms on an identical universe: drop seasons with no
    # first-place data (label_first_place_share NULL for the whole season), so
    # the vote-share baseline and the first-place arm see the same rows and
    # only ytr differs. Never drops within a covered season (zero-vote players
    # get share 0.0, not NULL).
    if "label_first_place_share" in df.columns:
        _null_fp = df["label_first_place_share"].isna().to_numpy()
        if _null_fp.any():
            _dropped = sorted(df.loc[_null_fp, "season"].unique().tolist())
            log.info("first-place coverage: dropping %d rows across %d season(s) %s "
                     "(no first-place-vote data)", int(_null_fp.sum()),
                     len(_dropped), _dropped)
            df = df.loc[~_null_fp].reset_index(drop=True)
    folds = build_folds(df)
    all_gids = factorise_groups(df)
    diag = _label_sum_diagnostic(df, all_gids, cfg.label_col)
    log.info("award=%s selection=%s features=%d rows=%d folds=%d label_sum_mean=%.3f "
             "(min=%.3f, groups<0.5=%d)", sel["award"], selection_id, len(feat_cols),
             len(df), len(folds), diag["label_sum_mean"], diag["label_sum_min"],
             diag["n_groups_sum_below_0.5"])

    rng = np.random.default_rng(cfg.seed)
    fold_results: list[FoldResult] = []
    season_verdicts: list[dict] = []          # one clean verdict per season (final snapshot)
    autopsies: list[dict] = []                 # decisive-failure (winner not in top-3) detail
    pos_accum: dict[str, list[float]] = {}     # in-season top1 by position bucket, across folds
    pos_accum3: dict[str, list[float]] = {}
    imp_accum = np.zeros(len(feat_cols), dtype=np.float64)  # gain importance, summed over folds
    imp_folds = 0
    for f in folds:
        tr = df.iloc[f.train_idx]
        te = df.iloc[f.test_idx].reset_index(drop=True)
        Xtr = tr[feat_cols].to_numpy(dtype=float)
        ytr = tr[cfg.label_col].to_numpy(dtype=float)
        gtr = factorise_groups(tr)                 # fresh per fold, train row order
        Xte = te[feat_cols].to_numpy(dtype=float)
        gte = factorise_groups(te)                 # fresh per fold, test row order
        won_te = te["label_won_flag"].to_numpy(dtype=float)
        true_share_te = te[cfg.label_col].to_numpy(dtype=float)
        # map each test group id -> its week_index (for the in-season breakdown)
        week_by_group = {int(g): float(te.loc[np.flatnonzero(gte == g)[0], "week_index"])
                         for g in np.unique(gte)}

        ens, imps = _fit_ensemble_parallel(
            Xtr, ytr, gtr, Xte, cfg, rng,
            progress_desc=f"{sel['award']} fold {f.test_season}")
        imp_accum += imps.mean(axis=0)
        imp_folds += 1

        pwin = pwin_from_ensemble(ens, gte)
        pred_share = grouped_softmax(ens.mean(axis=0), gte)
        fr = FoldResult(
            test_season=f.test_season,
            n_test_groups=int(len(np.unique(gte))),
            n_test_rows=int(len(te)),
            brier_pwin=brier_pwin(pwin, won_te),
            brier_uniform=uniform_benchmark_brier(gte, won_te),
            vote_share_rmse=vote_share_rmse(pred_share, true_share_te),
            top1_acc=winner_topk_accuracy(pwin, won_te, gte, k=1),
            top3_acc=winner_topk_accuracy(pwin, won_te, gte, k=3),
        )
        fold_results.append(fr)

        # in-season tracking: top1/top3 by snapshot position, accumulated across folds
        by1 = accuracy_by_position(pwin, won_te, gte, week_by_group, k=1)
        by3 = accuracy_by_position(pwin, won_te, gte, week_by_group, k=3)
        for b, v in by1.items():
            pos_accum.setdefault(b, []).append(v)
        for b, v in by3.items():
            pos_accum3.setdefault(b, []).append(v)

        # season-level verdict: at the FINAL (max week_index) scoreable snapshot,
        # did the model's favourite win, and was the winner in its top-3? This is
        # the clean one-verdict-per-season unit (the per-snapshot avg is bimodal
        # because ~26 near-identical snapshots in a season share the same answer).
        final_g = max(week_by_group, key=week_by_group.get)
        m = np.flatnonzero(gte == final_g)
        if won_te[m].sum() > 0:
            true_w = int(np.argmax(won_te[m]))
            order = np.argsort(pwin[m])[::-1]
            winner_in_top3 = bool(true_w in set(order[:3].tolist()))
            season_verdicts.append({
                "season": f.test_season,
                "won_outright": bool(order[0] == true_w),
                "winner_in_top3": winner_in_top3,
                "model_favourite_pwin": round(float(pwin[m][order[0]]), 3),
            })

            # AUTOPSY: when the winner is NOT in the model's top-3, dump the
            # winner's within-group stat percentiles vs the candidates the model
            # preferred. Answers (a) narrative-override [winner's stats genuinely
            # lower -> model correct, not a bug] vs (b) model/feature problem
            # [winner's stats comparable/higher yet ranked low -> investigate].
            if not winner_in_top3:
                te_local = te.iloc[m].reset_index(drop=True)
                # interpretable percentile columns present in the matrix
                pct_cols = [c for c in te_local.columns if c.endswith("_pct")
                            and c in feat_cols]
                # keep a readable subset: the headline production/efficiency/team axes
                headline = [c for c in pct_cols if any(k in c for k in (
                    "pra_std", "ppg_std", "ts_pct_std", "usg", "pie",
                    "stl", "blk", "dreb", "def_rating", "win_pct", "net_rating"))]
                def _row_stats(local_i):
                    r = te_local.iloc[local_i]
                    nm = r.get("nba_api_id", "?")
                    return {"nba_api_id": int(nm) if pd.notna(nm) else None,
                            "model_pwin": round(float(pwin[m][local_i]), 3),
                            "true_vote_share": round(float(true_share_te[m][local_i]), 3),
                            **{c.replace("box_", "").replace("adv_", "").replace("team_", "")
                               : round(float(r[c]), 2) for c in headline if pd.notna(r[c])}}
                autopsies.append({
                    "award": sel["award"], "season": f.test_season,
                    "winner_model_rank": int(np.where(order == true_w)[0][0]) + 1,
                    "winner": _row_stats(true_w),
                    "model_top3": [_row_stats(int(order[j])) for j in range(min(3, len(order)))],
                })

        log.info("  fold %d: top1=%.3f top3=%.3f rmse=%.4f | by-position top1 %s",
                 f.test_season, fr.top1_acc, fr.top3_acc, fr.vote_share_rmse, by1)

    brs = np.array([r.brier_pwin for r in fold_results])
    rms = np.array([r.vote_share_rmse for r in fold_results])
    # season-level tallies (the honest unit)
    won_outright = sum(1 for v in season_verdicts if v["won_outright"])
    in_top3 = sum(1 for v in season_verdicts if v["winner_in_top3"])
    n_seasons = len(season_verdicts)
    # in-season tracking means by bucket
    pos_top1 = {b: round(float(np.mean(v)), 3) for b, v in sorted(pos_accum.items())}
    pos_top3 = {b: round(float(np.mean(v)), 3) for b, v in sorted(pos_accum3.items())}
    # top feature importances (gain), averaged across folds, named
    imp_mean = imp_accum / max(imp_folds, 1)
    order = np.argsort(imp_mean)[::-1]
    top_features = [{"feature": feat_cols[i], "gain": round(float(imp_mean[i]), 1)}
                    for i in order[:25]]

    log.info("SEASON-LEVEL: won_outright=%d/%d (%.1f%%)  winner_in_top3=%d/%d (%.1f%%)",
             won_outright, n_seasons, 100*won_outright/max(n_seasons,1),
             in_top3, n_seasons, 100*in_top3/max(n_seasons,1))
    log.info("IN-SEASON top1 by position: %s", pos_top1)
    log.info("TOP FEATURES (gain): %s", [t["feature"] for t in top_features[:12]])

    return {
        "award": sel["award"], "selection_id": selection_id,
        "n_features": len(feat_cols), "n_rows": len(df), "n_folds": len(folds),
        "k_bootstrap": cfg.k_bootstrap,
        "label_diagnostic": diag,
        # SEASON-LEVEL is the honest unit (one verdict per season, final snapshot)
        "seasons_evaluated": n_seasons,
        "won_outright": won_outright,
        "won_outright_pct": round(100*won_outright/max(n_seasons,1), 1),
        "winner_in_top3": in_top3,
        "winner_in_top3_pct": round(100*in_top3/max(n_seasons,1), 1),
        # IN-SEASON tracking: does the model get the race right early or only late?
        "in_season_top1_by_position": pos_top1,
        "in_season_top3_by_position": pos_top3,
        # TREE decision-making: top gain-importance features across the ensemble
        "top_features_by_gain": top_features,
        "decisive_failures": autopsies,
        # secondary: snapshot-averaged metrics (noisy/bimodal, kept for reference)
        "mean_vote_share_rmse": round(float(rms.mean()), 4),
        "mean_brier_pwin": round(float(brs.mean()), 4),
        "season_verdicts": season_verdicts,
        "per_fold": [vars(r) for r in fold_results],
    }


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Plackett-Luce trainer scaffold.")
    ap.add_argument("--db", type=Path, default=Path("data/awards.db"))
    ap.add_argument("--selection-id", required=True)
    ap.add_argument("--k", type=int, default=200)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--num-boost-round", type=int, default=250,
                    help="tuned to ~99%% of grouped-CE gain (per tune_rounds): "
                         "MVP/DPOY ~250, ROTY ~100")
    ap.add_argument("--model-version", default=None,
                    help="NLI model_version; merges narrative + bridge cols. "
                         "OMIT for the stats-only control arm.")
    ap.add_argument("--pwin-key", default=None,
                    help="row column holding the model's out-of-fold pwin for "
                         "divergence; None uses the pre-model stat proxy.")
    ap.add_argument("--placebo-narrative", action="store_true",
                    help="scramble narrative columns (feature-count null test)")
    ap.add_argument("--label-col", default="label_vote_share",
                    choices=["label_vote_share", "label_first_place_share"],
                    help="PL training target; the first-place-share experiment")



    args = ap.parse_args(argv)
    conn = connect(args.db)
    cfg = TrainConfig(k_bootstrap=args.k, n_jobs=args.n_jobs,
                      num_boost_round=args.num_boost_round,
                      model_version=args.model_version, pwin_key=args.pwin_key,
                      placebo_narrative=args.placebo_narrative,
                      label_col=args.label_col)
    out = run_award(conn, args.selection_id, cfg)
    print(_json.dumps(out, indent=2, default=str))
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
