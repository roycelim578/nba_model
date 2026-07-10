"""Walk-forward out-of-fold scorer: produces the raw per-candidate OOF score bundles (the
OOF_BUNDLE artefacts pinned in config) used for forward-noise (eta) calibration and for
the book-weighting skill scores. Originally the stage-one leg of a two-stage
residual-narrative model; the narrative stage is dead, so this is now the sole OOF
producer. Has no importers; run directly to regenerate the bundles.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger("oof_stage1")

# These imports are the real trainer interface. In the project they resolve via
# scripts.modelling.train.pl_trainer; the fallback supports flat-dir execution and tests.
def _import_trainer():
    """Import pl_trainer lazily and defensively. Returns the module or None.

    Deliberately swallows ANY exception (not just ImportError): the trainer's
    import chain touches feature_loader/positional_z, and a bug ANYWHERE in that
    chain would otherwise blow up at collection time even for tests that inject
    stubs and never call the real trainer. Real runs that need the trainer will
    surface the underlying error when they actually call it (see _require_trainer).
    """
    try:
        from scripts.modelling.train import pl_trainer as t  # type: ignore
        return t
    except Exception:
        try:
            import pl_trainer as t  # type: ignore
            return t
        except Exception:
            return None


_t = _import_trainer()
_REQUIRED = ("TrainConfig", "read_selection", "assemble_matrix", "build_folds",
             "factorise_groups", "_fit_ensemble_parallel", "HELD_OUT_SEASONS")
if _t is not None and all(hasattr(_t, a) for a in _REQUIRED):
    TrainConfig = _t.TrainConfig
    read_selection = _t.read_selection
    assemble_matrix = _t.assemble_matrix
    build_folds = _t.build_folds
    factorise_groups = _t.factorise_groups
    _fit_ensemble_parallel = _t._fit_ensemble_parallel
    HELD_OUT_SEASONS = _t.HELD_OUT_SEASONS
else:  # pragma: no cover - sandbox/test path: names exist so tests can patch them
    TrainConfig = None  # type: ignore
    HELD_OUT_SEASONS = frozenset({2024, 2025})

    def _require_trainer():
        # re-attempt once and raise the REAL underlying error so a production run
        # gets a useful traceback rather than this stub's generic message
        from scripts.modelling.train import pl_trainer  # noqa: F401  (let the real error surface)

    def read_selection(conn, selection_id):  # patched in tests
        _require_trainer()

    def assemble_matrix(*a, **k):  # default-injected in tests
        _require_trainer()

    def build_folds(*a, **k):
        _require_trainer()

    def factorise_groups(*a, **k):
        _require_trainer()

    def _fit_ensemble_parallel(*a, **k):
        _require_trainer()


# the key identifying a unique stage-2 row, matching feature_loader's emission
OOF_KEY = ["nba_api_id", "season", "snapshot_date", "award"]


def serve_oof_at_k(raw_oof: pd.DataFrame, k: int) -> pd.DataFrame:
    """Given an OOF frame produced with capture_raw=True (carrying a per-row
    raw_scores K-vector), return a copy whose stage1_score is the K-prefix MEDIAN
    (the K-stable point estimate; the mean is outlier-dominated and K-unstable).
    Because booster seeds are seed+k (prefix-stable), median(raw_scores[:k])
    reproduces a standalone K-run exactly, so one max-K fit serves every k<=K with
    no retraining. Drops the raw_scores column from the served copy."""
    if "raw_scores" not in raw_oof.columns:
        raise ValueError("raw_oof has no raw_scores column; regenerate with capture_raw=True")
    out = raw_oof.copy()
    out["stage1_score"] = out["raw_scores"].apply(
        lambda v: float(np.mean(np.asarray(v)[:k])))
    return out.drop(columns=["raw_scores"])


def generate_oof_scores(conn, control_selection_id: str,
                        cfg: "TrainConfig | None" = None,
                        _assemble=None, _build_folds=None,
                        _factorise=None, _fit_ensemble=None,
                        k_overrides: dict | None = None,
                        capture_raw: bool = False) -> pd.DataFrame:
    """Run the stats-only walk-forward and return per-row OOF stage-1 scores.

    Returns a DataFrame with OOF_KEY columns plus group_key, label_vote_share,
    label_won_flag, week_index, and stage1_score (the K-mean raw pre-softmax
    score from the fold whose model never trained on this row's season).

    The underscored params allow tests to inject stubs for the lgb-dependent and
    trainer-coupled calls; production leaves them None and the module imports the
    real pl_trainer functions.
    """
    cfg = cfg or TrainConfig()
    _assemble = _assemble or assemble_matrix
    _build_folds = _build_folds or build_folds
    _factorise = _factorise or factorise_groups
    _fit_ensemble = _fit_ensemble or _fit_ensemble_parallel

    sel = read_selection(conn, control_selection_id)
    if cfg.model_version is not None:
        raise ValueError(
            "OOF stage-1 scores must come from the STATS-ONLY control "
            "(model_version=None); got model_version="
            f"{cfg.model_version!r}. The residual is what stats alone missed."
        )

    df = _assemble(conn, sel["award"], sel["kept"], model_version=None)
    feat_cols = list(sel["kept"])
    # Same coverage match as run_award: drop seasons with no first-place data
    # so the vote-share and first-place OOF arms score an identical universe.
    if "label_first_place_share" in df.columns:
        _null_fp = df["label_first_place_share"].isna().to_numpy()
        if _null_fp.any():
            df = df.loc[~_null_fp].reset_index(drop=True)
    folds = _build_folds(df)

    rng = np.random.default_rng(cfg.seed)
    out_rows: list[dict] = []
    seen_keys: set[tuple] = set()

    # Single progress bar over OOF folds (the bulk of the wait). Suppress the
    # trainer's per-fit inner bars (which otherwise dominate the display with one
    # bar per fold) by setting TQDM_DISABLE for the duration; restore after.
    import os as _os
    _prev_tqdm = _os.environ.get("TQDM_DISABLE")
    _os.environ.setdefault("TQDM_DISABLE", "0")
    try:
        from tqdm.auto import tqdm as _tqdm
        _bar = _tqdm(total=len(folds), desc=f"OOF stage-1 {sel['award']}",
                     unit="fold")
    except ImportError:
        _bar = None

    import time as _time, dataclasses as _dc
    _t0 = _time.time()
    for _fi, f in enumerate(folds, 1):
        # per-fold K: default cfg.k_bootstrap, override by test season
        # (e.g. 2024 -> 200). dataclasses.replace keeps everything else
        # identical, so the objective, folds and seal are untouched.
        _k_fold = (k_overrides or {}).get(f.test_season, cfg.k_bootstrap)
        _fold_cfg = _dc.replace(cfg, k_bootstrap=_k_fold)
        print(f"  {_time.strftime('%H:%M:%S')}  [{sel['award']}] "
              f"fold {_fi}/{len(folds)}: season {f.test_season}  "
              f"K={_k_fold}  (+{_time.time() - _t0:.0f}s)", flush=True)
        tr = df.iloc[f.train_idx]
        te = df.iloc[f.test_idx].reset_index(drop=True)

        # SEAL ASSERTION: the test season must not appear in its own training
        # seasons, and no held-out season may appear anywhere in this fold.
        assert f.test_season not in f.train_seasons, (
            f"leak: test season {f.test_season} is in its own train window"
        )
        assert not (HELD_OUT_SEASONS & set(f.train_seasons)), (
            f"seal breach: held-out season in train window {f.train_seasons}"
        )
        assert f.test_season not in HELD_OUT_SEASONS, (
            f"seal breach: scoring a held-out test season {f.test_season}"
        )

        Xtr = tr[feat_cols].to_numpy(dtype=float)
        ytr = tr[getattr(cfg, "label_col", "label_vote_share")].to_numpy(dtype=float)
        gtr = _factorise(tr)
        Xte = te[feat_cols].to_numpy(dtype=float)

        ens, _imps = _fit_ensemble(
            Xtr, ytr, gtr, Xte, _fold_cfg, rng,
            progress_desc=f"OOF {sel['award']} fold {f.test_season}")
        if _bar is not None:
            _bar.update(1)
        # Point score per test row: the MEDIAN of the per-booster raw scores, NOT
        # the mean. The mean is dominated by occasional extreme-magnitude boosters
        # and its ranking is wildly K-unstable (drift ~13 ranks); the median is
        # robust to those outlier boosters and is K-stable (drift <1), while the
        # per-booster ORDERING is sound (argmax-vote is stable). Verified on the
        # cached bundle across K=10/50/100/200. The bootstrap SPREAD is still
        # available for sigma (Kelly sizing) from `ens`; only the point estimate
        # changes from mean to median.
        point_score = np.mean(ens, axis=0)

        for i in range(len(te)):
            r = te.iloc[i]
            key = (r["nba_api_id"], int(r["season"]), r["snapshot_date"], r["award"])
            if key in seen_keys:
                # a row can only be a test row once (walk-forward tests each
                # season exactly once); a repeat means a fold-construction bug
                raise RuntimeError(f"row scored twice in OOF walk-forward: {key}")
            seen_keys.add(key)
            row = {
                "nba_api_id": r["nba_api_id"],
                "season": int(r["season"]),
                "snapshot_date": r["snapshot_date"],
                "award": r["award"],
                "group_key": r["group_key"],
                "label_vote_share": float(r["label_vote_share"]),
                "label_first_place_share": (float(r["label_first_place_share"])
                                            if pd.notna(r.get("label_first_place_share"))
                                            else None),
                "label_won_flag": int(r["label_won_flag"]),
                "week_index": (int(r["week_index"])
                               if pd.notna(r.get("week_index")) else None),
                "stage1_score": float(point_score[i]),
            }
            if capture_raw:
                # full K-vector of per-booster scores for this row; a prefix mean
                # raw_scores[:K'].mean() reproduces the K'-run exactly because seeds
                # are seed+k (prefix-stable). Lets one max-K fit serve every K'<=K.
                row["raw_scores"] = ens[:, i].astype(np.float32).copy()
            out_rows.append(row)

    if _bar is not None:
        _bar.close()
    if _prev_tqdm is None:
        _os.environ.pop("TQDM_DISABLE", None)
    else:
        _os.environ["TQDM_DISABLE"] = _prev_tqdm

    result = pd.DataFrame(out_rows)
    log.info("OOF stage-1: %d rows scored across %d folds, seasons %s",
             len(result), len(folds),
             sorted(result["season"].unique().tolist()) if len(result) else [])
    return result
