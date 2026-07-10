"""Train and PERSIST a single walk-forward fold's K-booster PL ensemble.

The oofraw bundles are out-of-fold SCORES, never persisted boosters. This builds
the deployable artefact they never saved: fit the K-booster grouped-bootstrap
ensemble on all eligible seasons <= train_le for one award/selection, and persist
every booster plus a manifest so a target season can be scored later (score_fold.py)
without retraining.

The fit path is byte-identical to pl_trainer's walk-forward worker: same
grouped_bootstrap_indices, same make_pl_objective (Hessian floor 1e-3), same
seeds (cfg.seed + k), same tree params, same num_boost_round. The ONLY difference
from _bootstrap_fit_predict is that we keep the booster text instead of discarding
it after predicting. That identity is what makes the score_fold 2023 validation
(reproduce the K=50 OOF 2023 number within bootstrap noise) meaningful.

NOT a walk-forward. ONE fold: train <= train_le, persist. Scoring is score_fold.py.
train_le is a parameter: 2022 for the 2023 validation pass, 2023 for the 2024
deliverable. Held-out seasons are never trained on; train_le <= 2023 keeps 2024/2025
out by construction, and we assert it.

British English. season = STARTING year. No em dashes.

Run from repo root:
  uv run python -m scripts.modelling.train.persist_fold --selection-id <id> --train-le 2023 --k 200
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np

try:
    from scripts.common.db import connect
    from scripts.modelling.train.pl_trainer import (
        TrainConfig, read_selection, assemble_matrix, factorise_groups,
        grouped_bootstrap_indices, HELD_OUT_SEASONS,
    )
    from scripts.modelling.train.pl_objective import make_pl_objective
except ImportError:  # pragma: no cover - flat-dir / test fallback
    from db import connect  # type: ignore
    from pl_trainer import (  # type: ignore
        TrainConfig, read_selection, assemble_matrix, factorise_groups,
        grouped_bootstrap_indices, HELD_OUT_SEASONS,
    )
    from pl_objective import make_pl_objective  # type: ignore

log = logging.getLogger("persist_fold")

ARTEFACT_ROOT = Path("models")
HESSIAN_FLOOR = 1e-3


def fold_dir(award: str, selection_id: str, k: int, train_le: int,
             root: Path = ARTEFACT_ROOT) -> Path:
    """Artefact directory for one persisted fold. Keyed so held-out discipline and
    every fit-determining parameter is legible in the path itself."""
    name = f"{award}|{selection_id}|K{k}|train_le{train_le}|hfloor{HESSIAN_FLOOR:g}"
    return root / name


def assert_one_winner_per_group(df, group_ids: np.ndarray) -> dict:
    """A3 stale-materialisation guard. Every (award, season, snapshot) group must
    have EXACTLY one label_won_flag=1. The stale feature_stats_asof bug produced a
    2022 DPOY double-winner; a stale matrix must never be trained on. Aborts the
    run on any violation. Returns a small summary when clean."""
    won = df["label_won_flag"].to_numpy(dtype=float)
    bad_multi: list = []
    bad_zero: list = []
    for g in np.unique(group_ids):
        m = np.flatnonzero(group_ids == g)
        s = int(round(won[m].sum()))
        if s > 1:
            bad_multi.append((int(g), s))
        elif s == 0:
            bad_zero.append(int(g))
    n_groups = int(len(np.unique(group_ids)))
    if bad_multi:
        raise SystemExit(
            f"A3 GUARD FAILED: {len(bad_multi)} group(s) with >1 winner "
            f"(stale feature_stats_asof?). First few (group_id, n_winners): "
            f"{bad_multi[:10]}. Re-materialise before training."
        )
    return {"n_groups": n_groups, "n_groups_zero_winner": len(bad_zero)}


def _fit_one_booster(args):
    """One persistable booster, picklable for a process pool. Mirror of
    pl_trainer._bootstrap_fit_predict EXACTLY (same resample, objective, seed,
    params, rounds) but returns the booster's model TEXT rather than predictions.
    Keeping the math identical is the whole point: a divergence here would make
    the score_fold 2023 validation a comparison of two different estimators."""
    Xtr, ytr, gtr, tree_params, num_boost_round, seed = args
    rng = np.random.default_rng(seed)
    bidx = grouped_bootstrap_indices(gtr, rng)
    b_gids = gtr[bidx]
    import lightgbm as lgb
    fobj = make_pl_objective(b_gids)            # hessian_floor defaults to 1e-3
    params = dict(tree_params)
    params["objective"] = fobj
    params["seed"] = seed
    dtrain = lgb.Dataset(Xtr[bidx], label=ytr[bidx], free_raw_data=False)
    booster = lgb.train(params, dtrain, num_boost_round=num_boost_round)
    return seed, booster.model_to_string()


def persist_fold(conn, selection_id: str, train_le: int, cfg: TrainConfig,
                 root: Path = ARTEFACT_ROOT) -> dict:
    if cfg.model_version is not None:
        raise SystemExit(
            "persist_fold is the STATS-ONLY control (model_version must be None); "
            f"got {cfg.model_version!r}. Narrative/NLI is the strategy chat's overlay."
        )
    if train_le >= min(HELD_OUT_SEASONS):
        raise SystemExit(
            f"train_le={train_le} would admit a held-out season "
            f"{sorted(HELD_OUT_SEASONS)}; train cutoff must be < {min(HELD_OUT_SEASONS)}."
        )

    sel = read_selection(conn, selection_id)
    award = sel["award"]
    feat_cols = list(sel["kept"])
    df = assemble_matrix(conn, award, feat_cols, model_version=None)

    # train rows = all eligible seasons <= train_le (held-out excluded by the
    # cutoff; assert no held-out leaked in regardless).
    train_seasons = sorted(s for s in df["season"].unique()
                           if s <= train_le and s not in HELD_OUT_SEASONS)
    if not train_seasons:
        raise SystemExit(f"no training seasons <= {train_le} for award={award}")
    leaked = HELD_OUT_SEASONS & set(train_seasons)
    if leaked:
        raise SystemExit(f"seal breach: held-out seasons in train set: {sorted(leaked)}")

    tr = df[df["season"].isin(train_seasons)].reset_index(drop=True)
    gtr = factorise_groups(tr)
    guard = assert_one_winner_per_group(tr, gtr)

    Xtr = tr[feat_cols].to_numpy(dtype=float)
    ytr = tr["label_vote_share"].to_numpy(dtype=float)

    # feature-order hash: score_fold MUST assemble columns in this exact order or
    # the boosters read the wrong columns. The hash is the cheap guard against
    # silent column-order drift between train and score.
    feat_hash = hashlib.sha256("\n".join(feat_cols).encode()).hexdigest()[:16]

    out = fold_dir(award, selection_id, cfg.k_bootstrap, train_le, root)
    boosters_dir = out / "boosters"
    boosters_dir.mkdir(parents=True, exist_ok=True)

    seeds = [cfg.seed + k for k in range(cfg.k_bootstrap)]
    jobs = [(Xtr, ytr, gtr, cfg.tree_params, cfg.num_boost_round, s) for s in seeds]

    try:
        from tqdm.auto import tqdm
    except ImportError:
        def tqdm(it=None, **_):
            return it if it is not None else []

    written = 0
    if cfg.n_jobs == 1:
        for k in tqdm(range(cfg.k_bootstrap), total=cfg.k_bootstrap,
                      desc=f"persist {award} train_le{train_le}", unit="fit"):
            seed, text = _fit_one_booster(jobs[k])
            (boosters_dir / f"booster_{k:04d}_seed{seed}.txt").write_text(text)
            written += 1
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=cfg.n_jobs) as ex:
            futs = {ex.submit(_fit_one_booster, jobs[k]): k
                    for k in range(cfg.k_bootstrap)}
            for fut in tqdm(as_completed(futs), total=cfg.k_bootstrap,
                            desc=f"persist {award} train_le{train_le}", unit="fit"):
                k = futs[fut]
                seed, text = fut.result()
                (boosters_dir / f"booster_{k:04d}_seed{seed}.txt").write_text(text)
                written += 1

    manifest = {
        "award": award,
        "selection_id": selection_id,
        "k_bootstrap": cfg.k_bootstrap,
        "train_le": train_le,
        "train_seasons": train_seasons,
        "held_out_seasons": sorted(HELD_OUT_SEASONS),
        "hessian_floor": HESSIAN_FLOOR,
        "num_boost_round": cfg.num_boost_round,
        "seed_base": cfg.seed,
        "seed_convention": "booster k uses seed = seed_base + k (prefix-stable)",
        "n_features": len(feat_cols),
        "feature_cols": feat_cols,
        "feature_order_sha256_16": feat_hash,
        "tree_params": cfg.tree_params,
        "n_train_rows": int(len(tr)),
        "n_train_groups": guard["n_groups"],
        "n_boosters_written": written,
        "point_estimate": "mean",
        "model_version_string": (
            f"{award}|{selection_id}|K{cfg.k_bootstrap}|train_le{train_le}"
            f"|hfloor{HESSIAN_FLOOR:g}|HELDOUT{min(HELD_OUT_SEASONS)}"
        ),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    log.info("persisted %d boosters -> %s", written, out)
    log.info("model_version string: %s", manifest["model_version_string"])
    return manifest


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Train and persist one PL fold's K-booster ensemble.")
    ap.add_argument("--db", type=Path, default=Path("data/awards.db"))
    ap.add_argument("--selection-id", required=True)
    ap.add_argument("--train-le", type=int, required=True,
                    help="train on all eligible seasons <= this (2022 for the 2023 "
                         "validation pass, 2023 for the 2024 deliverable)")
    ap.add_argument("--k", type=int, default=200)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--num-boost-round", type=int, default=250,
                    help="MVP/DPOY ~250, ROTY ~100 (match the walk-forward tune)")
    ap.add_argument("--root", type=Path, default=ARTEFACT_ROOT)
    args = ap.parse_args(argv)

    conn = connect(args.db)
    cfg = TrainConfig(k_bootstrap=args.k, n_jobs=args.n_jobs,
                      num_boost_round=args.num_boost_round, model_version=None)
    manifest = persist_fold(conn, args.selection_id, args.train_le, cfg, root=args.root)
    conn.close()
    print(json.dumps({"persisted": manifest["model_version_string"],
                      "n_boosters": manifest["n_boosters_written"],
                      "train_seasons": [int(x) for x in manifest["train_seasons"]],
                      "dir": str(fold_dir(manifest["award"], args.selection_id,
                                          args.k, args.train_le, args.root))}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
