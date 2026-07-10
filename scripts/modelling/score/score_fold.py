"""Score a target season with a persisted PL booster ensemble; write model_predictions.

Consumes the artefact persist_fold.py wrote (K boosters + manifest), assembles the
target season's as-of design matrix through the SAME load path the trainer uses
(feature_loader.load_design_matrix, model_version=None), predicts each candidate's
score under every booster, and reduces the K-score ensemble into the four distinct
prediction objects the strategy chat reads. Writes one model_predictions row per
(model_version, snapshot_date, season, award, player_id).

THE FOUR OBJECTS ARE NOT INTERCHANGEABLE (project has been bitten conflating them):
  mu              = bootstrap MEAN of the K raw scores for the candidate.
  sigma           = bootstrap STD of the K raw scores (Kelly sizing reads this).
  vote_share_pred = grouped softmax of the K-MEAN score within the group
                    (the mean-score object).
  p_win           = Monte Carlo: per booster, within-group softmax, argmax; p_win
                    is the fraction of boosters where the candidate is the argmax
                    (the noisy-near-ties object, NOT softmax-of-mean).
CIs are empirical bootstrap quantiles (default 10/90) of the per-booster
within-group share (vote_share) and of a bootstrapped p_win.

VALIDATION USE: score_fold with a manifest trained train_le=2022 against season=2023
must reproduce the existing K=50 OOF 2023 numbers within bootstrap noise, because
persist_fold's train set for train_le=2022 IS the walk-forward's 2023-fold train set
and the fit is byte-identical. That is the leak/identity gate before trusting 2024.

British English. season = STARTING year. No em dashes.

Run from repo root:
  uv run python -m scripts.modelling.score.score_fold --selection-id <id> --train-le 2023 --season 2024
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
    from scripts.common.db import connect, upsert, utcnow_iso
    from scripts.modelling.train.pl_trainer import (
        read_selection, assemble_matrix, factorise_groups, pwin_from_ensemble,
    )
    from scripts.modelling.train.pl_objective import grouped_softmax
    from scripts.modelling.train.persist_fold import fold_dir, ARTEFACT_ROOT, HESSIAN_FLOOR
except ImportError:  # pragma: no cover
    from db import connect, upsert, utcnow_iso  # type: ignore
    from pl_trainer import (  # type: ignore
        read_selection, assemble_matrix, factorise_groups, pwin_from_ensemble,
    )
    from pl_objective import grouped_softmax  # type: ignore
    from persist_fold import fold_dir, ARTEFACT_ROOT, HESSIAN_FLOOR  # type: ignore

log = logging.getLogger("score_fold")


def _load_manifest_and_boosters(d: Path):
    manifest = json.loads((d / "manifest.json").read_text())
    booster_files = sorted((d / "boosters").glob("booster_*.txt"))
    if not booster_files:
        raise SystemExit(f"no booster files under {d / 'boosters'}")
    if len(booster_files) != manifest["k_bootstrap"]:
        raise SystemExit(
            f"manifest says K={manifest['k_bootstrap']} but found "
            f"{len(booster_files)} booster files in {d}")
    import lightgbm as lgb
    boosters = [lgb.Booster(model_str=f.read_text()) for f in booster_files]
    return manifest, boosters


def _predict_ensemble(boosters, X: np.ndarray) -> np.ndarray:
    """(K, n_rows) raw scores: booster k's prediction for every scoring row."""
    K = len(boosters)
    ens = np.empty((K, X.shape[0]), dtype=np.float64)
    for k, b in enumerate(boosters):
        ens[k] = b.predict(X)
    return ens


def _pwin_ci_bootstrap(ens: np.ndarray, group_ids: np.ndarray,
                       lo: float, hi: float, n_resample: int,
                       seed: int) -> tuple[np.ndarray, np.ndarray]:
    """CI on p_win by resampling the K boosters with replacement n_resample times;
    each resample yields a p_win vector (argmax-fraction over the resampled
    boosters); return the lo/hi empirical quantiles per row. Separate from the
    point p_win (which uses all K)."""
    K, n = ens.shape
    rng = np.random.default_rng(seed)
    draws = np.empty((n_resample, n), dtype=np.float64)
    for b in range(n_resample):
        idx = rng.integers(0, K, size=K)
        draws[b] = pwin_from_ensemble(ens[idx], group_ids)
    return (np.percentile(draws, lo, axis=0), np.percentile(draws, hi, axis=0))


def score_fold(conn, selection_id: str, train_le: int, season: int,
               k: int, root: Path = ARTEFACT_ROOT,
               ci_lo: float = 10.0, ci_hi: float = 90.0,
               pwin_ci_resample: int = 200, seed: int = 17,
               write: bool = True) -> dict:
    sel = read_selection(conn, selection_id)
    award = sel["award"]
    d = fold_dir(award, selection_id, k, train_le, root)
    manifest, boosters = _load_manifest_and_boosters(d)

    if season in set(manifest["train_seasons"]):
        raise SystemExit(
            f"season {season} is IN the persisted train set (train_le={train_le}); "
            f"scoring it would be in-sample. Pick a season > train_le.")

    feat_cols = list(manifest["feature_cols"])
    feat_hash = hashlib.sha256("\n".join(feat_cols).encode()).hexdigest()[:16]
    if feat_hash != manifest["feature_order_sha256_16"]:
        raise SystemExit("manifest feature hash self-mismatch; artefact corrupt")

    # assemble ONLY the target season, model_version=None (stats-only control).
    df = assemble_matrix(conn, award, feat_cols, seasons=[season], model_version=None)
    if df.empty:
        raise SystemExit(f"no scoring rows for award={award} season={season}")

    # column-order guard: the matrix feature columns must match the trained order
    # exactly or the boosters read the wrong columns. assemble_matrix orders meta
    # first then kept features in selection order, so reselect kept in that order.
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"scoring matrix missing {len(missing)} trained columns: {missing[:10]}")
    X = df[feat_cols].to_numpy(dtype=float)
    gids = factorise_groups(df)

    ens = _predict_ensemble(boosters, X)            # (K, n_rows)

    mu = ens.mean(axis=0)
    sigma = ens.std(axis=0)
    vote_share_pred = grouped_softmax(mu, gids)     # softmax of the K-mean score
    p_win = pwin_from_ensemble(ens, gids)           # Monte Carlo argmax-fraction

    # per-booster within-group share, for the vote_share CI
    shares = np.empty_like(ens)
    for kk in range(ens.shape[0]):
        shares[kk] = grouped_softmax(ens[kk], gids)
    vs_lo = np.percentile(shares, ci_lo, axis=0)
    vs_hi = np.percentile(shares, ci_hi, axis=0)
    pw_lo, pw_hi = _pwin_ci_bootstrap(ens, gids, ci_lo, ci_hi, pwin_ci_resample, seed)

    model_version = manifest["model_version_string"]
    stamp = utcnow_iso()
    rows = []
    for i in range(len(df)):
        r = df.iloc[i]
        rows.append({
            "model_version": model_version,
            "snapshot_date": r["snapshot_date"],
            "season": int(r["season"]),
            "award": award,
            "player_id": int(r["player_id"]),
            "mu": float(mu[i]),
            "sigma": float(sigma[i]),
            "vote_share_pred": float(vote_share_pred[i]),
            "vote_share_ci_lo": float(vs_lo[i]),
            "vote_share_ci_hi": float(vs_hi[i]),
            "p_win": float(p_win[i]),
            "p_win_ci_lo": float(pw_lo[i]),
            "p_win_ci_hi": float(pw_hi[i]),
        })

    if write:
        upsert(conn, "model_predictions", rows,
               ["model_version", "snapshot_date", "award", "player_id"])
        conn.commit()

    # sanity: max |score| (Hessian-floor check; pathological boosters emit thousands)
    max_abs = float(np.abs(ens).max())
    summary = {
        "award": award, "season": season, "train_le": train_le, "k": k,
        "model_version": model_version,
        "n_rows": len(rows),
        "n_groups": int(len(np.unique(gids))),
        "n_snapshots": int(df["snapshot_date"].nunique()),
        "max_abs_score": round(max_abs, 3),
        "ci_interval": f"{ci_lo:g}/{ci_hi:g}",
        "wrote_rows": write,
    }
    log.info("scored %s season=%s: %d rows, %d groups, max|score|=%.2f",
             award, season, len(rows), summary["n_groups"], max_abs)
    if max_abs > 100:
        log.warning("max|score|=%.1f is large; check Hessian floor / pathological "
                    "boosters (healthy is single digits)", max_abs)
    return summary


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Score a target season with persisted PL boosters.")
    ap.add_argument("--db", type=Path, default=Path("data/awards.db"))
    ap.add_argument("--selection-id", required=True)
    ap.add_argument("--train-le", type=int, required=True)
    ap.add_argument("--season", type=int, required=True, help="target season to score (> train_le)")
    ap.add_argument("--k", type=int, default=200)
    ap.add_argument("--root", type=Path, default=ARTEFACT_ROOT)
    ap.add_argument("--ci-lo", type=float, default=10.0)
    ap.add_argument("--ci-hi", type=float, default=90.0)
    ap.add_argument("--no-write", action="store_true", help="compute and report but do not upsert")
    args = ap.parse_args(argv)
    conn = connect(args.db)
    summary = score_fold(conn, args.selection_id, args.train_le, args.season,
                         args.k, root=args.root, ci_lo=args.ci_lo, ci_hi=args.ci_hi,
                         write=not args.no_write)
    conn.close()
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
