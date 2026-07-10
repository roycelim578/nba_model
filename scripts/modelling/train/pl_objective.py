"""Custom LightGBM Plackett-Luce grouped multinomial cross-entropy objective.

WHAT THIS IS
------------
Within a (award, season, snapshot) the admitted candidates are NOT independent:
their vote shares sum to 1 by construction and they compete zero-sum for one
award. The model emits a latent worth SCORE per candidate; a softmax taken
STRICTLY within the admitted group turns those scores into shares that sum to 1,
mirroring how voters split a fixed pie. The loss is multinomial cross-entropy of
that grouped softmax against the observed (zero-padded, within-group-normalised)
vote share. Each candidate's gradient depends on every group member through the
shared softmax denominator: that learned codependence is the whole reason for a
custom objective rather than a stock tabular loss.

THE SEAM THE TRAINER IMPORTS
----------------------------
The trainer (master chat) imports exactly:

    from scripts.modelling.train.pl_objective import make_pl_objective

and uses it per fold like this:

    # 1. The loader (feature_loader.load_design_matrix) emits a string
    #    group_key = f"{award}|{season}|{snapshot}" per row, in the same row
    #    order as the design matrix X.
    # 2. Factorise that string column to a contiguous integer code array, SAME
    #    ROW ORDER as X (pandas.factorize is fine; np.unique(..., return_inverse)
    #    is fine). The integer values themselves do not matter; only equality and
    #    row alignment matter.
    group_ids = pd.factorize(df["group_key"])[0]            # int array, len n_rows
    fobj = make_pl_objective(group_ids)                      # closure over groups
    params = {"objective": fobj, "num_leaves": 31, ...}      # lgbm>=4: in params
    booster = lgb.train(
        params,                                              # objective lives here
        train_set,                                           # label = label_vote_share
        feval=make_pl_eval(group_ids),                       # optional, see below
        ...
    )

NOTE ON THE LIGHTGBM API (lightgbm >= 4.0, confirmed against 4.6):
lgb.train no longer accepts a separate `fobj=` argument; the custom objective is
passed as params["objective"] = make_pl_objective(group_ids). A custom metric is
still passed as feval=. The sklearn API equivalent
(LGBMRegressor(objective=fobj)) also works. pyproject already pins lightgbm>=4.0,
so the trainer must use the params-objective form, NOT the pre-4.0 fobj kwarg.

CRITICAL ALIGNMENT CONTRACT (trainer must honour)
-------------------------------------------------
group_ids[i] must describe the SAME row as train_set's row i and as predt[i] at
call time. LightGBM does not reorder rows for a custom fobj, so a single
group_ids array built in the loader's row order is correct for the whole fold.
Build a FRESH closure per fold (the group_ids change per fold); do not reuse one
closure across folds.

The label passed in the Dataset must be the within-group-normalised, zero-padded
vote share (the loader's label_vote_share). This objective NEVER renormalises,
clips, or re-pads the label. See the note at the bottom of this module about the
normalisation responsibility, which currently sits upstream.

group_ids may be non-contiguous and unsorted: grouping is by EQUALITY of id only.

DIAGONAL HESSIAN
----------------
hess = p * (1 - p) is the diagonal of the softmax-cross-entropy Hessian. The true
Hessian is the full softmax Jacobian (dense per group); LightGBM uses and expects
the diagonal, which is the standard, correct choice here. We deliberately do NOT
build the full matrix.

Run/test layout (PINNED):
    module: src/data/pl_objective.py   -> uv run python -m scripts.modelling.train.pl_objective
    tests : tests/test_pl_objective.py -> uv run pytest tests/test_pl_objective.py -q
"""

from __future__ import annotations

import numpy as np

__all__ = ["make_pl_objective", "make_pl_eval", "grouped_softmax", "grouped_ce_loss"]


def _group_slices(group_ids: np.ndarray) -> dict[int, np.ndarray]:
    """Map each distinct group id to the integer positions of its rows, in
    original row order. Grouping is by EQUALITY only; no sort/contiguity assumed.

    Returns a dict {gid: positions_array}. Built once and reused: the closures
    below capture group_ids and rebuild slices on each call, which is cheap
    relative to a boosting round and keeps the objective stateless w.r.t. the
    booster. (np.unique with return_inverse is O(n log n); fine for our row
    counts of a few thousand.)
    """
    # argsort-free bucketing preserving original positions
    order = np.argsort(group_ids, kind="stable")
    sorted_ids = group_ids[order]
    # boundaries where the sorted id changes
    boundaries = np.flatnonzero(np.diff(sorted_ids)) + 1
    chunks = np.split(order, boundaries)
    slices: dict[int, np.ndarray] = {}
    for ch in chunks:
        slices[int(group_ids[ch[0]])] = ch
    return slices


def grouped_softmax(scores: np.ndarray, group_ids: np.ndarray) -> np.ndarray:
    """Within-group softmax. p[i] = exp(s_i - max_g) / sum_{j in g} exp(s_j - max_g)
    where g is i's group. Numerically stabilised by per-group max subtraction.
    Row order of the output matches the input. Exposed for tests and for the
    trainer's inference path (it needs the identical within-group softmax to turn
    scores into P(share))."""
    scores = np.asarray(scores, dtype=np.float64)
    out = np.empty_like(scores)
    for _, pos in _group_slices(group_ids).items():
        sg = scores[pos]
        sg = sg - sg.max()           # stability: subtract group max before exp
        ex = np.exp(sg)
        out[pos] = ex / ex.sum()
    return out


def grouped_ce_loss(scores: np.ndarray, labels: np.ndarray,
                    group_ids: np.ndarray) -> float:
    """The scalar loss the gradient must be consistent with:
        L = - sum_i y_i * log(p_i),  p = within-group softmax(scores).
    Used by the finite-difference test as the ground-truth objective. A small
    floor guards log(0) for zero-probability rows that carry zero label mass
    anyway (those terms contribute 0 in the limit; the floor only avoids -inf*0
    NaNs in intermediate arithmetic)."""
    p = grouped_softmax(scores, group_ids)
    labels = np.asarray(labels, dtype=np.float64)
    return float(-np.sum(labels * np.log(np.clip(p, 1e-300, 1.0))))


def make_pl_objective(group_ids: np.ndarray, hessian_floor: float = 1e-3):
    """Factory: returns the fobj closure LightGBM calls each boosting round.

    group_ids : 1-D int array, length n_rows, SAME ROW ORDER as the training
                matrix. Rows sharing an id form one softmax group. Integer-coded
                upstream by the trainer (factorised from the loader's group_key);
                this objective only tests equality and never assumes the ids are
                sorted or contiguous.

    The returned closure has signature fobj(predt, dtrain) -> (grad, hess), both
    1-D float arrays length n_rows, in the SAME row order as predt.
    """
    group_ids = np.asarray(group_ids)
    if group_ids.ndim != 1:
        raise ValueError("group_ids must be 1-D")
    # precompute the grouping once; the slice map is fixed for this fold
    slices = _group_slices(group_ids)
    n_rows = group_ids.shape[0]

    def pl_grouped_objective(predt: np.ndarray, dtrain):
        predt = np.asarray(predt, dtype=np.float64)
        if predt.shape[0] != n_rows:
            raise ValueError(
                f"predt length {predt.shape[0]} != group_ids length {n_rows}; "
                "the trainer must build group_ids in the design-matrix row order "
                "and pass a fresh closure per fold."
            )
        y = np.asarray(dtrain.get_label(), dtype=np.float64)
        grad = np.empty(n_rows, dtype=np.float64)
        hess = np.empty(n_rows, dtype=np.float64)
        for _, pos in slices.items():
            sg = predt[pos]
            sg = sg - sg.max()
            ex = np.exp(sg)
            pg = ex / ex.sum()
            grad[pos] = pg - y[pos]
            hess[pos] = np.maximum(pg * (1.0 - pg), hessian_floor)   # floored diagonal Hessian
        return grad, hess

    return pl_grouped_objective


def make_pl_eval(group_ids: np.ndarray):
    """Optional matching eval metric for the watchlist: mean grouped CE loss per
    row (lower is better). Signature feval(predt, dtrain) -> (name, value,
    is_higher_better). Provided as a nice-to-have so the trainer can monitor the
    SAME loss the objective optimises; the trainer may ignore it. Built per fold
    with the fold's group_ids, exactly like the objective."""
    group_ids = np.asarray(group_ids)
    n_rows = group_ids.shape[0]

    def pl_grouped_eval(predt: np.ndarray, dtrain):
        predt = np.asarray(predt, dtype=np.float64)
        y = np.asarray(dtrain.get_label(), dtype=np.float64)
        loss = grouped_ce_loss(predt, y, group_ids)
        return "pl_grouped_ce", loss / max(n_rows, 1), False

    return pl_grouped_eval


if __name__ == "__main__":  # tiny smoke; real tests live in tests/
    rng = np.random.default_rng(0)
    g = np.array([0, 0, 0, 1, 1, 2])
    s = rng.normal(size=6)
    y = np.zeros(6)
    y[0] = 1.0           # group 0 winner
    y[3] = 0.7; y[4] = 0.3
    fobj = make_pl_objective(g)

    class _D:
        def get_label(self):
            return y

    grad, hess = fobj(s, _D())
    print("grad", np.round(grad, 4))
    print("hess", np.round(hess, 4))
    print("loss", round(grouped_ce_loss(s, y, g), 4))
