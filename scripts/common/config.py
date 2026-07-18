"""Pinned artefacts and sealed-run constants. Single source of truth for which
model, OOF bundle and parameters the backtest loads, replacing glob-latest so a
later retrain cannot silently change a sealed result. Paths are relative to the
repo root (the run cwd). British English."""
from __future__ import annotations

FINAL_MODEL = {
    "MVP":  "models/MVP/final/final_MVP_baseline_K200_20260707T155629Z.pkl",
    "DPOY": "models/DPOY/final/final_DPOY_baseline_K200_20260707T155629Z.pkl",
    "ROTY": "models/ROTY/final/final_ROTY_vs_K200_20260707T155629Z.pkl",
}

OOF_BUNDLE = {
    "MVP":  "models/MVP/oof/oofraw_MVP_0834c9e4af92cc7e_20260705T071517Z.pkl",
    "DPOY": "models/DPOY/oof/oofraw_DPOY_0769610297ef9ccf_20260705T052958Z.pkl",
    "ROTY": "models/ROTY/oof/oofraw_ROTY_ef8e4bf34ca1f05d_20260705T072115Z.pkl",
}

# eta forward-uncertainty is fit on OOF residuals from seasons at or before this,
# strictly before the 2024/2025 test seasons. Do not raise without re-seeding eta.
OOF_SEASON_CAP = 2023

# Pinned per-season book weights (shrunk contender-conditioned Brier skill score, see
# scripts/strategy/allocation/book_weighting.py). Computed once per season from OOF of
# seasons strictly before it, eyeballed, pinned here. Regenerate and repin when the book
# set changes (e.g. adding 6MOTY re-splits every listed season). Dollar budgets at the
# 3000 bankroll; DPOY 741 is 742 rounded down so the split sums to 3000.
BOOK_WEIGHTS = {
    2025: {"MVP": 951, "DPOY": 741, "ROTY": 1308},
    # Validation-only, NOT production. Flat 1000 per stat book for the 2024
    # six-book integration run; the principled shrunk-BSS re-split is deferred
    # until the 2024 run yields the stat books' realised skill. Do not ship.
    2024: {"MVP": 951, "DPOY": 741, "ROTY": 1308,
           "PTS": 1000, "REB": 1000, "AST": 1000,
           "STL": 1000, "BLK": 1000},
}

# Held-out sealed seasons per award. A backtest or score that touches one of these for
# that award raises (see assert_not_sealed). MVP/DPOY/ROTY have spent their one-shot 2025
# test, so they hold out nothing further. Any award NOT listed defaults to holding out
# both 2024 and 2025, so a new award (6MOTY, stat leaders) is protected even if someone
# forgets to register it.
SEAL_REGISTRY = {
    "MVP": [],
    "DPOY": [],
    "ROTY": [],
    "6MOTY": [2024, 2025],
    "PTS": [],
    "REB": [],
    "AST": [],
    "STL": [],
    "BLK": [],
}
_DEFAULT_SEAL = [2024, 2025]


def assert_not_sealed(award, season):
    """Raise if `season` is a sealed held-out season for `award`. Protects the one-shot
    out-of-sample test from being spent during development."""
    held = SEAL_REGISTRY.get(award, _DEFAULT_SEAL)
    if int(season) in held:
        raise RuntimeError(
            f"season {season} is SEALED for {award} (SEAL_REGISTRY gives {held}); "
            f"refusing to score or backtest it. This guard protects the one-shot test.")

COST_PARAMS_PATH = "data/cost_params.json"
FORWARD_VOL_PATH = "models/forward_vol/forward_vol.pkl"
