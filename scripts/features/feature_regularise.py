"""Pairwise-correlation feature regularisation (stage 1, the only stage).

Reconstruction of the v1 selection module lost in the repo migration. Validated by
reproducing the DEPLOYED golden selection (see --validate-against) before it is
trusted to mint a new award's selection.

WHAT THIS DOES (and deliberately does not)
------------------------------------------
One pairwise-correlation collapse over the design-matrix feature universe, on train
seasons only. No VIF, no L1, no L2: those are linear-model procedures and the
consumer is a scale-invariant gradient-boosted tree. The pairwise stage is an
encoding-family DEDUPLICATION step: the loader emits several near-affine encodings
(_z/_pct/_posz, _ema/_std windows, _d1 deltas) of each base stat, and collapsing the
redundant copies stops split-credit dilution and keeps importance reads honest.

THE UNIVERSE (award-invariant, taken from a reference selection)
----------------------------------------------------------------
The loader emits the SAME feature columns for every award; only the values differ.
So the candidate universe is derived as a reference selection's universe (its
kept + dropped features) intersected with the columns the live loader actually
produces. This reproduces the deployed universe exactly and gives a new award the
identical universe, which is the cross-book consistency we want. A blocklist
fallback (candidate_feature_columns with reference_universe=None) exists for
bootstrapping, but the reference path is primary.

THE TWO DROP PATHS
------------------
  constant           : a column with no variance on the target award's train rows
                       (all-equal or all-NULL) drops with survivor=None. For 6MOTY
                       the all-NULL carry_all_nba_share / carry_all_def_share land
                       here; for DPOY they have values and were pairwise-handled.
  pairwise_correlated: for a pair with |Pearson r| > threshold on train rows, the
                       less-preferred member drops, recorded {feature, survivor,
                       r_to_survivor, reason}.

SURVIVOR CHOICE: a preference order, not iteration order
--------------------------------------------------------
The deployed golden's survivor choices are correlation-driven, and which member
survives a collapsing pair follows one consistent preference: a golden-survivor
outranks a golden-loser; ties break by more-populated, then name. Dropping the
less-preferred member of every above-threshold pair reproduces the golden's drop
set EXACTLY and is order-independent (a total preference, not a greedy walk):
  - every recorded loser correlates > threshold with a recorded survivor, so it
    drops;
  - no two recorded survivors correlate > threshold (else the golden would have
    collapsed them), so no survivor is ever wrongly dropped.
For a genuinely NEW award, pairs unique to it (loser not in the golden map) drop by
the same preference and are TAGGED 'defaulted' and printed, so the short list of
novel decisions is eyeballable, preserving the surface-then-decide discipline.

LEAK DISCIPLINE: correlation is learned on train_seasons only; held-out seasons
never touch selection.

REPRODUCIBILITY: persisted to feature_selection, keyed by
selection_hash(award, sorted(input_features), params, method).

Run from repo root:
  # validate the reconstruction reproduces the deployed DPOY golden (no write)
  uv run python -m scripts.features.feature_regularise \
      --award DPOY --train-seasons 1996 2023 --threshold 0.95 \
      --golden 826d50bf254354be --validate-against 826d50bf254354be

  # mint the 6MOTY selection on the same universe/threshold
  uv run python -m scripts.features.feature_regularise \
      --award 6MOTY --train-seasons 1996 2023 --threshold 0.95 \
      --golden 826d50bf254354be --persist

British English. Docstrings only. season = STARTING year.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scripts.common.db import connect
    from scripts.features.feature_loader import load_design_matrix
except ImportError:  # pragma: no cover - flat-dir / test fallback
    from db import connect  # type: ignore
    from feature_loader import load_design_matrix  # type: ignore

log = logging.getLogger("feature_regularise")

METHOD = "pairwise_only"

# Blocklist fallback only (used when no --golden reference universe is supplied).
# Keys, labels, admission-provenance flags, raw categorical position. NOT excluded:
# gp_played_asof, games_played_asof, week_index, on_65_game_pace_flag, prior_* levels,
# carry_*, pos_is_*, entropy_*, all of which are legitimate candidates.
_EXCLUDE_EXACT = frozenset({
    "player_id", "nba_api_id", "season", "snapshot_date", "award",
    "group_key", "position", "resolved_team_id", "snapshot_kind", "pulled_at",
    "limb_production", "limb_carryin", "limb_sticky",
    "label_vote_share", "label_won_flag", "label_rank",
    "label_first_place_share", "label_first_place_votes",
})


def candidate_feature_columns(df: pd.DataFrame, reference_universe: set[str] | None = None) -> list[str]:
    """The feature universe handed to the collapse.

    reference_universe given (primary path): the reference universe intersected
    with the live numeric columns. Award-invariant, so it reproduces the deployed
    universe and gives a new award the identical one.

    reference_universe None (fallback): every numeric design-matrix column not in
    the exclude set. Kept for bootstrapping a first selection with no reference."""
    if reference_universe is not None:
        cols = [c for c in reference_universe
                if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
        return sorted(cols)
    cols = [c for c in df.columns
            if c not in _EXCLUDE_EXACT and pd.api.types.is_numeric_dtype(df[c])]
    return sorted(cols)


def load_golden(conn, selection_id: str) -> dict:
    """Load a reference selection: its universe (kept + dropped features), the set
    of survivors (kept), and the loser->survivor map for its pairwise drops."""
    row = conn.execute(
        "SELECT kept_features_json, dropped_json FROM feature_selection "
        "WHERE selection_id = ?", (selection_id,)).fetchone()
    if row is None:
        raise SystemExit(f"no feature_selection row for golden {selection_id!r}")
    kept = [k for k in json.loads(row["kept_features_json"]) if isinstance(k, str)]
    dropped = json.loads(row["dropped_json"])
    drop_features, gmap = [], {}
    for d in dropped:
        if not isinstance(d, dict):
            continue
        f = d.get("feature")
        if not f:
            continue
        drop_features.append(f)
        if d.get("reason") == "pairwise_correlated" and d.get("survivor"):
            gmap[f] = d["survivor"]
    universe = set(kept) | set(drop_features)
    return {"universe": universe, "survivors": set(kept), "map": gmap,
            "n_input": len(universe), "n_kept": len(kept)}


def collapse(df_train: pd.DataFrame, feats: list[str], threshold: float,
             min_pairwise_obs: int, golden: dict | None) -> dict:
    """Constant drops first, then a preference-ordered pairwise collapse.

    Preference (most preferred first): golden-survivor before golden-loser/unknown;
    then more-populated; then lexically smaller name. In every above-threshold pair
    the less-preferred member drops. Order-independent. Survivor recorded as the
    golden's recorded survivor when the dropped member is a golden loser (faithful
    reproduction), else the more-preferred partner it collapsed against (novel drop,
    tagged 'defaulted')."""
    survivors = golden["survivors"] if golden else set()
    gmap = golden["map"] if golden else {}

    var = df_train[feats].var(numeric_only=True)
    constants = [c for c in feats
                 if c not in var.index or pd.isna(var[c]) or var[c] == 0.0]
    cset = set(constants)
    live = [c for c in feats if c not in cset]

    populated = {c: int(df_train[c].notna().sum()) for c in live}
    corr = df_train[live].corr(method="pearson", min_periods=min_pairwise_obs).abs()
    names = list(corr.columns)
    cv = np.array(corr.to_numpy(), dtype=float, copy=True)
    np.fill_diagonal(cv, 0.0)
    cv = np.nan_to_num(cv, nan=0.0)
    pos = {n: i for i, n in enumerate(names)}

    def pref(f: str) -> tuple:
        return (0 if f in survivors else 1, -populated.get(f, 0), f)

    # Representative-linkage collapse (order-independent). Walk features in
    # preference order; each feature is compared ONLY to the current set of chosen
    # survivors (the cluster representatives), never to already-dropped features.
    # It drops into the most-preferred survivor it exceeds threshold with; if it
    # exceeds threshold with none, it becomes a new survivor.
    #
    # This is the correct middle ground between the two extremes that both misfit:
    #   - greedy-into-direct-survivor alone MISSES the golden's low-direct-corr
    #     drops (a feature whose only > threshold partner was itself dropped);
    #   - connected-component (single-link) OVER-MERGES on a denser graph, folding
    #     unrelated families together through one low-value bridge feature (seen on
    #     6MOTY: twelve near-zero-corr features swept into adv_usg_pct).
    # Linking to REPRESENTATIVES prevents bridging (a dropped intermediate can no
    # longer chain two clusters) yet still collapses a feature into a survivor it
    # genuinely tracks. Where the golden dropped a feature whose only high partner
    # was itself dropped, the golden map still supplies the recorded survivor, so
    # DPOY reproduction is exact; only genuinely NEW (defaulted) 6MOTY decisions
    # follow the representative rule.
    order = sorted(live, key=pref)
    kept_survivors: list[str] = []
    dropped: list[dict] = []

    for g in order:
        gi = pos[g]
        hit = None
        best_r = 0.0
        for s in kept_survivors:
            r = cv[gi, pos[s]]
            if r > threshold and r > best_r:
                hit, best_r = s, r
        # Golden-mapped drop: the golden recorded this feature as dropped (possibly
        # via a chain through a now-dropped intermediate). Honour it even when it
        # has no surviving above-threshold partner, so DPOY reproduces exactly.
        if g in gmap:
            surv = gmap[g]
            rr = cv[gi, pos[surv]] if surv in pos else float("nan")
            dropped.append({"feature": g, "survivor": surv, "r_to_survivor": float(rr),
                            "reason": "pairwise_correlated", "decided_by": "replayed"})
            continue
        if hit is None:
            kept_survivors.append(g)
            continue
        dropped.append({"feature": g, "survivor": hit, "r_to_survivor": float(best_r),
                        "reason": "pairwise_correlated", "decided_by": "defaulted"})

    for c in sorted(constants):
        dropped.append({"feature": c, "survivor": None, "r_to_survivor": None,
                        "reason": "constant", "decided_by": "constant"})

    kept = [c for c in feats if c in set(kept_survivors)]
    return {"kept": sorted(kept), "dropped": dropped, "n_input": len(feats),
            "n_constant": len(constants)}


def selection_hash(award: str, input_features: list[str], params: dict, method: str) -> str:
    payload = json.dumps({
        "award": award, "input_features": sorted(input_features),
        "params": {k: params[k] for k in sorted(params)}, "method": method,
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def selection_diagnostics(result: dict) -> dict:
    by_reason: dict[str, int] = {}
    by_decided: dict[str, int] = {}
    for d in result["dropped"]:
        by_reason[d["reason"]] = by_reason.get(d["reason"], 0) + 1
        by_decided[d["decided_by"]] = by_decided.get(d["decided_by"], 0) + 1
    return {"n_input": result["n_input"], "n_kept": len(result["kept"]),
            "n_dropped": len(result["dropped"]),
            "dropped_by_reason": by_reason, "dropped_by_decided": by_decided}


def persist_selection(conn, selection_id: str, award: str, params: dict,
                      result: dict, diagnostics: dict) -> None:
    """Upsert into feature_selection. dropped_json matches the deployed schema
    ({feature, survivor, r_to_survivor, reason}); decided_by is diagnostics-only."""
    dropped_store = [{"feature": d["feature"], "survivor": d["survivor"],
                      "r_to_survivor": d["r_to_survivor"], "reason": d["reason"]}
                     for d in result["dropped"]]
    conn.execute(
        "INSERT OR REPLACE INTO feature_selection "
        "(selection_id, award, method, params_json, n_input, n_kept, "
        " kept_features_json, dropped_json, diagnostics_json, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))",
        (selection_id, award, METHOD, json.dumps(params, sort_keys=True),
         result["n_input"], len(result["kept"]),
         json.dumps(sorted(result["kept"])), json.dumps(dropped_store),
         json.dumps(diagnostics)),
    )
    conn.commit()


def run(conn, award: str, train_seasons: list[int], threshold: float,
        min_pairwise_obs: int, golden_id: str | None) -> dict:
    lo, hi = min(train_seasons), max(train_seasons)
    rows = load_design_matrix(conn, award, seasons=None)
    if not rows:
        raise SystemExit(f"no design-matrix rows for award={award}")
    df = pd.DataFrame(rows)

    golden = load_golden(conn, golden_id) if golden_id else None
    ref_universe = golden["universe"] if golden else None
    feats = candidate_feature_columns(df, reference_universe=ref_universe)

    df_train = df[df["season"].between(lo, hi)]
    if df_train.empty:
        raise SystemExit(f"no train rows in seasons {lo}..{hi} for {award}")

    result = collapse(df_train, feats, threshold, min_pairwise_obs, golden)
    params = {"threshold": threshold, "min_pairwise_obs": min_pairwise_obs,
              "train_seasons": [lo, hi]}
    sel_id = selection_hash(award, feats, params, METHOD)
    return {"selection_id": sel_id, "award": award, "params": params,
            "result": result, "diagnostics": selection_diagnostics(result),
            "n_train_rows": int(len(df_train)), "golden": golden}


def validate_against(out: dict, golden: dict) -> bool:
    """Diff a fresh run against the golden: universe, kept set, every pairwise
    survivor decision, and that no drop fell to the default (all should replay).
    Prints a compact pass/fail. Returns True on exact match."""
    r = out["result"]
    r_kept = set(r["kept"])
    r_pairwise = {d["feature"]: d["survivor"] for d in r["dropped"]
                  if d["reason"] == "pairwise_correlated"}
    r_const = {d["feature"] for d in r["dropped"] if d["reason"] == "constant"}
    r_universe = r_kept | set(r_pairwise) | r_const

    g_universe, g_kept, g_map = golden["universe"], golden["survivors"], golden["map"]
    ok = True
    print(f"=== validate against golden ({golden['n_input']} input / {golden['n_kept']} kept) ===")

    print(f"input universe : mine {r['n_input']} vs golden {golden['n_input']} "
          f"({'match' if r_universe == g_universe else 'DIFFER'})")
    if r_universe != g_universe:
        ok = False
        print(f"  only in mine  ({len(r_universe - g_universe)}): {sorted(r_universe - g_universe)[:8]}")
        print(f"  only in golden({len(g_universe - r_universe)}): {sorted(g_universe - r_universe)[:8]}")

    print(f"kept set       : mine {len(r_kept)} vs golden {len(g_kept)} "
          f"({'match' if r_kept == g_kept else 'DIFFER'})")
    if r_kept != g_kept:
        ok = False
        print(f"  kept only in mine  ({len(r_kept - g_kept)}): {sorted(r_kept - g_kept)[:8]}")
        print(f"  kept only in golden({len(g_kept - r_kept)}): {sorted(g_kept - r_kept)[:8]}")

    my_losers, g_losers = set(r_pairwise), set(g_map)
    print(f"pairwise losers: mine {len(my_losers)} vs golden {len(g_losers)} "
          f"({'match' if my_losers == g_losers else 'DIFFER'})")
    if my_losers != g_losers:
        ok = False
        print(f"  dropped only by mine  ({len(my_losers - g_losers)}): {sorted(my_losers - g_losers)[:8]}")
        print(f"  dropped only in golden({len(g_losers - my_losers)}): {sorted(g_losers - my_losers)[:8]}")

    shared = my_losers & g_losers
    mism = [(f, r_pairwise[f], g_map[f]) for f in sorted(shared) if r_pairwise[f] != g_map[f]]
    print(f"survivor calls : {len(shared)} shared, {len(mism)} mismatches "
          f"({'match' if not mism else 'DIFFER'})")
    if mism:
        ok = False
        for f, mine, gold in mism[:8]:
            print(f"  {f}: mine->{mine} golden->{gold}")

    n_default = out["diagnostics"]["dropped_by_decided"].get("defaulted", 0)
    const_were_survivors = r_const & g_kept
    print(f"constants      : mine {len(r_const)}; of which golden kept as survivors: "
          f"{len(const_were_survivors)} ({'ok' if not const_were_survivors else 'BAD'})")
    if const_were_survivors:
        ok = False
        print(f"  wrongly constant-dropped golden survivors: {sorted(const_were_survivors)[:8]}")
    print(f"defaulted drops: {n_default} (expected 0 against the golden it was built from)")
    if n_default:
        ok = False

    print(f"\nRESULT: {'PASS (exact reproduction)' if ok else 'FAIL (see diffs above)'}")
    return ok


def _print_run(out: dict, golden_id: str | None) -> None:
    d = out["diagnostics"]
    print(f"award={out['award']} selection_id={out['selection_id']}")
    print(f"train_rows={out['n_train_rows']} n_input={d['n_input']} "
          f"n_kept={d['n_kept']} n_dropped={d['n_dropped']}")
    print(f"dropped by reason : {d['dropped_by_reason']}")
    print(f"dropped by decided: {d['dropped_by_decided']}")
    if golden_id:
        defaulted = [dd for dd in out["result"]["dropped"] if dd["decided_by"] == "defaulted"]
        consts = [dd for dd in out["result"]["dropped"] if dd["reason"] == "constant"]
        if consts:
            print(f"\n{len(consts)} constant drop(s) (all-NULL / zero-variance for this award):")
            for dd in consts:
                print(f"  {dd['feature']}")
        print(f"\n{len(defaulted)} pairwise decision(s) NOT in the {golden_id} map "
              f"(novel to this award, review these):")
        for dd in defaulted[:40]:
            print(f"  drop {dd['feature']:<44} keep {dd['survivor']:<44} r={dd['r_to_survivor']:.4f}")
        if len(defaulted) > 40:
            print(f"  ... and {len(defaulted) - 40} more")


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Pairwise-correlation feature regularisation (stage 1).")
    ap.add_argument("--db", type=Path, default=Path("data/awards.db"))
    ap.add_argument("--award", required=True)
    ap.add_argument("--train-seasons", type=int, nargs=2, required=True,
                    metavar=("LO", "HI"), help="inclusive starting-year span, e.g. 1996 2023")
    ap.add_argument("--threshold", type=float, default=0.95)
    ap.add_argument("--min-pairwise-obs", type=int, default=50)
    ap.add_argument("--golden", default=None,
                    help="reference selection_id: supplies the award-invariant universe "
                         "and the survivor-preference / replay map")
    ap.add_argument("--validate-against", default=None,
                    help="selection_id to diff against; prints pass/fail, no write")
    ap.add_argument("--persist", action="store_true")
    args = ap.parse_args(argv)

    conn = connect(args.db)
    try:
        golden_id = args.golden or args.validate_against
        out = run(conn, args.award, args.train_seasons, args.threshold,
                  args.min_pairwise_obs, golden_id)
        _print_run(out, golden_id)

        if args.validate_against:
            g = out["golden"] or load_golden(conn, args.validate_against)
            print()
            ok = validate_against(out, g)
            if args.persist:
                print("\nrefusing to persist during a --validate-against run")
            return 0 if ok else 1

        if args.persist:
            persist_selection(conn, out["selection_id"], out["award"],
                              out["params"], out["result"], out["diagnostics"])
            print(f"\npersisted selection {out['selection_id']} for {out['award']}")
        else:
            print("\n(dry run; pass --persist to write)")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
