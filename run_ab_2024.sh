#!/usr/bin/env bash
# Unattended 2024 A/B: control (curvature band) vs reform (vol band), both arms.
# Run from the repo root, wrapped in caffeinate so the Mac does not sleep:
#     caffeinate -i bash run_ab_2024.sh              sequential, ~1.5-2h
#     caffeinate -i bash run_ab_2024.sh --parallel   two arms at once, ~half that
# In parallel mode each arm still runs its control then reform in sequence; only the
# two arms overlap, so at most two heavy processes at a time. Scoring is read-only on
# the WAL database, so concurrent reads are safe. All logs land under
# out/ab_2024_<timestamp>/. No 2025 is touched. British English.

set -uo pipefail
cd "$(dirname "$0")"

PARALLEL=0
if [ "${1:-}" = "--parallel" ] || [ "${1:-}" = "-p" ]; then PARALLEL=1; fi

RA=scripts/strategy/trade_regions/region_adapter.py
SENTINEL='_pool.ndim == 2'
TS=$(date +%Y%m%d_%H%M%S)
LOG=out/ab_2024_$TS
mkdir -p "$LOG"

say () { echo; echo "======== $* ========"; }

say "[0] ensure region_adapter carries the fixed dispersion capture"
if grep -qF "$SENTINEL" "$RA"; then
  echo "already patched; skipping patch step."
else
  if ! grep -qF "$SENTINEL" patch_dispersion_diag.py; then
    echo "FATAL: patch_dispersion_diag.py is not the fixed version; copy the fixed one in first."
    exit 1
  fi
  [ -f "$RA.bak_disp" ] && cp "$RA.bak_disp" "$RA"
  uv run python3 patch_dispersion_diag.py --apply
  grep -qF "$SENTINEL" "$RA" || { echo "FATAL: dispersion block absent after patch; aborting."; exit 1; }
fi
echo "dispersion capture OK"

say "[0b] voter fold train-season check (warns only)"
for m in MVP_monotone_2024 DPOY_monotone_2024 ROTY_vs_2024; do
  python3 - "$m" <<'PY'
import json, sys
m = sys.argv[1]
try:
    ts = json.load(open(f"models/folds/{m}.pkl.manifest.json")).get("train_seasons") or []
    print(m, "INCLUDES 2024 -> voter runs will SEAL-fail" if 2024 in ts else "ok (excludes 2024)")
except Exception as e:
    print(m, "manifest unreadable:", e)
PY
done

OV='{"MVP":"models/folds/MVP_monotone_2024.pkl","DPOY":"models/folds/DPOY_monotone_2024.pkl","ROTY":"models/folds/ROTY_vs_2024.pkl"}'
STAT='{"PTS":1000,"REB":1000,"AST":1000}'
VOTER='{"MVP":1000,"DPOY":1000,"ROTY":1000}'

# run_one TAG BAND(""|vol) "AWARDS" ALLOC_JSON OVERRIDE_JSON(""|json)
run_one () {
  local tag="$1" band="$2" awards="$3" alloc="$4" ov="$5"
  local diag="out/region_diag_2024_${tag}.csv"
  rm -f "$diag"
  local -a envv=(USE_REGION=1 REGION_CONFIRM=2 REGION_HYST=1.0 REGION_BAND_FLOOR=0.02 ASYM_TRIM=1
                 "ALLOC_BUDGETS_JSON=$alloc" "REGION_DIAG_PATH=$diag")
  if [ -n "$band" ]; then envv+=("REGION_BAND_MODE=$band"); fi
  if [ -n "$ov" ]; then envv+=("FINAL_MODEL_OVERRIDE_JSON=$ov"); fi
  echo ">> start $tag"
  env "${envv[@]}" uv run python3 -m scripts.backtest.engine.backtest_singlepass \
      --awards $awards --season 2024 --out "out/sp_2024_${tag}" > "$LOG/run_${tag}.log" 2>&1
  echo ">> done  $tag (exit $?)"
}

run_arm_stat () {
  run_one ctrl_stat ""  "PTS REB AST" "$STAT" ""
  run_one vol_stat  vol "PTS REB AST" "$STAT" ""
}
run_arm_voter () {
  run_one ctrl_voter ""  "MVP DPOY ROTY" "$VOTER" "$OV"
  run_one vol_voter  vol "MVP DPOY ROTY" "$VOTER" "$OV"
}

if [ "$PARALLEL" = "1" ]; then
  say "running the two arms in PARALLEL (control then reform within each arm)"
  run_arm_stat &
  SPID=$!
  run_arm_voter &
  VPID=$!
  wait "$SPID" "$VPID"
else
  say "running the four backtests SEQUENTIALLY"
  run_arm_stat
  run_arm_voter
fi

say "reports"
for tag in ctrl_stat vol_stat ctrl_voter vol_voter; do
  echo "----- report $tag -----"
  uv run python3 diag_report.py --diag "out/region_diag_2024_${tag}.csv" \
    --eval "out/sp_2024_${tag}/model_eval_2024.csv" 2>&1 | tee "$LOG/report_${tag}.txt"
done

say "SUMMARY: pooled and per-book PnL across the four runs"
for tag in ctrl_stat vol_stat ctrl_voter vol_voter; do
  echo "--- $tag ---"
  grep -hE "POOLED|PnL|SEAL" "$LOG/run_${tag}.log" 2>/dev/null || echo "  (no summary line; check $LOG/run_${tag}.log)"
done
echo
echo "all logs and reports under $LOG"
