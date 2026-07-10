#!/usr/bin/env bash
set -e
USE_REGION=1 REGION_CONFIRM=2 REGION_HYST=1.0 REGION_BAND_FLOOR=0.02 ASYM_TRIM=1 \
  \
  uv run python -m scripts.backtest.engine.backtest_singlepass --season 2025 --out out/sp_2025_shrunk >/tmp/gate.log 2>&1
tail -6 /tmp/gate.log
a=$(cat out/_golden_sp2025/*.csv 2>/dev/null | md5); b=$(cat out/sp_2025_shrunk/*.csv 2>/dev/null | md5)
[ "$a" = "$b" ] && echo "GATE PASS (bit-identical)" || echo "GATE FAIL (output changed)"
