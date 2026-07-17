"""Anchored idempotent patcher: link trade_ledger.book_summary()'s Sharpe/Sortino/
max-drawdown calc to scripts.common.risk_metrics instead of a separately maintained
formula, so per-book and pooled figures come from the same code and agree by
construction. Renames sharpe_per_step/sortino_per_step -> sharpe/sortino (no other
module reads those keys; the gate diffs trade_log/position_log/equity_curve/
model_eval, not book_summary.json).

Usage:
    python3 patch_link_risk_metrics.py            # dry run, shows the diff
    python3 patch_link_risk_metrics.py --apply    # writes the change, keeps a .bak
"""
from __future__ import annotations
import argparse
import shutil
import sys
from pathlib import Path

TARGET = Path("scripts/backtest/settle/trade_ledger.py")

OLD = '''        if len(eq) >= 3:
            rets = np.diff(eq) / eq[:-1]
            rets = rets[np.isfinite(rets)]
            mu = rets.mean()
            sd = rets.std(ddof=1) if rets.size > 1 else float("nan")
            downside = rets[rets < 0]
            dsd = downside.std(ddof=1) if downside.size > 1 else float("nan")
            peak = np.maximum.accumulate(eq)
            dd = (eq - peak) / peak
            out.update(sharpe_per_step=float(mu / sd) if sd and np.isfinite(sd) and sd > 0 else float("nan"),
                       sortino_per_step=float(mu / dsd) if dsd and np.isfinite(dsd) and dsd > 0 else float("nan"),
                       max_drawdown_pct=float(100.0 * dd.min()),
                       n_marks=int(eq.size))'''

NEW = '''        if len(eq) >= 3:
            from scripts.common import risk_metrics as _risk
            out.update(sharpe=_risk.sharpe(eq),
                       sortino=_risk.sortino(eq),
                       max_drawdown_pct=float(100.0 * _risk.max_drawdown(eq)),
                       n_marks=int(eq.size))'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    if not TARGET.exists():
        sys.exit(f"not found: {TARGET} (run from repo root)")
    text = TARGET.read_text()
    count = text.count(OLD)
    if count == 0:
        sys.exit("anchor not found (already applied, or file has drifted); aborting")
    if count > 1:
        sys.exit(f"anchor matched {count} times, expected exactly 1; aborting")

    patched = text.replace(OLD, NEW)

    if not args.apply:
        print(f"DRY RUN: would replace 1 block in {TARGET}\n")
        print("--- old ---")
        print(OLD)
        print("\n--- new ---")
        print(NEW)
        print("\nre-run with --apply to write")
        return

    backup = TARGET.with_suffix(TARGET.suffix + ".bak")
    shutil.copy2(TARGET, backup)
    TARGET.write_text(patched)
    print(f"applied. backup at {backup}")


if __name__ == "__main__":
    main()
