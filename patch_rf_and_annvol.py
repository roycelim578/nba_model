"""Anchored idempotent patcher, two files, six blocks, one gate at the end.

1. risk_metrics.py
   a. sharpe()             reformulate to (annualised_return - rf) / annualised_vol,
                           rf now an ANNUAL rate defaulting to 0.03. The old body
                           subtracted rf from a PER-STEP mean, so an annual 3%
                           there was dimensionally wrong; this makes rf=3% mean 3%.
   b. sortino()            same numerator; denominator is the annualised downside
                           deviation, downside measured against the de-annualised
                           rf as the per-step minimum acceptable return.
   c. report_pooled()      add rf=0.03 param, pass it to sharpe/sortino, add
                           annualised_vol_pct to the output, echo rf.

2. backtest_singlepass.py
   a. run_singlepass()     add rf=0.03 param
   b. report_pooled call   pass rf=rf through
   c. main()               add --rf CLI flag (default 0.03) and thread to run_singlepass

book_summary() in trade_ledger calls sharpe(eq)/sortino(eq) with no rf, so the
per-book figures inherit the 0.03 default automatically (the intended behaviour).

GATE: changes book_summary.json numbers and the printed POOLED MTM only. It does
NOT touch trade_log, position_log, equity_curve or model_eval, the four artefacts
gate.sh diffs, so the sealed 2025 gate stays green. The reported Sharpe/Sortino
in RESULTS.md will move (rf now 3%, formula now CAGR-based); re-baseline them off
the re-run, do not treat the shift as a regression.

Usage:
    python3 patch_rf_and_annvol.py            # dry run, shows every block
    python3 patch_rf_and_annvol.py --apply    # writes both files, .bak each
"""
from __future__ import annotations
import argparse
import shutil
import sys
from pathlib import Path

RISK = Path("scripts/common/risk_metrics.py")
SINGLEPASS = Path("scripts/backtest/engine/backtest_singlepass.py")

RISK_PATCHES = [
("sharpe(): reformulate to (annret - rf)/annvol, rf annual, default 3%",
'''def sharpe(equity, periods_per_year=252, rf=0.0):
    r = _pct_returns(equity)
    if r.size < 2:
        return 0.0
    sd = r.std(ddof=1)
    if sd <= 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * (r.mean() - rf) / sd)''',
'''def sharpe(equity, periods_per_year=252, rf=0.03):
    """Annualised Sharpe: (annualised return - rf) / annualised vol. rf is an
    ANNUAL rate (3% default), not a per-step rate."""
    av = annualised_vol(equity, periods_per_year)
    if av <= 0:
        return 0.0
    return float((annualised_return(equity, periods_per_year) - rf) / av)'''),

("sortino(): reformulate to (annret - rf)/annualised downside dev, rf annual",
'''def sortino(equity, periods_per_year=252, rf=0.0):
    r = _pct_returns(equity)
    if r.size < 2:
        return 0.0
    downside = np.minimum(r - rf, 0.0)
    dd = np.sqrt(np.mean(downside ** 2))
    if dd <= 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * (r.mean() - rf) / dd)''',
'''def sortino(equity, periods_per_year=252, rf=0.03):
    """Annualised Sortino: (annualised return - rf) / annualised downside
    deviation. rf is an ANNUAL rate (3% default); the downside threshold is the
    de-annualised per-step rf, so only sub-rf steps count as downside."""
    r = _pct_returns(equity)
    if r.size < 2:
        return 0.0
    mar = rf / periods_per_year
    downside = np.minimum(r - mar, 0.0)
    dd = np.sqrt(np.mean(downside ** 2))
    if dd <= 0:
        return 0.0
    dd_ann = np.sqrt(periods_per_year) * dd
    return float((annualised_return(equity, periods_per_year) - rf) / dd_ann)'''),

("report_pooled(): add rf param, annualised_vol_pct output, echo rf",
'''def report_pooled(ledgers, periods_per_year=252):
    """Return (metrics_dict, dates, pooled_equity)."""
    dates, eq = pooled_curve(ledgers)
    if eq.size == 0:
        return {"n": 0}, dates, eq
    metrics = dict(
        start=float(eq[0]),
        final=float(eq[-1]),
        total_return_pct=float(100.0 * (eq[-1] / eq[0] - 1.0)) if eq[0] else 0.0,
        annualised_return_pct=float(100.0 * annualised_return(eq, periods_per_year)),
        sharpe=sharpe(eq, periods_per_year),
        sortino=sortino(eq, periods_per_year),
        calmar=calmar(eq, periods_per_year),
        max_drawdown_pct=float(100.0 * max_drawdown(eq)),
        diversification_ratio=diversification_ratio(ledgers, periods_per_year),
        n=int(eq.size),
    )
    return metrics, dates, eq''',
'''def report_pooled(ledgers, periods_per_year=252, rf=0.03):
    """Return (metrics_dict, dates, pooled_equity). rf is an ANNUAL risk-free
    rate (3% default) feeding Sharpe and Sortino."""
    dates, eq = pooled_curve(ledgers)
    if eq.size == 0:
        return {"n": 0}, dates, eq
    metrics = dict(
        start=float(eq[0]),
        final=float(eq[-1]),
        total_return_pct=float(100.0 * (eq[-1] / eq[0] - 1.0)) if eq[0] else 0.0,
        annualised_return_pct=float(100.0 * annualised_return(eq, periods_per_year)),
        annualised_vol_pct=float(100.0 * annualised_vol(eq, periods_per_year)),
        sharpe=sharpe(eq, periods_per_year, rf=rf),
        sortino=sortino(eq, periods_per_year, rf=rf),
        calmar=calmar(eq, periods_per_year),
        max_drawdown_pct=float(100.0 * max_drawdown(eq)),
        diversification_ratio=diversification_ratio(ledgers, periods_per_year),
        n=int(eq.size),
        rf=float(rf),
    )
    return metrics, dates, eq'''),
]

SINGLEPASS_PATCHES = [
("run_singlepass(): add rf param",
'''def run_singlepass(season, awards=DEFAULT_AWARDS, budget_per_book=1000.0,
                   use_stub=False, verbose=False, compound=False):''',
'''def run_singlepass(season, awards=DEFAULT_AWARDS, budget_per_book=1000.0,
                   use_stub=False, verbose=False, compound=False, rf=0.03):'''),

("run_singlepass(): pass rf into report_pooled",
'''    from scripts.common.risk_metrics import report_pooled
    _mtm, _mtm_dates, _mtm_eq = report_pooled([r["ledger"] for r in results.values()])''',
'''    from scripts.common.risk_metrics import report_pooled
    _mtm, _mtm_dates, _mtm_eq = report_pooled([r["ledger"] for r in results.values()], rf=rf)'''),

("main(): add --rf flag",
'''    ap.add_argument("--compound", action="store_true")
    args = ap.parse_args()''',
'''    ap.add_argument("--compound", action="store_true")
    ap.add_argument("--rf", type=float, default=0.03,
                    help="annual risk-free rate for Sharpe/Sortino (default 0.03)")
    args = ap.parse_args()'''),

("main(): thread --rf into run_singlepass",
'''    results, pooled = run_singlepass(
        args.season, awards=tuple(args.awards), budget_per_book=args.budget,
        use_stub=args.stub_cost, compound=compound)''',
'''    results, pooled = run_singlepass(
        args.season, awards=tuple(args.awards), budget_per_book=args.budget,
        use_stub=args.stub_cost, compound=compound, rf=args.rf)'''),
]

FILES = [
    (RISK, RISK_PATCHES),
    (SINGLEPASS, SINGLEPASS_PATCHES),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    all_ok = True
    staged = []
    for path, patches in FILES:
        if not path.exists():
            print(f"NOT FOUND: {path} (run from repo root)")
            all_ok = False
            continue
        text = path.read_text()
        for label, old, new in patches:
            count = text.count(old)
            if count != 1:
                print(f"[{path}] '{label}': anchor matched {count} times, expected 1")
                all_ok = False
                continue
            text = text.replace(old, new)
            if not args.apply:
                print(f"[{path}] OK: '{label}'")
        staged.append((path, text))

    if not all_ok:
        sys.exit("\naborting: one or more anchors did not match cleanly, nothing written")

    if not args.apply:
        print("\nDRY RUN only, all anchors matched. Re-run with --apply to write.")
        return

    for path, text in staged:
        backup = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, backup)
        path.write_text(text)
        print(f"applied: {path} (backup at {backup})")


if __name__ == "__main__":
    main()
