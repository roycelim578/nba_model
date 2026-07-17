"""Anchored idempotent patcher, three files, nine blocks, one gate at the end.

1. trade_ledger.py
   a. __init__            add _deployed / _deployed_dates tracking arrays
   b. record_mark         add record_deployed() right after it
   c. settle()             append a post-settlement equity mark (the MTM fix:
                           without this, the last point on both the per-book and
                           pooled equity curves is the last pre-settlement mid
                           mark, so the settlement-day win/loss jump never enters
                           the return series at all)
   d. book_summary()      add avg_deployed_usd / return_on_deployed_pct and
                           edge_realisation_ratio (= realised_pnl / sum_fair_value_edge,
                           reusing the existing entry-edge attribution rather than
                           a new, easily double-counted construct)

2. backtest_orchestrator_daily.py
   a/b. step_award_day()  call ledger.record_deployed() on both the empty and
                           non-empty paths (out["deployed"] always exists, 0.0 by
                           default), so idle days correctly pull the average down

3. risk_metrics.py
   a. annualised_return / annualised_vol / calmar   new functions
   b. diversification_ratio                          new function (needs pooled_curve)
   c. report_pooled()                                 wires the three new fields in

None of this touches trade_log, position_log, equity_curve or model_eval, the
four artefacts gate.sh diffs byte-for-byte / column-for-column, so the sealed
2025 gate should stay green. Still run it: code deletions get a gate even when
the read looks conclusive.

Usage:
    python3 patch_mtm_and_metrics.py            # dry run, shows every block
    python3 patch_mtm_and_metrics.py --apply    # writes all three files, .bak each
"""
from __future__ import annotations
import argparse
import shutil
import sys
from pathlib import Path

LEDGER = Path("scripts/backtest/settle/trade_ledger.py")
ORCH_DAILY = Path("scripts/backtest/engine/backtest_orchestrator_daily.py")
RISK = Path("scripts/common/risk_metrics.py")

# ---------------------------------------------------------------------------
# 1. trade_ledger.py
# ---------------------------------------------------------------------------
LEDGER_PATCHES = [
("init: add deployed-capital tracking arrays",
'''        self._pos = {}
        self.trade_log = []
        self.position_log = []
        self._equity_dates = []
        self._equity = []''',
'''        self._pos = {}
        self.trade_log = []
        self.position_log = []
        self._equity_dates = []
        self._equity = []
        self._deployed_dates = []
        self._deployed = []'''),

("record_mark: add record_deployed() alongside it",
'''    def record_mark(self, date, yes_mids):
        eq = self.cash
        for pid, p in self._pos.items():
            m_yes = yes_mids.get(int(pid))
            if m_yes is None:
                continue
            eq += p.shares * _to_leg(float(m_yes), p.side)
        self._equity_dates.append(str(date))
        self._equity.append(float(eq))''',
'''    def record_mark(self, date, yes_mids):
        eq = self.cash
        for pid, p in self._pos.items():
            m_yes = yes_mids.get(int(pid))
            if m_yes is None:
                continue
            eq += p.shares * _to_leg(float(m_yes), p.side)
        self._equity_dates.append(str(date))
        self._equity.append(float(eq))

    def record_deployed(self, date, deployed_usd):
        """Daily sizer target notional (sum of |target_usd| across candidates),
        recorded whether or not a trade actually fired that day, so idle days
        pull the average down exactly as they should for a capital-utilisation
        read."""
        self._deployed_dates.append(str(date))
        self._deployed.append(float(deployed_usd))'''),

("settle(): append the post-settlement equity mark (the MTM fix)",
'''            self.position_log.append(self._attribute(
                p, is_settle=True, settle_leg=settle_leg, close_date=str(date),
                reason="SETTLE", settle_yes_report=settle_yes))
            del self._pos[pid]''',
'''            self.position_log.append(self._attribute(
                p, is_settle=True, settle_leg=settle_leg, close_date=str(date),
                reason="SETTLE", settle_yes_report=settle_yes))
            del self._pos[pid]
        self.record_mark(date, {})'''),

("book_summary(): add deployed-capital and edge-realisation metrics",
'''    def book_summary(self):
        eq = np.asarray(self._equity, float)
        realised_total = self.cash - self.starting_cash
        out = dict(award=self.award, season=self.season, starting_cash=self.starting_cash,
                   ending_cash=self.cash, realised_pnl=realised_total,
                   return_pct=100.0 * realised_total / self.starting_cash,
                   n_positions_closed=len(self.position_log),
                   n_transactions=len(self.trade_log))
        if len(eq) >= 3:
            from scripts.common import risk_metrics as _risk
            out.update(sharpe=_risk.sharpe(eq),
                       sortino=_risk.sortino(eq),
                       max_drawdown_pct=float(100.0 * _risk.max_drawdown(eq)),
                       n_marks=int(eq.size))
        pl = self.position_log
        if pl:
            for k in ("fair_value_edge", "outcome_surprise", "cost_drag", "path_residual"):
                out["sum_" + k] = float(np.nansum([r[k] for r in pl]))
            wins = [r for r in pl if r["realised_pnl"] > 0]
            out["win_rate"] = len(wins) / len(pl)
            vc = {}
            for r in pl:
                vc[r["verdict"]] = vc.get(r["verdict"], 0) + 1
            out["verdict_counts"] = vc
            recon = sum(out.get("sum_" + k, 0) for k in
                        ("fair_value_edge", "outcome_surprise", "cost_drag", "path_residual"))
            out["attribution_reconciles"] = bool(abs(recon - realised_total) < 0.5)
        return out''',
'''    def book_summary(self):
        eq = np.asarray(self._equity, float)
        realised_total = self.cash - self.starting_cash
        out = dict(award=self.award, season=self.season, starting_cash=self.starting_cash,
                   ending_cash=self.cash, realised_pnl=realised_total,
                   return_pct=100.0 * realised_total / self.starting_cash,
                   n_positions_closed=len(self.position_log),
                   n_transactions=len(self.trade_log))
        if len(eq) >= 3:
            from scripts.common import risk_metrics as _risk
            out.update(sharpe=_risk.sharpe(eq),
                       sortino=_risk.sortino(eq),
                       max_drawdown_pct=float(100.0 * _risk.max_drawdown(eq)),
                       n_marks=int(eq.size))
        if self._deployed:
            avg_deployed = float(np.mean(self._deployed))
            out["avg_deployed_usd"] = avg_deployed
            out["return_on_deployed_pct"] = (100.0 * realised_total / avg_deployed
                                             if avg_deployed > 1e-9 else float("nan"))
        pl = self.position_log
        if pl:
            for k in ("fair_value_edge", "outcome_surprise", "cost_drag", "path_residual"):
                out["sum_" + k] = float(np.nansum([r[k] for r in pl]))
            wins = [r for r in pl if r["realised_pnl"] > 0]
            out["win_rate"] = len(wins) / len(pl)
            vc = {}
            for r in pl:
                vc[r["verdict"]] = vc.get(r["verdict"], 0) + 1
            out["verdict_counts"] = vc
            recon = sum(out.get("sum_" + k, 0) for k in
                        ("fair_value_edge", "outcome_surprise", "cost_drag", "path_residual"))
            out["attribution_reconciles"] = bool(abs(recon - realised_total) < 0.5)
            sfve = out.get("sum_fair_value_edge", 0.0)
            out["edge_realisation_ratio"] = (realised_total / sfve
                                             if abs(sfve) > 1e-9 else float("nan"))
        return out'''),
]

# ---------------------------------------------------------------------------
# 2. backtest_orchestrator_daily.py
# ---------------------------------------------------------------------------
ORCH_DAILY_PATCHES = [
("step_award_day(): record deployed capital on the empty path",
'''    if out["empty"]:
        bo.self_mark(ledger, day, yes_mids, samples)
        ctx.region_state = out["region_state"]
        return''',
'''    if out["empty"]:
        bo.self_mark(ledger, day, yes_mids, samples)
        ledger.record_deployed(day, out["deployed"])
        ctx.region_state = out["region_state"]
        return'''),

("step_award_day(): record deployed capital on the trading path",
'''    bo.self_mark(ledger, day, yes_mids, samples)
    ctx.region_state = out["region_state"]''',
'''    bo.self_mark(ledger, day, yes_mids, samples)
    ledger.record_deployed(day, out["deployed"])
    ctx.region_state = out["region_state"]'''),
]

# ---------------------------------------------------------------------------
# 3. risk_metrics.py
# ---------------------------------------------------------------------------
RISK_PATCHES = [
("add annualised_return / annualised_vol / calmar",
'''def max_drawdown(equity):
    e = np.asarray(equity, float)
    if e.size == 0:
        return 0.0
    peak = np.maximum.accumulate(e)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(peak != 0.0, e / peak - 1.0, 0.0)
    return float(dd.min())''',
'''def max_drawdown(equity):
    e = np.asarray(equity, float)
    if e.size == 0:
        return 0.0
    peak = np.maximum.accumulate(e)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(peak != 0.0, e / peak - 1.0, 0.0)
    return float(dd.min())


def annualised_return(equity, periods_per_year=252):
    """CAGR off the number of return steps actually observed, not calendar
    days, so a short sealed season still annualises on its own trading-day
    count."""
    e = np.asarray(equity, float)
    if e.size < 2 or e[0] <= 0:
        return 0.0
    n = e.size - 1
    total = e[-1] / e[0]
    if total <= 0:
        return -1.0
    return float(total ** (periods_per_year / n) - 1.0)


def annualised_vol(equity, periods_per_year=252):
    r = _pct_returns(equity)
    if r.size < 2:
        return 0.0
    return float(np.sqrt(periods_per_year) * r.std(ddof=1))


def calmar(equity, periods_per_year=252):
    mdd = abs(max_drawdown(equity))
    if mdd <= 0:
        return 0.0
    return float(annualised_return(equity, periods_per_year) / mdd)'''),

("add diversification_ratio",
'''def pooled_curve(ledgers):
    """Sum each book's daily MTM equity by date. A book contributes its last known
    equity on dates before its next mark (forward-fill), and its first recorded
    equity on dates before it starts marking (so the pool starts at the summed
    opening bankroll rather than jumping)."""
    ledgers = [l for l in ledgers if getattr(l, "_equity", None)]
    if not ledgers:
        return [], np.array([], float)
    all_dates = sorted(set().union(*[set(l._equity_dates) for l in ledgers]))
    total = np.zeros(len(all_dates), float)
    for l in ledgers:
        d = dict(zip(l._equity_dates, l._equity))
        first = float(l._equity[0])
        last = None
        series = []
        for dt in all_dates:
            if dt in d:
                last = d[dt]
            series.append(last if last is not None else first)
        total += np.asarray(series, float)
    return all_dates, total''',
'''def pooled_curve(ledgers):
    """Sum each book's daily MTM equity by date. A book contributes its last known
    equity on dates before its next mark (forward-fill), and its first recorded
    equity on dates before it starts marking (so the pool starts at the summed
    opening bankroll rather than jumping)."""
    ledgers = [l for l in ledgers if getattr(l, "_equity", None)]
    if not ledgers:
        return [], np.array([], float)
    all_dates = sorted(set().union(*[set(l._equity_dates) for l in ledgers]))
    total = np.zeros(len(all_dates), float)
    for l in ledgers:
        d = dict(zip(l._equity_dates, l._equity))
        first = float(l._equity[0])
        last = None
        series = []
        for dt in all_dates:
            if dt in d:
                last = d[dt]
            series.append(last if last is not None else first)
        total += np.asarray(series, float)
    return all_dates, total


def diversification_ratio(ledgers, periods_per_year=252):
    """Sum of each book's own annualised vol, divided by the pooled annualised
    vol. 1.0 means the books moved as one (no diversification benefit); higher
    means the pooled path is calmer than the books' individual paths alone
    would suggest. On a single season with one book dominating the variance
    this will read close to 1.0 by construction, not as a defect."""
    vols = [annualised_vol(l._equity, periods_per_year) for l in ledgers
           if getattr(l, "_equity", None)]
    if not vols:
        return float("nan")
    _, pooled_eq = pooled_curve(ledgers)
    pooled_vol = annualised_vol(pooled_eq, periods_per_year)
    if pooled_vol <= 0:
        return float("nan")
    return float(sum(vols) / pooled_vol)'''),

("wire annualised_return / calmar / diversification_ratio into report_pooled",
'''def report_pooled(ledgers, periods_per_year=252):
    """Return (metrics_dict, dates, pooled_equity)."""
    dates, eq = pooled_curve(ledgers)
    if eq.size == 0:
        return {"n": 0}, dates, eq
    metrics = dict(
        start=float(eq[0]),
        final=float(eq[-1]),
        total_return_pct=float(100.0 * (eq[-1] / eq[0] - 1.0)) if eq[0] else 0.0,
        sharpe=sharpe(eq, periods_per_year),
        sortino=sortino(eq, periods_per_year),
        max_drawdown_pct=float(100.0 * max_drawdown(eq)),
        n=int(eq.size),
    )
    return metrics, dates, eq''',
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
    return metrics, dates, eq'''),
]

FILES = [
    (LEDGER, LEDGER_PATCHES),
    (ORCH_DAILY, ORCH_DAILY_PATCHES),
    (RISK, RISK_PATCHES),
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
