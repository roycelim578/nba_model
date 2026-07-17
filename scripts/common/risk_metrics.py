"""Pooled portfolio risk metrics for the cross-award backtester.

The concentration penalty is meant to LOWER return versus a concentrate-into-the
-longshot baseline while IMPROVING risk-adjusted return, so PnL is the wrong axis
to judge it on. This module builds the pooled daily mark-to-market equity curve
by summing each book's marked equity by date, and reports Sharpe, Sortino and
maximum drawdown on it. Mark-to-market, not the at-cost sizing frame:
concentration risk is a path-volatility phenomenon and the at-cost frame holds
equity flat until settlement, which would hide exactly the volatility we want to
measure. Each book's daily MTM series is already recorded on the ledger
(_equity_dates / _equity via record_mark).

On a single season these numbers are one noisy draw and the top book dominates
the variance, so read the DIRECTION between alloc-on and alloc-off (vol down,
drawdown shallower, PnL down), not the magnitudes. British English.
"""
from __future__ import annotations
import numpy as np


def _pct_returns(equity):
    e = np.asarray(equity, float)
    if e.size < 2:
        return np.array([], float)
    prev = e[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(prev != 0.0, np.diff(e) / prev, 0.0)
    return r


def sharpe(equity, periods_per_year=252, rf=0.03):
    """Annualised Sharpe: (annualised return - rf) / annualised vol. rf is an
    ANNUAL rate (3% default), not a per-step rate."""
    av = annualised_vol(equity, periods_per_year)
    if av <= 0:
        return 0.0
    return round(float((annualised_return(equity, periods_per_year) - rf) / av), 2)


def sortino(equity, periods_per_year=252, rf=0.03):
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
    return round(float((annualised_return(equity, periods_per_year) - rf) / dd_ann), 2)


def max_drawdown(equity):
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
    return round(float(annualised_return(equity, periods_per_year) / mdd), 2)


def pooled_curve(ledgers):
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
    return round(float(sum(vols) / pooled_vol), 2)


def report_pooled(ledgers, periods_per_year=252, rf=0.03):
    """Return (metrics_dict, dates, pooled_equity). rf is an ANNUAL risk-free
    rate (3% default) feeding Sharpe and Sortino."""
    dates, eq = pooled_curve(ledgers)
    if eq.size == 0:
        return {"n": 0}, dates, eq
    metrics = dict(
        start=float(eq[0]),
        final=float(eq[-1]),
        total_return_pct=round(float(100.0 * (eq[-1] / eq[0] - 1.0)) if eq[0] else 0.0, 1),
        annualised_return_pct=round(float(100.0 * annualised_return(eq, periods_per_year)), 1),
        annualised_vol_pct=round(float(100.0 * annualised_vol(eq, periods_per_year)), 1),
        sharpe=sharpe(eq, periods_per_year, rf=rf),
        sortino=sortino(eq, periods_per_year, rf=rf),
        calmar=calmar(eq, periods_per_year),
        max_drawdown_pct=round(float(100.0 * max_drawdown(eq)), 1),
        diversification_ratio=diversification_ratio(ledgers, periods_per_year),
        n=int(eq.size),
        rf=float(rf),
    )
    return metrics, dates, eq