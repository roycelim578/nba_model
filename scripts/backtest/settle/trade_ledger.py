"""
Trade ledger, per-player PnL attribution, and model-vs-strategy verdict layer.

Sits ABOVE the net-exposure engine. The orchestrator executes each transaction
through the real cost layer and RECORDS the executed fill here. Pure accounting,
DB-independent, self-testable.

One TradeLedger is ONE award-book with ONE bankroll. Equal-fixed-budget v1 makes
three independent books; pooled reporting concatenates the logs.

CASH-FLOW ACCOUNTING (reconciliation invariant)
  Every position tracks cash_in (spent on buys) and cash_out (received on trims +
  settlement). realised = cash_out - cash_in. Summed over all closed positions
  this EQUALS the book's cash PnL (ending_cash - starting_cash) exactly, because
  each is the running sum of the same cash deltas. Partial trims bank cash at their
  exit price into cash_out, so a position trimmed at prices different from its final
  close no longer loses that P&L from attribution (the earlier bug).

ATTRIBUTION (leg basis; tb = total shares bought, e_buy = cash_in/tb entry effective
  VWAP, m_buy = raw-mid entry VWAP, f = model terminal fair value on the leg,
  settle_shares held to resolution at settle_leg, trim_shares trimmed at avg_exit)
    fair_value_edge  = tb * (f - m_buy)                  was the edge real at entry
    cost_drag        = tb * (m_buy - e_buy) - exit_cost  entry cost + exit cost (both legs)
    outcome_surprise = settle_shares * (settle_leg - f)  model right/wrong on held portion
    path_residual    = trim_shares * (avg_exit_mid - f)  pure mid-convergence, cost-free
  where exit_cost = trim_proceeds_mid - trim_proceeds (dollars given up to spread/impact
  on the way out). All costs now live in cost_drag; path_residual is measured at the raw
  exit mid, so it is pure in-season convergence with no cost contamination.
  These sum to realised by construction. Held-to-resolution -> trim_shares 0 ->
  path 0, outcome_surprise is the clean model error; fully trimmed -> settle_shares
  0 -> outcome_surprise 0, path carries the convergence/timing P&L.

British English.
"""
from __future__ import annotations
from dataclasses import dataclass, field, replace
import numpy as np


@dataclass
class _Position:
    player_id: int
    side: str
    name: str = ""
    shares: float = 0.0
    total_bought: float = 0.0
    cash_in: float = 0.0
    outlay_mid_bought: float = 0.0
    fv_yes_entry_wsum: float = 0.0
    trim_shares: float = 0.0
    trim_proceeds: float = 0.0
    trim_proceeds_mid: float = 0.0
    fv_yes_terminal: float = float("nan")
    pwin_leg_terminal: float = float("nan")
    cloud_lo_yes: float = float("nan")
    cloud_hi_yes: float = float("nan")
    n_fills: int = 0
    first_date: str = ""
    last_trade_date: str = ""

    @property
    def vwap_eff(self):
        return self.cash_in / self.total_bought if self.total_bought > 1e-12 else float("nan")

    @property
    def vwap_mid(self):
        return self.outlay_mid_bought / self.total_bought if self.total_bought > 1e-12 else float("nan")

    @property
    def fv_yes_entry(self):
        return self.fv_yes_entry_wsum / self.total_bought if self.total_bought > 1e-12 else float("nan")

    @property
    def outlay_eff(self):
        return self.shares * self.vwap_eff if self.total_bought > 1e-12 else 0.0


def _to_leg(x_yes, side):
    return x_yes if side == "YES" else 1.0 - x_yes


class TradeLedger:
    def __init__(self, award, season, starting_cash, names=None):
        self.award = award
        self.season = int(season)
        self.starting_cash = float(starting_cash)
        self.cash = float(starting_cash)
        self.names = {int(k): v for k, v in (names or {}).items()}
        self._pos = {}
        self.trade_log = []
        self.position_log = []
        self._equity_dates = []
        self._equity = []
        self._deployed_dates = []
        self._deployed = []

    def _name(self, pid):
        return self.names.get(int(pid), str(int(pid)))

    def set_model_context(self, player_id, fv_yes_terminal=None, pwin_leg_terminal=None,
                          cloud_lo_yes=None, cloud_hi_yes=None):
        p = self._pos.get(int(player_id))
        if p is None:
            return
        if fv_yes_terminal is not None:
            p.fv_yes_terminal = float(fv_yes_terminal)
        if pwin_leg_terminal is not None:
            p.pwin_leg_terminal = float(pwin_leg_terminal)
        if cloud_lo_yes is not None:
            p.cloud_lo_yes = float(cloud_lo_yes)
        if cloud_hi_yes is not None:
            p.cloud_hi_yes = float(cloud_hi_yes)

    def record_trade(self, date, player_id, side, shares_delta, eff_price, mid_leg,
                     fv_yes=None, action=None, close_reason="REBALANCE_CLOSE"):
        """shares_delta > 0 buys (eff_price is the cost-inclusive entry price), < 0
        sells/trims (eff_price is the exit proceeds price). A full close attributes."""
        player_id = int(player_id)
        side = side.upper()
        shares_delta = float(shares_delta)
        eff_price = float(eff_price)
        mid_leg = float(mid_leg)
        if abs(shares_delta) < 1e-12:
            return

        p = self._pos.get(player_id)
        if p is not None and p.side != side and p.shares > 1e-12 and shares_delta > 0:
            raise ValueError(
                f"side flip on open position {player_id}: hold {p.side}, new {side}; "
                "close the existing leg before opening the opposite")
        if p is None:
            p = _Position(player_id=player_id, side=side, name=self._name(player_id),
                          first_date=str(date))
            self._pos[player_id] = p

        cash_delta = -shares_delta * eff_price
        self.cash += cash_delta

        if shares_delta > 0:
            p.shares += shares_delta
            p.total_bought += shares_delta
            p.cash_in += shares_delta * eff_price
            p.outlay_mid_bought += shares_delta * mid_leg
            if fv_yes is not None:
                p.fv_yes_entry_wsum += shares_delta * float(fv_yes)
        else:
            reduce = min(-shares_delta, p.shares)
            p.shares -= reduce
            p.trim_shares += reduce
            p.trim_proceeds += reduce * eff_price
            p.trim_proceeds_mid += reduce * mid_leg

        p.n_fills += 1
        p.last_trade_date = str(date)
        cost_usd = abs(shares_delta) * abs(eff_price - mid_leg)
        self.trade_log.append(dict(
            date=str(date), award=self.award, season=self.season, player_id=player_id,
            name=self._name(player_id),
            action=(action or ("ENTRY" if p.n_fills == 1 else "REBALANCE")),
            side=side, shares_delta=shares_delta, eff_price=eff_price, mid_leg=mid_leg,
            cost_usd=cost_usd, cash_delta=cash_delta, shares_after=p.shares,
            cash_after=self.cash))
        if p.shares <= 1e-9:
            self.position_log.append(self._attribute(
                p, is_settle=False, settle_leg=None, close_date=str(date),
                reason=close_reason, settle_yes_report=float("nan")))
            del self._pos[player_id]

    def record_mark(self, date, yes_mids):
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
        self._deployed.append(float(deployed_usd))

    def force_close(self, date, player_id, exit_eff_price, exit_mid_leg, reason="EXIT"):
        """Fully liquidate before resolution. Routes through the sell path so close
        attribution and cash both update."""
        pid = int(player_id)
        p = self._pos.get(pid)
        if p is None:
            return
        self.record_trade(date, pid, p.side, -p.shares, float(exit_eff_price),
                          float(exit_mid_leg), action="EXIT", close_reason=reason)

    def settle(self, date, winner_player_id):
        winner = int(winner_player_id)
        for pid in list(self._pos.keys()):
            p = self._pos[pid]
            settle_yes = 1.0 if pid == winner else 0.0
            settle_leg = _to_leg(settle_yes, p.side)
            proceeds = p.shares * settle_leg
            self.cash += proceeds
            self.trade_log.append(dict(
                date=str(date), award=self.award, season=self.season, player_id=pid,
                name=self._name(pid), action="SETTLE", side=p.side, shares_delta=-p.shares,
                eff_price=settle_leg, mid_leg=settle_leg, cost_usd=0.0, cash_delta=proceeds,
                shares_after=0.0, cash_after=self.cash))
            self.position_log.append(self._attribute(
                p, is_settle=True, settle_leg=settle_leg, close_date=str(date),
                reason="SETTLE", settle_yes_report=settle_yes))
            del self._pos[pid]
        self.record_mark(date, {})

    def _attribute(self, p, is_settle, settle_leg, close_date, reason, settle_yes_report):
        tb = p.total_bought
        e = p.vwap_eff
        m = p.vwap_mid
        f_yes = p.fv_yes_terminal
        f_leg = _to_leg(f_yes, p.side) if np.isfinite(f_yes) else float("nan")

        settle_shares = p.shares if is_settle else 0.0
        settle_proceeds = settle_shares * settle_leg if is_settle else 0.0
        cash_out = p.trim_proceeds + settle_proceeds
        realised = cash_out - p.cash_in
        avg_exit = p.trim_proceeds / p.trim_shares if p.trim_shares > 1e-12 else float("nan")
        close_leg = settle_leg if is_settle else avg_exit

        if np.isfinite(f_leg):
            fair_value_edge = tb * (f_leg - m)
            exit_cost = p.trim_proceeds_mid - p.trim_proceeds
            cost_drag = tb * (m - e) - exit_cost
            outcome_surprise = settle_shares * (settle_leg - f_leg) if is_settle else 0.0
            path_residual = realised - fair_value_edge - cost_drag - outcome_surprise
        else:
            fair_value_edge = cost_drag = outcome_surprise = path_residual = float("nan")

        outcome_in_cloud = None
        if np.isfinite(settle_yes_report) and np.isfinite(p.cloud_lo_yes):
            outcome_in_cloud = bool(p.cloud_lo_yes <= settle_yes_report <= p.cloud_hi_yes)

        verdict = self._verdict(realised, fair_value_edge, outcome_surprise, cost_drag,
                                path_residual, p.pwin_leg_terminal, outcome_in_cloud,
                                close_leg, reason=reason)

        return dict(
            award=self.award, season=self.season, player_id=p.player_id, name=p.name,
            side=p.side, close_reason=reason, shares_bought=tb, shares_settled=settle_shares,
            shares_trimmed=p.trim_shares, vwap_eff=e, vwap_mid=m,
            entry_cost_frac=(e - m) / m if m > 1e-9 else float("nan"),
            fv_yes_entry=p.fv_yes_entry, fv_yes_terminal=f_yes, fv_leg_terminal=f_leg,
            pwin_leg_terminal=p.pwin_leg_terminal, settle_yes=settle_yes_report,
            close_leg=close_leg, avg_exit_price=avg_exit, outcome_in_cloud=outcome_in_cloud,
            n_fills=p.n_fills, first_date=p.first_date, close_date=close_date,
            realised_pnl=realised, ret_on_outlay=realised / p.cash_in if p.cash_in > 1e-9 else float("nan"),
            fair_value_edge=fair_value_edge, outcome_surprise=outcome_surprise,
            cost_drag=cost_drag, path_residual=path_residual, verdict=verdict)

    @staticmethod
    def _verdict(realised, fve, osur, cost, path, pwin_leg, in_cloud, close_leg,
                 model_conf_hi=0.55, model_conf_lo=0.30, reason="SETTLE"):
        if not np.isfinite(fve):
            return "UNATTRIBUTED"
        if fve <= 0:
            return "ENTRY_ERROR"
        if reason != "SETTLE":
            return "TRIM_WIN" if realised >= 0 else "TRIM_LOSS"
        if realised >= 0:
            if np.isfinite(pwin_leg) and pwin_leg < model_conf_lo:
                return "WIN_LOW_CONF"
            return "WIN_ON_THESIS"
        dominant_neg = min([("outcome_surprise", osur), ("cost_drag", cost),
                            ("path_residual", path)], key=lambda kv: kv[1])[0]
        if dominant_neg == "outcome_surprise":
            if np.isfinite(pwin_leg) and pwin_leg >= model_conf_hi:
                return "MODEL_MISS"
            if in_cloud is True:
                return "VARIANCE"
            if np.isfinite(pwin_leg) and pwin_leg < model_conf_lo:
                return "VARIANCE"
            return "MODEL_SOFT_MISS"
        return "STRATEGY_COST"

    def open_positions(self):
        return {pid: dict(side=p.side, shares=p.shares, vwap_eff=p.vwap_eff)
                for pid, p in self._pos.items()}

    def book_summary(self):
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
        return out
