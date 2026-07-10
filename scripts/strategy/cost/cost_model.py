"""The cost function the backtest trades against. Loads frozen params from
cost_fit; pure function of (size, price, award, frac_to_resolution, knobs).

NO model references. The cost is endogenous to size (for downstream fixed-point
sizing) but blind to anything the model predicts.

Three modes:
  mode='none'   : zero cost. Tests raw predictive edge (flatters the tail).
  mode='spread' : bowl only, zero-size spread crossing. f(price) * tier_shift.
  mode='impact' : bowl * g(size, tier, frac). Full size-and-liquidity-aware cost.

Charged on BOTH entry and exit; exit on a thin tier carries a steeper effective
impact (the penny-cliff: exiting size walks off the near-zero-price wall).

All costs are PROPORTIONAL (fraction of traded notional). Currency USD.

Tier and frac_to_resolution are IMPOSED regime inputs (set + swept by the
backtest), not measured per market. Defaults force the caller to be explicit.
"""

from __future__ import annotations

import json
from bisect import bisect_left
from pathlib import Path

PARAMS_PATH = Path("data/cost_params.json")

# Liquidity regime multipliers on the near-touch scale. illiquid << liquid.
# These are TOGGLES the backtest sets per run; the cost function does NOT assign
# a regime to any award. The backtest reports the surface across regimes.
_REGIME_SCALE_MULT = {"illiquid": 0.25, "normal": 1.0, "liquid": 4.0}

# Exit-leg impact steepening by regime (penny-cliff asymmetry; measured
# structure, not an imposed prior). Illiquid exits walk off the near-zero wall.
_EXIT_STEEPEN = {"illiquid": 1.6, "normal": 1.2, "liquid": 1.0}


class CostModel:
    def __init__(self, params: dict):
        self._p = params
        b = params["bowl"]
        self._knot_p = b["knot_price"]
        self._knot_y = b["knot_rel_half_spread"]
        self._floor = b["favourite_floor"]
        self._cap = b["tail_cap"]
        self._crossover = b["crossover_price"]
        self._nt_by_bucket = params["impact"]["near_touch_notional_by_price_bucket_usd"]
        self._c_central = params["impact_coefficient_c"]["central"]

    @classmethod
    def load(cls, path: Path = PARAMS_PATH) -> "CostModel":
        return cls(json.loads(Path(path).read_text()))

    def _bowl(self, price: float) -> float:
        """Proportional half-spread at this price, monotone, floored and capped."""
        if price <= 0:
            return self._cap
        if price < self._crossover:
            return self._cap  # tail hands off to cliff regime; flat at cap
        kp, ky = self._knot_p, self._knot_y
        if price <= kp[0]:
            val = ky[0]
        elif price >= kp[-1]:
            val = ky[-1]
        else:
            i = bisect_left(kp, price)
            x0, x1 = kp[i - 1], kp[i]
            y0, y1 = ky[i - 1], ky[i]
            val = y0 + (y1 - y0) * (price - x0) / (x1 - x0) if x1 > x0 else y0
        return max(self._floor, min(self._cap, val))

    def _bucket(self, price: float) -> str:
        return ("00-02" if price < 0.02 else "02-05" if price < 0.05 else "05-10"
                if price < 0.10 else "10-25" if price < 0.25 else "25-50"
                if price < 0.50 else "50+")

    def _near_touch_scale(self, price: float, regime: str, frac_to_resolution: float,
                          depth_decay: float) -> float:
        base = self._nt_by_bucket.get(self._bucket(price)) or 1.0
        base = max(base, 1.0)
        base *= _REGIME_SCALE_MULT[regime]
        base *= max(0.05, 1.0 - depth_decay * frac_to_resolution)
        return base

    def cost_frac(self, *, size_usd: float, price: float, mode: str,
                  liquidity_regime: str = "normal", frac_to_resolution: float = 0.0,
                  c: float | None = None, depth_decay: float = 0.6,
                  leg: str = "entry") -> float:
        """Proportional cost (fraction of notional) for one leg of a trade.

        mode: 'none' | 'spread' | 'impact'.
        liquidity_regime: 'illiquid' | 'normal' | 'liquid' (TOGGLE, default normal).
        frac_to_resolution: 0 at resolution (time layer INERT), 1 at season start.
          The time-decay layer only bites when this is > 0, so it is off by
          default. Only meaningful in impact mode.
        leg: 'entry' or 'exit' (exit steeper on illiquid).
        """
        if mode == "none":
            return 0.0
        spread = self._bowl(price)
        if mode == "spread":
            return spread
        if mode != "impact":
            raise ValueError(f"unknown mode {mode!r}")
        c = self._c_central if c is None else c
        scale = self._near_touch_scale(price, liquidity_regime, frac_to_resolution, depth_decay)
        g = 1.0 + c * (max(size_usd, 0.0) / scale) ** 0.5
        if leg == "exit":
            g = 1.0 + (g - 1.0) * _EXIT_STEEPEN[liquidity_regime]
        return spread * g

    def round_trip_frac(self, *, size_usd: float, price: float, mode: str,
                        liquidity_regime: str = "normal", frac_to_resolution: float = 0.0,
                        c: float | None = None, depth_decay: float = 0.6) -> float:
        entry = self.cost_frac(size_usd=size_usd, price=price, mode=mode,
                               liquidity_regime=liquidity_regime,
                               frac_to_resolution=frac_to_resolution, c=c,
                               depth_decay=depth_decay, leg="entry")
        exit_ = self.cost_frac(size_usd=size_usd, price=price, mode=mode,
                               liquidity_regime=liquidity_regime,
                               frac_to_resolution=frac_to_resolution, c=c,
                               depth_decay=depth_decay, leg="exit")
        return entry + exit_
