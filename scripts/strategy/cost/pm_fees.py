"""Polymarket taker-fee module. Exact protocol formula, taker-only.

    fee_usdc = shares * fee_rate * p * (1 - p)

where shares = notional / p. Makers are never charged; every fill in this
engine is a taker fill (we cross the spread), so the taker fee always applies.

The fee in USDC is symmetric and peaks at p=0.5. As a FRACTION of notional it is
fee_rate * (1 - p), which is largest as p -> 0 (cheap longshot tickets), so the
proportional fee burden tilts toward the tail, compounding the spread cost there.
This is the OPPOSITE price profile to the spread bowl (which also peaks in the
tail in proportional terms) but the SAME tail-tilt once both are proportional.

Rates are the protocol feeRate coefficients (the p*(1-p) curve shape is identical
across categories; the rate just scales it). NBA award markets default to Sports
(0.03); confirm the actual category per market via getClobMarketInfo(conditionID)
since "Other/General" (0.05) is possible.
"""

from __future__ import annotations

# feeRate coefficients by category (from Polymarket fee docs).
FEE_RATE_BY_CATEGORY = {
    "crypto": 0.07,
    "sports": 0.03,
    "finance": 0.04,
    "politics": 0.04,
    "economics": 0.05,
    "culture": 0.05,
    "weather": 0.05,
    "other": 0.05,
    "mentions": 0.04,
    "tech": 0.04,
    "geopolitics": 0.0,
}

DEFAULT_CATEGORY = "sports"
MIN_FEE_USDC = 0.00001  # protocol rounds to 5 dp; smaller rounds to zero.


def fee_rate(category: str | None = None) -> float:
    return FEE_RATE_BY_CATEGORY.get((category or DEFAULT_CATEGORY).lower(),
                                    FEE_RATE_BY_CATEGORY[DEFAULT_CATEGORY])


def taker_fee_usdc(notional_usd: float, price: float, category: str | None = None) -> float:
    """Taker fee in USDC for a fill of `notional_usd` at `price`.

    shares = notional / price; fee = shares * rate * p * (1 - p)
           = notional * rate * (1 - p).
    """
    if price <= 0 or price >= 1 or notional_usd <= 0:
        return 0.0
    r = fee_rate(category)
    fee = notional_usd * r * (1.0 - price)
    fee = round(fee, 5)
    return fee if fee >= MIN_FEE_USDC else 0.0


def taker_fee_frac(price: float, category: str | None = None) -> float:
    """Taker fee as a FRACTION of notional: rate * (1 - p)."""
    if price <= 0 or price >= 1:
        return 0.0
    return fee_rate(category) * (1.0 - price)
