"""Fill-fraction sizer glue (master spec 6, sizer wiring).

The outcome-cloud portfolio-Kelly owns bet MAGNITUDE and the within-race anti-correlation
(one winner per joint draw). This module scales that target by a single fill fraction in
[0, 1] and gates it. The design discipline: the two variances enter through two different
doors exactly once each. Outcome variance is already in the Kelly magnitude. Price variance
enters ONLY here, via the fill fraction keyed off price_dispersion, so it is never
double-counted. The fill fraction IS the effective Kelly fraction (fill=1 is full Kelly,
fill=0.5 half); there is no separate damping coefficient on top.

Two gates, both snap the position to zero:
  edge-quality gate  risk_adjusted_edge (composite, both variances) below hurdle: skip.
  fill-floor gate    fill below min_fill: too fluid / too small to be worth transacting.

The fill CEILING is fill_ceiling (default 1.0, full Kelly at zero price dispersion); it is a
gridded parameter, and the 2024 backtest is allowed to lower it if realised drawdown is
ugly, because full Kelly is fragile to the probabilities being wrong and our validation is
thin. The functional form mapping price_dispersion to fill is likewise gridded, not asserted.
"""

import numpy as np


def fill_fraction(price_sigma, form="inverse", k=3.2, sigma_ref=0.08, ceiling=1.0):
    """Fill fraction in [0, ceiling], decreasing in price_sigma. Pinned v1: form='inverse',
    k=3.2 (corpus-calibrated so median dispersion ~0.13 maps to ~0.70 fill, spanning ~0.82
    to ~0.62 across the 10th-90th dispersion percentile). Gridded alternate: form='exp',
    k=2.66 (near-identical spread). Forms:
      inverse  ceiling / (1 + k*sigma)      smooth, always positive
      exp      ceiling * exp(-k*sigma)       smooth, always positive
      linear   ceiling * (1 - k*sigma)       reaches zero (a natural fill-floor)
      ref      ceiling * sigma_ref/sigma     full up to a reference dispersion, shrinks above
    """
    s = float(price_sigma)
    if form == "inverse":
        f = 1.0 / (1.0 + k * s)
    elif form == "exp":
        f = np.exp(-k * s)
    elif form == "linear":
        f = 1.0 - k * s
    elif form == "ref":
        f = sigma_ref / max(s, 1e-9)
    else:
        raise ValueError(f"unknown fill form {form!r}")
    return float(np.clip(ceiling * f, 0.0, ceiling))


def size_positions(kelly_targets, risk_adjusted_edges, price_sigmas, hurdle=0.05,
                   min_fill=0.05, fill_form="inverse", fill_kwargs=None):
    """Apply gate and fill to signed outcome-Kelly targets.

    kelly_targets         signed dollars per candidate from the outcome-cloud Kelly solve.
    risk_adjusted_edges   composite radj per candidate (gate metric).
    price_sigmas          composite price dispersion per candidate (fill key).
    Returns (final_alloc, diagnostics-per-candidate).
    """
    fill_kwargs = {} if fill_kwargs is None else fill_kwargs
    targets = np.asarray(kelly_targets, dtype=float)
    radj = np.asarray(risk_adjusted_edges, dtype=float)
    sig = np.asarray(price_sigmas, dtype=float)
    out = np.zeros_like(targets)
    diag = []
    for i in range(targets.size):
        f = fill_fraction(sig[i], form=fill_form, **fill_kwargs)
        gated = radj[i] <= hurdle
        floored = f < min_fill
        take = (not gated) and (not floored)
        out[i] = targets[i] * f if take else 0.0
        diag.append(dict(cand=i, kelly_target=float(targets[i]), radj=float(radj[i]),
                         price_sigma=float(sig[i]), fill=f, gated_by_edge=gated,
                         gated_by_fill=floored, final=float(out[i])))
    return out, diag
