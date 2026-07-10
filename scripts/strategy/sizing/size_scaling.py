"""
Smooth size-scaling layer. Sits at the same architectural level as the forward-vol
predictor: it takes the sizer's RAW signed Kelly allocation and shrinks it by a product
of continuous multiplicative scalers, each in (0, 1]. No hard caps, no kinks, so every
handoff into the executed size stays a smooth fade.

Two scalers, composed multiplicatively:

  f_vol   PRICE VARIANCE. From the forward-vol price dispersion sigma_p (std of the
          survival-mixture future price). A position that is right but early sits at
          adverse prices for longer; higher dispersion shrinks it. Rational roll-off
          f_vol = 1 / (1 + (sigma_p / sigma_ref)^2). If dispersion is not supplied the
          caller has already applied the vol fill fraction, so this defaults to 1.

  f_conc  SETTLEMENT TAIL (per name). A NO on a longshot is a sold deep option: it loses
          the whole outlay if that one name wins. Because exactly one candidate wins the
          award, at most one NO in the book can lose at settlement, so the option-selling
          tail is a PER-NAME risk and a per-name smooth roll-off on the outlay share is
          the correct and sufficient control. share = |raw| / portfolio;
          f_conc = 1 / (1 + (share / s_soft)^2). At share = s_soft the size is halved;
          above it the position rolls off smoothly rather than being clipped.

final = raw * kelly_fraction * f_vol * f_conc. British English.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class ScaleParams:
    kelly_fraction: float = 0.5
    sigma_ref: float = 0.12
    s_soft_no: float = 0.12
    s_soft_yes: float = 0.30


@dataclass
class ScaledAllocation:
    scaled: np.ndarray
    f_vol: np.ndarray
    f_conc: np.ndarray
    f_total: np.ndarray
    player_ids: list


def _f_vol(price_dispersion, sigma_ref):
    if price_dispersion is None:
        return None
    sp = np.asarray(price_dispersion, dtype=float)
    return 1.0 / (1.0 + (sp / max(sigma_ref, 1e-9)) ** 2)


def _f_conc(raw_alloc, portfolio, s_soft_no, s_soft_yes):
    """Leg-aware concentration roll-off. NO legs (option-selling tail) get the firmer
    budget s_soft_no; YES legs (favourite conviction, well-estimated downside) get the
    generous s_soft_yes so Kelly's conviction on a backed favourite is tempered, not
    erased."""
    raw = np.asarray(raw_alloc, dtype=float)
    share = np.abs(raw) / max(portfolio, 1e-9)
    s = np.where(raw < 0, s_soft_no, s_soft_yes)
    return 1.0 / (1.0 + (share / np.maximum(s, 1e-9)) ** 2)


def scale_allocation(raw_alloc, portfolio, params: ScaleParams,
                     price_dispersion=None, player_ids=None):
    """Apply the smooth scalers to the raw signed allocation. Returns ScaledAllocation
    with the final size and every intermediate scaler exposed for audit."""
    raw = np.asarray(raw_alloc, dtype=float)
    ncand = raw.size
    fv = _f_vol(price_dispersion, params.sigma_ref)
    if fv is None:
        fv = np.ones(ncand)
    fc = _f_conc(raw, portfolio, params.s_soft_no, params.s_soft_yes)
    ftot = params.kelly_fraction * fv * fc
    scaled = raw * ftot
    scaled[np.abs(scaled) < 0.5] = 0.0
    return ScaledAllocation(
        scaled=scaled, f_vol=fv, f_conc=fc,
        f_total=ftot, player_ids=list(player_ids) if player_ids is not None else list(range(ncand)))
