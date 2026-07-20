"""Almgren-Chriss-flavoured daily execution schedule.

The engine trades a name at most once per day and the per-trade cost is convex in the
size pushed through at once (the cost curve carries a sqrt-of-size impact term), so a
large target change taken in one day pays more slippage than the same change spread over
days. This paces each name toward its target: fill a fraction phi of the remaining gap
today, the rest on later days.

phi follows the Almgren-Chriss trade-off between impact and price risk. Trading slowly
saves impact but leaves the unfilled gap exposed to price drift; trading fast avoids that
drift but pays impact. The optimal rate rises with volatility, so a high forward-vol name,
which is likely to move away from us before we are filled and is therefore expensive to
wait on, fills fast, while a low-vol name is spread to bank the convex cost saving.

Volatility is read relative to the day's cross-section (the median forward sigma), not an
absolute level, so nothing here is calibrated to the cost model's own parameters. A
settlement ramp drives phi to one as the season fraction approaches one, so every position
is fully in place by resolution whatever its vol. Because the rule paces the gap in both
directions it also bleeds a fallen target out gradually rather than dumping it, which is
where much of the realised path cost was going. British English.
"""
from __future__ import annotations

import numpy as np


def pace(target_by_pid, current_by_pid, psig_by_pid, frac, phi_base=0.4, phi_min=0.15):
    """Return a paced target per pid, current + phi * (target - current).

    phi = phi_vol + (1 - phi_vol) * frac, where phi_vol rises with the name's forward
    sigma relative to the day's median and is floored at phi_min. frac is the season
    fraction (0 early, 1 at resolution); the ramp guarantees a full fill by settlement.
    Names with no sigma supplied (held names the fresh solve did not re-quote) fall back
    to the day's median, so they pace at phi_base before the ramp."""
    sigs = [float(psig_by_pid[p]) for p in target_by_pid
            if p in psig_by_pid and np.isfinite(psig_by_pid[p]) and psig_by_pid[p] > 0]
    ref = float(np.median(sigs)) if sigs else 1.0
    frac = float(np.clip(frac, 0.0, 1.0))
    phi_base = float(phi_base)
    phi_min = float(phi_min)

    paced = {}
    for pid, tgt in target_by_pid.items():
        cur = float(current_by_pid.get(pid, 0.0))
        sig = float(psig_by_pid.get(pid, ref))
        if not np.isfinite(sig) or sig <= 0:
            sig = ref
        phi_vol = float(np.clip(phi_base * (sig / max(ref, 1e-9)), phi_min, 1.0))
        phi = phi_vol + (1.0 - phi_vol) * frac
        paced[pid] = cur + phi * (float(tgt) - cur)
    return paced
