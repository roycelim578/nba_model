"""Portfolio-level allocation for the multi-book voter/stat engine.

Three pure primitives used by backtest_portfolio, all independent of the per-book
solve so they can be reasoned about and unit-tested in isolation. Re-pricing of risk
lives only in the per-book Kelly objective and in the tilt below; the caps only ever
clip, so nothing here double-counts what the objective and the tilt have priced.

within_arm_tilts
    A relative, mean-one skill tilt applied per arm. Books are grouped by arm (voter,
    stat); within each arm the raw skill vector is floored at zero, shrunk halfway to
    the arm mean and normalised so the arm's tilts average one. A book at the arm-mean
    skill gets tilt 1.0 and is untouched; better books tilt above one, worse below,
    and mean deployment within the arm is preserved. This is a weight, not a haircut.
    The two arms are tilted separately because their skill metrics are not on a common
    scale; the cross-arm split is the job of the type cap, not this tilt.

type_ceilings
    A breadth-weighted partition of equity across the arms present. Each arm gets a raw
    weight b/(b+c) in its validated book count b, normalised across present arms. More
    validated books in an arm earns it a larger ceiling, with diminishing returns; and
    the per-arm ceiling falls as arms are added, because a fixed pool is split among
    more claimants. This is the model-risk hedge: it caps how far the portfolio can
    lever into any single modelling stack.

apply_structural_caps
    Three structural ceilings then the joint capital constraint, each a clamp that only
    reduces a leg's absolute size. In order: player (A, covariance-agnostic, ceiling
    base * n**alpha of equity, n the number of books the player anchors), award (B, a
    per-book ceiling as a fraction of equity), type (C, the partition above), then the
    joint constraint (total deployment <= equity). Every binding clamp trims its group
    from the lowest risk-adjusted edge upward, so scarce capital is retained on the
    highest-radj legs.

British English, no em dashes.
"""
from __future__ import annotations

import numpy as np


def within_arm_tilts(skill_by_book, arm_by_book, shrink=0.5):
    """Return {book: tilt}, mean-one within each arm. skill_by_book maps book -> raw
    skill (any scale, negatives allowed, missing treated as zero after the caller has
    neutralised absent books); arm_by_book maps book -> arm label. shrink is the
    fraction of the raw signal kept (0.5 = halfway to the arm mean)."""
    arms = {}
    for b, a in arm_by_book.items():
        arms.setdefault(a, []).append(b)
    tilts = {}
    for a, books in arms.items():
        s = np.array([max(0.0, float(skill_by_book.get(b, 0.0))) for b in books], float)
        m = float(s.mean()) if s.size else 0.0
        if m <= 0.0:
            for b in books:
                tilts[b] = 1.0
            continue
        shrunk = shrink * s + (1.0 - shrink) * m
        shrunk = shrunk / float(shrunk.mean())    # mean-one within the arm
        for b, t in zip(books, shrunk):
            tilts[b] = float(t)
    return tilts


def type_ceilings(arms_present, validated_by_arm, c, equity):
    """Breadth-weighted partition of `equity` across arms_present. Weight b/(b+c) in the
    validated book count b, normalised across the arms present, scaled by equity."""
    w = {}
    for a in arms_present:
        b = float(validated_by_arm.get(a, 0.0))
        w[a] = b / (b + float(c)) if (b + float(c)) > 0 else 0.0
    tot = sum(w.values()) or 1.0
    return {a: (w[a] / tot) * float(equity) for a in arms_present}


def _trim_group(legs, ceiling):
    """Reduce the summed absolute target of `legs` to at most `ceiling`, trimming from
    the lowest radj upward. Mutates leg['target'] in place. NaN radj is lowest priority
    (trimmed first). A ceiling <= 0 zeroes the group."""
    total = sum(abs(l["target"]) for l in legs)
    excess = total - max(0.0, ceiling)
    if excess <= 1e-9:
        return
    order = sorted(legs, key=lambda l: (l["radj"] if l["radj"] == l["radj"] else -np.inf))
    for l in order:
        if excess <= 1e-9:
            break
        cut = min(abs(l["target"]), excess)
        if l["target"] != 0.0:
            l["target"] -= float(np.sign(l["target"])) * cut
        excess -= cut


def apply_structural_caps(legs, equity, player_base, player_alpha, award_frac,
                          type_ceiling_by_arm):
    """Clamp a flat list of legs in place and return it. Each leg is a mutable dict with
    keys book, pid, arm, target (signed dollars) and radj. Caps applied in the order
    A player, B award, C type, then the joint capital constraint. Absolute sizes only
    shrink, and each binding cap trims its group from the lowest radj upward."""
    # A. Player cap: n is the count of distinct books this player has a live leg in.
    by_pid = {}
    for l in legs:
        by_pid.setdefault(l["pid"], []).append(l)
    for _pid, pl in by_pid.items():
        n = len({l["book"] for l in pl})
        ceil_p = float(player_base) * (n ** float(player_alpha)) * float(equity)
        _trim_group(pl, ceil_p)

    # B. Award (per-book) cap.
    by_book = {}
    for l in legs:
        by_book.setdefault(l["book"], []).append(l)
    for _book, bl in by_book.items():
        _trim_group(bl, float(award_frac) * float(equity))

    # C. Type (per-arm) cap.
    by_arm = {}
    for l in legs:
        by_arm.setdefault(l["arm"], []).append(l)
    for arm, al in by_arm.items():
        _trim_group(al, float(type_ceiling_by_arm.get(arm, equity)))

    # Joint capital constraint across the whole portfolio.
    _trim_group(legs, float(equity))
    return legs
