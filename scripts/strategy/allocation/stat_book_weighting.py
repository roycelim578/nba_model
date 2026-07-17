"""Within-arm capital split for the stat books by shrunk leaderboard Brier skill.

The stat-arm twin of book_weighting. Same shrinkage mechanism, different skill
input: leaderboard-anchored BSS rather than the voter climatology BSS, because the
stat thesis is beating the retail leaderboard anchor, so leaderboard-relative skill
is the right within-arm proxy for alpha. The two arms are never compared on this
number; it splits the stat sub-bankroll only, and the cross-arm split is handled
separately. British English.

Mechanism per book for target season T:
  1. Read the persisted per-season leaderboard-BSS (stat_bss_persist output).
  2. Trailing-mean over the `trailing` seasons strictly before T.
  3. Floor negatives to zero: a book below the leaderboard earns no capital.
  4. Normalise to sum 1, then shrink halfway to 1/N.
  5. Multiply by the stat sub-bankroll, round to whole dollars.

Walk-forward: a target season uses only seasons before it. Books absent from the
artefact (STL, BLK until their scorecard lands) raise, so a premature eight-book
call fails loudly rather than silently dropping a book.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

DEFAULT_JSON = "models/stat_leader/bss_by_season_pra.json"
DEFAULT_AWARDS = ("PTS", "REB", "AST")
TRAILING = 10
SHRINK_TO_EQUAL = 0.5


def _load(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def book_skill_trailing(award, season, path=DEFAULT_JSON, trailing=TRAILING):
    """Trailing-mean leaderboard-BSS for a book over the seasons before `season`."""
    d = _load(path)
    if award not in d:
        raise KeyError(f"{award} not in {path}; run stat_bss_persist for it first "
                       f"(STL/BLK need the STL/BLK-capable scorecard).")
    pairs = sorted((int(s), v) for s, v in d[award].items() if v == v)
    prior = [v for s, v in pairs if s < season]
    if not prior:
        raise ValueError(f"no seasons before {season} for {award} in {path}")
    return float(np.mean(prior[-trailing:]))


def compute_weights(season, awards=DEFAULT_AWARDS, path=DEFAULT_JSON,
                    trailing=TRAILING, shrink_to_equal=SHRINK_TO_EQUAL):
    """Shrunk leaderboard-BSS weights (sum ~1) for `season`. Walk-forward."""
    skill = {aw: book_skill_trailing(aw, season, path, trailing) for aw in awards}
    floored = {aw: max(skill[aw], 0.0) for aw in awards}
    total = sum(floored.values())
    n = len(awards)
    raw = ({aw: 1.0 / n for aw in awards} if total <= 0
           else {aw: floored[aw] / total for aw in awards})
    return {aw: shrink_to_equal / n + (1.0 - shrink_to_equal) * raw[aw] for aw in awards}


def compute_budgets(season, bankroll, awards=DEFAULT_AWARDS, path=DEFAULT_JSON,
                    trailing=TRAILING, shrink_to_equal=SHRINK_TO_EQUAL):
    """Per-book dollar budgets: weights times sub-bankroll, rounded to whole dollars."""
    w = compute_weights(season, awards, path, trailing, shrink_to_equal)
    return {aw: float(round(w[aw] * bankroll)) for aw in awards}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Shrunk leaderboard-BSS stat book budgets.")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--bankroll", type=float, default=3000.0)
    ap.add_argument("--awards", nargs="+", default=list(DEFAULT_AWARDS))
    ap.add_argument("--path", default=DEFAULT_JSON)
    a = ap.parse_args()
    aws = tuple(a.awards)
    w = compute_weights(a.season, aws, a.path)
    b = compute_budgets(a.season, a.bankroll, aws, a.path)
    for aw in aws:
        print(f"  {aw}: skill_trailing={book_skill_trailing(aw, a.season, a.path):+.3f} "
              f"weight={w[aw]:.3f} budget=${b[aw]:.0f}")
    print(f"  sum budget ${sum(b.values()):.0f}")
