"""Cross-book capital allocation by shrunk contender-conditioned Brier skill score.

Splits the pooled bankroll across the award books by how well each book's model has
predicted its winner out-of-sample, then shrinks halfway to an equal split so ten noisy
seasons do not justify the full tilt. Walk-forward by construction: a target season's
weights use only seasons strictly before it, so this never leaks test-season information
into sizing.

Mechanism per book for target season T (see DECISIONS and the shrunk-BSS handoff):
  1. Contender-conditioned Brier skill score per season. Within each race, softmax the
     ensemble-mean OOF score to P(win); restrict to vote-getters (fallback top-5 by
     P(win) when a race has fewer than two); skill = 1 - Brier / Brier_baserate. The
     base-rate normalisation strips field-size uncertainty so books with different field
     widths are comparable, which is why raw Brier (which buried ROTY) is not used.
  2. Trailing-mean of the skill scores over the `trailing` seasons before T (10; longer
     would import pre-analytics voting regimes).
  3. Floor negatives to zero: a book below base-rate skill earns no capital.
  4. Normalise to sum 1.
  5. Shrink halfway to 1/N: w = 0.5 w_raw + 0.5 / N.
  6. Multiply by the bankroll and round to whole dollars. Unused budget stays cash; no
     spillover between books.

For 2025 at a 3000 bankroll this reproduces MVP 951 / DPOY 741 / ROTY 1308. British
English.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from scripts.common import config

TRAILING = 10
SHRINK_TO_EQUAL = 0.5   # weight placed on the equal (1/N) split; 0.5 = halfway
DEFAULT_AWARDS = ("MVP", "DPOY", "ROTY")


def _season_skill(d: pd.DataFrame) -> pd.Series:
    """Contender-conditioned Brier skill score per season for one book's OOF frame."""
    d = d.copy()
    if "raw_scores" in d.columns:
        d["_score"] = d["raw_scores"].apply(lambda v: float(np.mean(v)))
    else:
        d["_score"] = d["stage1_score"].astype(float)
    out = {}
    for season, g in d.groupby("season"):
        skills = []
        for _, race in g.groupby("group_key"):
            z = race["_score"].to_numpy()
            p = np.exp(z - z.max())
            p = p / p.sum()
            y = race["label_won_flag"].to_numpy().astype(float)
            contender = race["label_vote_share"].to_numpy() > 0
            if contender.sum() < 2:
                contender = np.zeros(len(p), dtype=bool)
                contender[np.argsort(-p)[:5]] = True
            pc, yc = p[contender], y[contender]
            brier = float(((pc - yc) ** 2).mean())
            base = float(yc.mean())
            bref = float(((base - yc) ** 2).mean())
            skills.append(1.0 - brier / bref if bref > 0 else np.nan)
        out[int(season)] = float(np.nanmean(skills))
    return pd.Series(out).sort_index()


def book_skill_trailing(award: str, season: int, trailing: int = TRAILING) -> float:
    """Trailing-mean skill for a book over the `trailing` seasons before `season`."""
    d = pd.read_pickle(config.OOF_BUNDLE[award])
    sk = _season_skill(d)
    prior = sk[sk.index < season]
    return float(prior.tail(trailing).mean())


def compute_weights(season: int, awards=DEFAULT_AWARDS, trailing: int = TRAILING,
                    shrink_to_equal: float = SHRINK_TO_EQUAL) -> dict:
    """Shrunk-BSS weights (sum ~1) for `season`. Walk-forward; seasons before `season`
    only. Returns {award: weight}."""
    skill = {aw: book_skill_trailing(aw, season, trailing) for aw in awards}
    floored = {aw: max(skill[aw], 0.0) for aw in awards}
    total = sum(floored.values())
    n = len(awards)
    raw = ({aw: 1.0 / n for aw in awards} if total <= 0
           else {aw: floored[aw] / total for aw in awards})
    return {aw: shrink_to_equal / n + (1.0 - shrink_to_equal) * raw[aw] for aw in awards}


def compute_budgets(season: int, bankroll: float, awards=DEFAULT_AWARDS,
                    trailing: int = TRAILING, shrink_to_equal: float = SHRINK_TO_EQUAL) -> dict:
    """Per-book dollar budgets: weights times bankroll, rounded to whole dollars.
    Reproduces MVP 951 / DPOY 741 / ROTY 1308 for season=2025, bankroll=3000."""
    w = compute_weights(season, awards, trailing, shrink_to_equal)
    return {aw: float(round(w[aw] * bankroll)) for aw in awards}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Shrunk-BSS book budgets.")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--bankroll", type=float, default=3000.0)
    a = ap.parse_args()
    w = compute_weights(a.season)
    b = compute_budgets(a.season, a.bankroll)
    for aw in DEFAULT_AWARDS:
        print(f"  {aw}: skill_trailing={book_skill_trailing(aw, a.season):.3f} "
              f"weight={w[aw]:.3f} budget=${b[aw]:.0f}")
    print(f"  sum budget ${sum(b.values()):.0f}")
