"""Award classification for Polymarket NBA markets.

Maps a PM market's title/slug to the award enum. Deliberately conservative:
when nothing matches confidently, return OTHER rather than guessing. The raw
fragment that was classified from is returned alongside so a misclassification
can be fixed by re-running the classifier over stored `award_raw` values WITHOUT
re-hitting the network (the whole reason award_raw exists in the schema).

PM titles are not standardised across seasons, so this WILL be wrong sometimes.
Keep the rules here, keep them readable, and expect to tune them.
"""

from __future__ import annotations

import re

# Award enum values written to pm_markets.award.
MVP = "MVP"
DPOY = "DPOY"
ROTY = "ROTY"
CHAMPIONSHIP = "CHAMPIONSHIP"
OTHER = "OTHER"

# Ordered rules: first match wins. Order matters because some titles contain
# multiple award-ish tokens (e.g. "Defensive Player" also contains "Player").
# More specific patterns must precede more general ones.
#
# Each rule is (compiled_regex, award_enum). Patterns are matched against a
# normalised lowercased string built from title + slug.
_RULES: list[tuple[re.Pattern, str]] = [
    # ROTY: rookie of the year. Check before MVP in case "rookie ... most valuable"
    # phrasing ever appears.
    (re.compile(r"rookie\s+of\s+the\s+year|\broty\b|\brotys?\b"), ROTY),
    # DPOY: defensive player of the year. Must precede MVP because the substring
    # "player of the year" is shared and we want the defensive qualifier to win.
    (re.compile(r"defensive\s+player\s+of\s+the\s+year|\bdpoy\b"), DPOY),
    # MVP: most valuable player / mvp. The regular-season award. Exclude Finals
    # MVP explicitly (out of scope, and structurally different) by not matching
    # "finals mvp" here; that falls through to OTHER unless a championship rule
    # catches it.
    (re.compile(r"(?<!finals\s)most\s+valuable\s+player|(?<!finals\s)\bmvp\b"), MVP),
    # CHAMPIONSHIP: title/champion/win the finals/nba champion.
    (re.compile(
        r"nba\s+champion|win\s+the\s+(?:nba\s+)?finals|"
        r"to\s+win\s+the\s+(?:nba\s+)?title|\bnba\s+championship\b|"
        r"\bchampions?\b"
    ), CHAMPIONSHIP),
]

# "finals mvp" should NOT classify as MVP (out of scope, conditional on finals).
# It is left as OTHER deliberately. This pattern is used to short-circuit.
_FINALS_MVP = re.compile(r"finals\s+mvp|finals\s+most\s+valuable")


def _normalise(*parts: str | None) -> str:
    """Join title/slug parts into one lowercased, single-spaced search string."""
    joined = " ".join(p for p in parts if p)
    joined = joined.replace("-", " ").replace("_", " ")
    return re.sub(r"\s+", " ", joined).strip().lower()


def classify_award(title: str | None, slug: str | None = None) -> tuple[str, str]:
    """Classify a market into the award enum.

    Returns (award_enum, award_raw) where award_raw is the normalised fragment
    that classification ran against, stored so corrections need no re-pull.

    NBA-relevance is assumed to be filtered upstream (by tag/slug). This function
    only decides WHICH award; a non-award NBA market returns OTHER.
    """
    raw = _normalise(title, slug)

    # Finals MVP is explicitly out of scope -> OTHER, even though it contains "mvp".
    if _FINALS_MVP.search(raw):
        return OTHER, raw

    for pattern, award in _RULES:
        if pattern.search(raw):
            return award, raw

    return OTHER, raw
