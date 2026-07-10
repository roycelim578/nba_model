"""Reason-to-category classifier for NBA injury-report descriptions.

Maps the free-text injuries.description ("Injury/Illness - Right Acl; Tear")
into coarse severity categories ordered by expected absence duration.

season_ending is GATED, not a bare keyword match, because the report uses the
same body-part word for a rupture and for soreness ("Achilles; Tear" is
season-ending; "Achilles; Soreness" and "Achilles; Tendinopathy" are not). A
severe body part (ACL, Achilles, patellar/quad tendon) counts as season-ending
only when paired with a severe diagnosis token (tear, torn, rupture, repair,
surgery, reconstruction), or when it appears bare with no benign qualifier
(a conservative default for an unqualified serious-ligament mention). Anything
carrying a benign token (soreness, tendinitis, tendinopathy, sprain, strain)
falls through and is classified on that token instead. A short list of
diagnoses with no benign version (rupture, blood clot, pulmonary embolism,
season-ending leg-bone fractures, Lisfranc) match unconditionally.

The remaining categories are ordered, first-match-wins keyword rules. These are
transparent rules, not a learned model: their job is to create cells whose
realised miss-count distributions separate, which injury_miss_model validates
empirically. Refine against that diagnostic and the real reason strings (use the
injury_miss_model --dump-category audit), not by intuition.

Category is the severity of the diagnosis, NOT its timing; carryover-versus-fresh
is an onset question answered from the game logs in injury_miss_model.
"""

from __future__ import annotations

import re

CATEGORY_ORDER = [
    "season_ending",   # ACL/Achilles tear, ruptures, major leg fractures, clots
    "major",           # surgery, meniscus, labrum, stress fracture: multi-week
    "moderate",        # sprain, strain, generic tear/fracture, tendinopathy
    "minor",           # soreness, contusion, spasm: days
    "illness",         # illness, flu, covid, concussion, conditioning
    "recovery",        # rehab / injury recovery / return-to-competition: ambiguous
    "rest",            # load management, personal, G-League two-way
    "unknown",         # unmatched
]

# season_ending gating
_SEVERE_BODYPARTS = re.compile(
    r"\bacl\b|\ba\.c\.l|achilles|patell\w*\s+tendon|quad\w*\s+tendon")
_SEVERE_TOKENS = re.compile(
    r"\btear\b|\btorn\b|ruptur|\brepair\b|surgery|reconstruction")
_BENIGN_TOKENS = re.compile(
    r"soreness|tightness|tendinitis|tendonitis|tendinopathy|bursitis|"
    r"inflammation|impingement|sprain|strain|spasm|contusion|bruise|"
    r"irritation|management|maintenance|\bache\b|discomfort|\bsore\b|tendinosis")
_UNCONDITIONAL_SE = re.compile(
    r"ruptur|blood\s+clot|pulmonary|season[- ]ending|out\s+for\s+(the\s+)?season|"
    r"(tibia|fibula|femur)\b.*?(fractur|stress|surgery)|navicular\b.*?fractur|jones\s+fracture|lisfranc")


def _is_season_ending(s: str) -> bool:
    if _UNCONDITIONAL_SE.search(s):
        return True
    if _SEVERE_BODYPARTS.search(s):
        if _SEVERE_TOKENS.search(s):
            return True
        if not _BENIGN_TOKENS.search(s):
            return True  # bare severe body-part mention: conservative default
    return False


_RULES = [
    ("major", [
        r"sur?gery", r"\brepair\b", r"procedure", r"stress\s+fracture",
        r"stress\s+reaction", r"meniscus", r"labrum", r"plantar\s+fasc",
        r"microfracture", r"\bfusion\b", r"\bbroken\b", r"meniscectomy",
        r"arthroscop", r"thrombosis", r"pneumothorax", r"reconstruction",
        r"plantar[^a-z]*fasc",
    ]),
    ("moderate", [
        r"sprain", r"strain", r"fracture", r"\btear\b", r"\btorn\b", r"bursitis",
        r"tendinitis", r"tendonitis", r"tendinopathy", r"impingement",
        r"dislocat", r"subluxation", r"\bpartial\b", r"\bhigh\s+ankle\b",
        r"effusion", r"hyperexten", r"chondromalacia", r"herniat", r"\bhernia\b",
        r"disc\b", r"bulge", r"tendinosis", r"periostitis", r"synovitis",
    ]),
    ("minor", [
        r"soreness", r"tightness", r"spasm", r"contusion", r"bruise",
        r"laceration", r"irritation", r"inflammation", r"stinger", r"cramp",
        r"\bache\b", r"discomfort", r"\bsore\b", r"\bpain\b", r"swelling",
        r"swollen", r"stiffness", r"\bstiff\b",
    ]),
    ("illness", [
        r"illness", r"\bflu\b", r"covid", r"virus", r"gastro", r"\bsick\b",
        r"non[- ]covid", r"conditioning", r"respiratory", r"migraine",
        r"dental", r"concussion", r"\bnasal\b",
    ]),
    ("recovery", [
        r"injury\s+recovery", r"\brehab", r"injury\s+management",
        r"return\s+to\s+competition", r"reconditioning", r"injury\s+maintenance",
    ]),
    ("rest", [
        r"\brest\b", r"load\s+management", r"maintenance", r"\bcoach",
        r"personal", r"not\s+with\s+team", r"g[- ]?league", r"two[- ]way",
        r"suspension", r"\btrade\b", r"\bwaiv",
    ]),
]

_COMPILED = [(cat, [re.compile(p) for p in pats]) for cat, pats in _RULES]


def classify_reason(reason) -> str:
    """Return the severity category for a report reason string. Empty text
    returns "unknown"."""
    s = (reason or "").strip().lower()
    if not s:
        return "unknown"
    # boilerplate "Injury/Illness - " prefix would otherwise match `illness`
    s = re.sub(r"injury\s*/\s*illness", " ", s)
    if _is_season_ending(s):
        return "season_ending"
    for cat, pats in _COMPILED:
        if any(p.search(s) for p in pats):
            return cat
    return "unknown"


def classify_many(reasons) -> dict:
    """Convenience: {category: count} over an iterable of reason strings."""
    out = {c: 0 for c in CATEGORY_ORDER}
    for r in reasons:
        out[classify_reason(r)] += 1
    return out
