"""DO NOT RUN AGAINST THE POPULATED LIVE data/awards.db (reference implementation).

Post-migration finding (2026-07): re-running this on a populated DB rebuilds `players`
from staging + reconcile and, being DOB-gated, drops the 4 manual bref->nba overrides
(jacksja02, paytoga01, ewingpa01, johnsla02) back to null. The live DB already holds the
correct resolved state (5106 players, ~480 bref links, award_voting complete), which the
sealed result depends on. For NEW award labels (e.g. 6MOTY) use a targeted ADDITIVE
promoter that maps new stg_award_voting rows to player_ids via the EXISTING players table
and upserts only those rows, never rebuilding players. Before this file is run in anger
again it needs a hardening pass: re-add the 4 overrides and make build_players preserve
existing links on re-run. Safe to run only against a throwaway copy for reference.

Identity resolution: the single-threaded reconciler of player identity.

This module is the SOLE writer of `players.player_id`, the SOLE promoter of
`stg_award_voting` into canonical `award_voting`, and the only writer of
`pm_candidates.player_id` (UPDATE only). Routing all identity through one
component is what lets the three pullers (nba_api, bref, Polymarket) run in
parallel keyed by their own native ids without colliding.

Run:  uv run python -m scripts.data_pull.identity.resolve_identity
      uv run python -m scripts.data_pull.identity.resolve_identity --db data/awards.db

Idempotent and safely re-runnable as new staging data arrives. A second run
after more staging lands must NOT renumber existing players (see the synthetic
allocator) and must only freshly resolve newly-arrived names.

------------------------------------------------------------------------------
PINNED DECISIONS (ratified with Royce; do not silently change)
------------------------------------------------------------------------------
1. player_id allocation
   - Player present in stg_nba_players  -> player_id = nba_api_id, is_synthetic=0.
   - bref-only player (no nba_api match) -> NEGATIVE synthetic player_id
     (-1, -2, ...), is_synthetic=1, nba_api_id NULL. Negative space can never
     intersect nba_api's always-positive, ever-growing id space, so the
     guarantee is permanent (a fixed positive ceiling only defers collision).
   - is_synthetic is AUTHORITATIVE; the sign is a query convenience. Always set
     the flag, never rely on sign alone.

2. Synthetic allocation is PERSISTENT, not positional.
   On each run we read existing (bref_id -> negative player_id) pairs and
   preserve them EXACTLY, allocating fresh ids (continuing downward from the
   current minimum) only to genuinely-new bref-only players. "Deterministic by
   ascending bref_id" means allocation ORDER for new ids, not id = sort-rank.
   Positional assignment would renumber the whole synthetic space the moment an
   earlier-sorting bref_id arrived, orphaning every already-promoted
   award_voting.player_id / pm_candidates.player_id.

3. Reconciliation confidence (bref <-> nba_api when BOTH sources have a player)
   - Match on normalised name + dob + overlapping career seasons.
   - dob present on BOTH and equal  -> auto-confirm eligible (method nba_api_match).
   - dob missing on EITHER side      -> drops to fuzzy/review, NEVER auto-confirmed
     at 1.0 (bref _parse_dob returns None when the necro-birth span is absent, so
     a missing dob is silence, not evidence).

4. Fuzzy thresholds (rapidfuzz token_sort_ratio, 0-100)
   - normalised-exact            -> confidence 1.0, reviewed_flag=1, method exact
   - score >= 92                 -> tentative resolved_player_id, reviewed_flag=0
   - 88 <= score < 92            -> tentative resolved_player_id, reviewed_flag=0
   - score < 88                  -> resolved_player_id NULL,       reviewed_flag=0
   Nicknames are NOT string-distance matchable; they go through the committed
   alias seed (aliases.yaml). Any unaliased nickname is unresolved-for-review.

NEVER drop an unmatched name: it stays in name_resolution with NULL player_id.
NEVER modify staging tables or pm_markets/pm_prices.
NEVER hardcode training-window logic: resolve every player in staging, any era.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import unicodedata
from pathlib import Path

try:  # canonical shared helper in the clean tree
    from scripts.common.db import connect, upsert, utcnow_iso
except ImportError:  # pragma: no cover - db.py may export utc_now instead
    from scripts.common.db import connect, upsert, utc_now as utcnow_iso

try:
    from rapidfuzz import fuzz
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "rapidfuzz is required: add `rapidfuzz>=3.0` to pyproject and `uv sync`."
    ) from exc

log = logging.getLogger("resolve_identity")

# --- fuzzy thresholds (decision 4) -------------------------------------------
FUZZY_TENTATIVE = 88   # >= this: write a tentative resolved_player_id (review)
FUZZY_STRONG = 92      # >= this: strong-but-still-review tentative match
# < FUZZY_TENTATIVE: resolved_player_id stays NULL, reviewed_flag=0.

ALIAS_PATH = Path(__file__).with_name("aliases.yaml")

# career-overlap fallback needs SOME overlap to count toward a name+dob-missing
# match; we never auto-confirm on overlap alone (decision 3).
_SUFFIX_RE = re.compile(r"\b(jr|sr|ii|iii|iv|v)\.?\b", re.IGNORECASE)
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s+")


# -----------------------------------------------------------------------------
# Normalisation
# -----------------------------------------------------------------------------

def strip_diacritics(s: str) -> str:
    """Jokić -> Jokic. NFKD decompose, drop combining marks."""
    return "".join(
        c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)
    )


def deconstruct_reversed(raw: str) -> str:
    """'Gilgeous-Alexander, Shai' -> 'Shai Gilgeous-Alexander'.

    Only treats a single top-level comma as a reversal. Names with no comma pass
    through unchanged. Suffix-only commas ('Jokic, Jr.') are handled by suffix
    stripping downstream, but we guard the obvious case here.
    """
    if "," not in raw:
        return raw
    head, _, tail = raw.partition(",")
    head, tail = head.strip(), tail.strip()
    # If the tail is just a suffix, it's not a real First/Last reversal.
    if _SUFFIX_RE.fullmatch(tail.replace(".", "")):
        return raw
    if head and tail:
        return f"{tail} {head}"
    return raw


def normalise_name(raw: str) -> str:
    """Aggressive normalisation for MATCHING only. raw_name is stored verbatim.

    Pipeline: reversal-fix -> diacritic strip -> lowercase -> drop suffixes ->
    drop punctuation -> collapse whitespace. The result is a comparison key, not
    a display name.
    """
    if raw is None:
        return ""
    s = deconstruct_reversed(raw)
    s = strip_diacritics(s)
    s = s.lower()
    s = _SUFFIX_RE.sub(" ", s)
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


# -----------------------------------------------------------------------------
# Alias seed
# -----------------------------------------------------------------------------

def load_aliases(path: Path = ALIAS_PATH) -> dict[str, str]:
    """Load nickname -> canonical-name map, keyed by NORMALISED nickname.

    Uses PyYAML if available; falls back to a tiny line parser so the resolver
    runs even if yaml isn't installed (the seed format is deliberately flat
    'key: value' pairs). Missing file -> empty map (aliases are optional).
    """
    if not path.exists():
        log.warning("alias seed not found at %s; proceeding with no aliases", path)
        return {}
    text = path.read_text(encoding="utf-8")
    raw_map: dict[str, str] = {}
    try:
        import yaml  # type: ignore
        loaded = yaml.safe_load(text) or {}
        if isinstance(loaded, dict):
            raw_map = {str(k): str(v) for k, v in loaded.items()}
    except ImportError:  # pragma: no cover - minimal fallback parser
        for line in text.splitlines():
            line = line.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue
            k, _, v = line.partition(":")
            k = k.strip().strip('"').strip("'")
            v = v.strip().strip('"').strip("'")
            if k and v:
                raw_map[k] = v
    # Re-key by normalised nickname so lookup matches the same way names match.
    return {normalise_name(k): v for k, v in raw_map.items()}


# -----------------------------------------------------------------------------
# Loading staging into memory
# -----------------------------------------------------------------------------

def _rows(conn, sql: str, params: tuple = ()) -> list[dict]:
    return [dict(r) for r in conn.execute(sql, params).fetchall()]


def load_nba_players(conn) -> list[dict]:
    return _rows(conn, "SELECT * FROM stg_nba_players")


def load_bref_players(conn) -> list[dict]:
    return _rows(conn, "SELECT * FROM stg_bref_players")


def seasons_overlap(a_first, a_last, b_first, b_last) -> bool:
    """True if the two [first,last] STARTING-year spans intersect.

    Missing endpoints make overlap unknowable; we treat unknown as NON-overlap
    so it can't manufacture confidence. (It only ever ADDS evidence; its absence
    never auto-confirms anything on its own.)
    """
    if None in (a_first, a_last, b_first, b_last):
        return False
    return a_first <= b_last and b_first <= a_last


# -----------------------------------------------------------------------------
# Reconciliation: bref_id <-> nba_api_id (decision 3)
# -----------------------------------------------------------------------------

def reconcile(nba_players: list[dict], bref_players: list[dict]) -> dict:
    """Match bref players to nba_api players. Returns a dict with:

        matches:   {bref_id: nba_api_id}        confident name+dob(+overlap)
        review:    [(bref_id, nba_api_id, score, reason)]  needs eyeballing
        bref_only: [bref_id, ...]               no acceptable nba_api candidate

    A confident match requires normalised-name agreement AND dob present-and-
    equal on both sides. Name agreement with dob missing on either side is
    downgraded to review, never auto-confirmed (decision 3).
    """
    # Index nba players by normalised name for O(1)-ish candidate lookup.
    by_norm: dict[str, list[dict]] = {}
    for p in nba_players:
        by_norm.setdefault(normalise_name(p["name"]), []).append(p)

    matches: dict[str, int] = {}
    review: list[tuple] = []
    bref_only: list[str] = []

    for b in bref_players:
        b_norm = normalise_name(b["name"])
        candidates = by_norm.get(b_norm, [])

        if not candidates:
            # No name-equal nba candidate. Try a fuzzy pass for review only;
            # we never auto-confirm a fuzzy reconciliation.
            best, best_score = None, -1.0
            for p in nba_players:
                sc = fuzz.token_sort_ratio(b_norm, normalise_name(p["name"]))
                if sc > best_score:
                    best, best_score = p, sc
            if best is not None and best_score >= FUZZY_TENTATIVE:
                review.append((b["bref_id"], best["nba_api_id"], best_score,
                               f"fuzzy name {best_score:.0f}, manual confirm"))
            else:
                bref_only.append(b["bref_id"])
            continue

        # Name-equal candidate(s) exist. Disambiguate / confirm via dob.
        b_dob = b.get("dob")
        confirmed = None
        for p in candidates:
            p_dob = p.get("dob")
            if b_dob and p_dob and b_dob == p_dob:
                confirmed = p
                break
        if confirmed is not None:
            matches[b["bref_id"]] = confirmed["nba_api_id"]
            continue

        # Name matches but dob missing on either side, or present-but-unequal.
        # Decision 3: do NOT auto-confirm. Route to review with the best
        # name-equal candidate (prefer one with overlapping seasons if any).
        chosen = candidates[0]
        for p in candidates:
            if seasons_overlap(
                b.get("first_season"), b.get("last_season"),
                p.get("first_season"), p.get("last_season"),
            ):
                chosen = p
                break
        reason = (
            "name match but dob missing on a side"
            if not (b_dob and chosen.get("dob"))
            else "name match but dob mismatch"
        )
        review.append((b["bref_id"], chosen["nba_api_id"], 100.0, reason))

    return {"matches": matches, "review": review, "bref_only": bref_only}


# -----------------------------------------------------------------------------
# Synthetic id allocation (decision 2: persistent, not positional)
# -----------------------------------------------------------------------------

def existing_synthetic_map(conn) -> dict[str, int]:
    """Read existing bref-only synthetic allocations from `players`.

    Keyed by bref_id -> negative player_id. is_synthetic is authoritative; we do
    not rely on the sign. These are preserved verbatim across runs so re-runs
    never renumber (rule 7).
    """
    rows = conn.execute(
        "SELECT bref_id, player_id FROM players WHERE is_synthetic = 1"
    ).fetchall()
    return {r["bref_id"]: r["player_id"] for r in rows}


def allocate_synthetic_ids(conn, new_bref_only: list[str]) -> dict[str, int]:
    """Return bref_id -> player_id for ALL bref-only players (existing + new).

    Existing synthetic players keep their exact id. Genuinely-new bref-only
    players get fresh negative ids assigned in ASCENDING bref_id order,
    continuing downward from the current minimum. This is the persistent
    allocator: stable under incremental staging growth.
    """
    existing = existing_synthetic_map(conn)
    out = dict(existing)  # preserve every prior allocation exactly

    truly_new = sorted(b for b in new_bref_only if b not in existing)
    # Continue downward from the current minimum (most-negative) id, or 0 -> -1.
    current_min = min(existing.values()) if existing else 0
    next_id = current_min - 1
    for bref_id in truly_new:
        out[bref_id] = next_id
        next_id -= 1
    return out


# -----------------------------------------------------------------------------
# Building canonical players
# -----------------------------------------------------------------------------

def build_players(conn, nba_players, bref_players, recon) -> dict:
    """Upsert canonical `players`. Returns lookup maps for downstream stages.

    Returns:
        bref_to_pid:    {bref_id: player_id}    for award_voting promotion
        name_to_pid:    {normalised_name: player_id}  for PM/news name resolution
        pid_method:     {player_id: method}     for the audit trail
    """
    stamp = utcnow_iso()
    bref_to_nba = recon["matches"]
    nba_matched_ids = set(bref_to_nba.values())

    rows: list[dict] = []
    bref_to_pid: dict[str, int] = {}
    pid_method: dict[int, str] = {}

    # 1) Every nba_api player becomes a canonical row with player_id = nba_api_id.
    nba_by_id = {p["nba_api_id"]: p for p in nba_players}
    # invert matches for bref_id lookup per nba id
    nba_to_bref = {v: k for k, v in bref_to_nba.items()}

    for p in nba_players:
        nid = p["nba_api_id"]
        bref_id = nba_to_bref.get(nid)  # set only if reconciled
        rows.append({
            "player_id": nid,
            "bref_id": bref_id,
            "nba_api_id": nid,
            "name": p["name"],
            "position": p.get("position"),
            "primary_centre": 1 if (p.get("position") or "").upper().startswith("C") else 0,
            "dob": p.get("dob"),
            "draft_year": p.get("draft_year"),
            "draft_position": p.get("draft_position"),
            "lottery_pick": p.get("lottery_pick"),
            "is_synthetic": 0,
        })
        pid_method[nid] = "nba_api_match" if bref_id else "exact"
        if bref_id:
            bref_to_pid[bref_id] = nid

    # 2) bref-only players (no nba match) get persistent negative synthetic ids.
    bref_only = recon["bref_only"]
    syn_map = allocate_synthetic_ids(conn, bref_only)
    bref_by_id = {b["bref_id"]: b for b in bref_players}
    for bref_id in bref_only:
        pid = syn_map[bref_id]
        b = bref_by_id[bref_id]
        rows.append({
            "player_id": pid,
            "bref_id": bref_id,
            "nba_api_id": None,                 # EXPECTED NULL for synthetic
            "name": b["name"],
            "position": b.get("position"),
            "primary_centre": 1 if (b.get("position") or "").upper().startswith("C") else 0,
            "dob": b.get("dob"),
            "draft_year": None,
            "draft_position": None,
            "lottery_pick": None,
            "is_synthetic": 1,
        })
        pid_method[pid] = "bref_match"
        bref_to_pid[bref_id] = pid

    upsert(conn, "players", rows, ["player_id"])

    # Build normalised-name -> player_id map from the freshly written players.
    name_to_pid: dict[str, int] = {}
    for r in conn.execute("SELECT player_id, name FROM players").fetchall():
        name_to_pid.setdefault(normalise_name(r["name"]), r["player_id"])

    return {
        "bref_to_pid": bref_to_pid,
        "name_to_pid": name_to_pid,
        "pid_method": pid_method,
        "review": recon["review"],
        "syn_map": syn_map,
    }


# -----------------------------------------------------------------------------
# name_resolution audit writes
# -----------------------------------------------------------------------------

def _resolution_row(raw_name, source, native_id, pid, method, confidence,
                    reviewed_flag, notes, stamp) -> dict:
    return {
        "raw_name": raw_name,
        "source": source,
        "native_id": None if native_id is None else str(native_id),
        "resolved_player_id": pid,
        "method": method,
        "confidence": confidence,
        "reviewed_flag": reviewed_flag,
        "notes": notes,
        "resolved_at": stamp,
    }


def record_source_resolutions(conn, maps, nba_players, bref_players) -> None:
    """Write name_resolution rows for the nba_api and bref sources.

    These are the 'native' sources: their names ARE the canonical/staged names,
    so they resolve at confidence 1.0. The audit row exists so every raw name
    from every source has a traceable resolution (DO #2, #8).
    """
    stamp = utcnow_iso()
    name_to_pid = maps["name_to_pid"]
    bref_to_pid = maps["bref_to_pid"]
    out: list[dict] = []

    for p in nba_players:
        pid = p["nba_api_id"]
        out.append(_resolution_row(
            p["name"], "nba_api", pid, pid, "exact", 1.0, 1,
            "nba_api native name", stamp,
        ))

    for b in bref_players:
        pid = bref_to_pid.get(b["bref_id"])
        method = maps["pid_method"].get(pid, "bref_match")
        # native bref name maps to its (possibly reconciled) canonical id
        out.append(_resolution_row(
            b["name"], "bref", b["bref_id"], pid,
            method if method in ("nba_api_match", "bref_match") else "bref_match",
            1.0, 1, "bref native name", stamp,
        ))

    # Reconciliation review items: name matched but dob missing/mismatch, or a
    # fuzzy-only name candidate. Tentatively point at the candidate id but flag.
    nba_id_to_pid = {p["nba_api_id"]: p["nba_api_id"] for p in nba_players}
    for bref_id, nba_id, score, reason in maps["review"]:
        tentative_pid = nba_id_to_pid.get(nba_id)
        # >= strong/tentative -> keep tentative pid; below -> NULL.
        if score >= FUZZY_TENTATIVE:
            pid, conf = tentative_pid, round(score / 100.0, 3)
        else:
            pid, conf = None, round(score / 100.0, 3)
        out.append(_resolution_row(
            # raw_name is the bref display name being reconciled
            next((b["name"] for b in bref_players if b["bref_id"] == bref_id), bref_id),
            "bref", bref_id, pid, "fuzzy", conf, 0,
            f"reconcile review: {reason}", stamp,
        ))

    upsert(conn, "name_resolution", out, ["raw_name", "source", "native_id"])


# -----------------------------------------------------------------------------
# PM candidate name resolution + backfill
# -----------------------------------------------------------------------------

def resolve_pm_candidates(conn, maps, aliases) -> dict:
    """Resolve pm_candidates.candidate_name -> player_id, write audit + backfill.

    Yes/No outcome rows are NOT player names; we skip backfilling those (their
    candidate_name is 'Yes'/'No'), but we still leave them untouched rather than
    erroring. Player-name candidates go through: alias -> normalised-exact ->
    fuzzy bands (decision 4).
    """
    stamp = utcnow_iso()
    name_to_pid = maps["name_to_pid"]
    # Precompute (normalised_name, player_id) list for fuzzy scanning.
    norm_pid_pairs = list(name_to_pid.items())

    cands = _rows(
        conn,
        "SELECT market_id, candidate_id, candidate_name, player_id FROM pm_candidates",
    )

    audit: list[dict] = []
    backfill: list[tuple[int, str, str]] = []  # (player_id, market_id, candidate_id)
    counts = {"auto": 0, "review_tentative": 0, "unresolved": 0, "skipped_yesno": 0}

    for c in cands:
        raw = c["candidate_name"]
        if raw is None or raw.strip() in ("", "Yes", "No"):
            counts["skipped_yesno"] += 1
            continue

        norm = normalise_name(raw)
        alias_note = ""

        # 1) alias hint -> canonical name, then resolve that name normally.
        if norm in aliases:
            canonical = aliases[norm]
            alias_note = f"alias '{raw}' -> '{canonical}'; "
            norm = normalise_name(canonical)

        # 2) normalised-exact
        if norm in name_to_pid:
            pid = name_to_pid[norm]
            audit.append(_resolution_row(
                raw, "pm_gamma", c["candidate_id"], pid, "exact", 1.0, 1,
                alias_note + "normalised-exact", stamp,
            ))
            backfill.append((pid, c["market_id"], c["candidate_id"]))
            counts["auto"] += 1
            continue

        # 3) fuzzy bands
        best_pid, best_score = None, -1.0
        for nname, pid in norm_pid_pairs:
            sc = fuzz.token_sort_ratio(norm, nname)
            if sc > best_score:
                best_pid, best_score = pid, sc

        if best_score >= FUZZY_TENTATIVE:
            # tentative resolved id, flagged for review (covers both 88-92 and >=92)
            band = "strong" if best_score >= FUZZY_STRONG else "weak"
            audit.append(_resolution_row(
                raw, "pm_gamma", c["candidate_id"], best_pid, "fuzzy",
                round(best_score / 100.0, 3), 0,
                alias_note + f"fuzzy {band} {best_score:.0f}, review", stamp,
            ))
            # tentative ids are written to the audit but NOT backfilled into
            # pm_candidates until reviewed: a wrong id silently corrupts labels.
            counts["review_tentative"] += 1
        else:
            audit.append(_resolution_row(
                raw, "pm_gamma", c["candidate_id"], None, "fuzzy",
                round(best_score / 100.0, 3), 0,
                alias_note + (f"below threshold {best_score:.0f}"
                              if not alias_note else "alias target unresolved"),
                stamp,
            ))
            counts["unresolved"] += 1

    upsert(conn, "name_resolution", audit, ["raw_name", "source", "native_id"])

    # Backfill only the confident (auto) resolutions. UPDATE, never insert.
    for pid, market_id, candidate_id in backfill:
        conn.execute(
            "UPDATE pm_candidates SET player_id = ? "
            "WHERE market_id = ? AND candidate_id = ?",
            (pid, market_id, candidate_id),
        )
    conn.commit()
    return counts


# -----------------------------------------------------------------------------
# Promote stg_award_voting -> award_voting
# -----------------------------------------------------------------------------

def promote_award_voting(conn, bref_to_pid: dict[str, int]) -> dict:
    """Promote staged voting rows whose bref_id has resolved to a player_id.

    Carries vote_share/rank/etc unchanged. Rows whose bref_id is not yet
    resolved are left in staging (not an error: they promote on a later run once
    identity resolves). Idempotent via upsert on (season, award, player_id).
    """
    staged = _rows(conn, "SELECT * FROM stg_award_voting")
    rows: list[dict] = []
    unresolved_bref: set[str] = set()

    for s in staged:
        pid = bref_to_pid.get(s["bref_id"])
        if pid is None:
            unresolved_bref.add(s["bref_id"])
            continue
        rows.append({
            "season": s["season"],
            "award": s["award"],
            "player_id": pid,
            "first_place_votes": s.get("first_place_votes"),
            "total_points": s.get("total_points"),
            "vote_share": s.get("vote_share"),
            "rank": s.get("rank"),
            "won_flag": s.get("won_flag"),
        })

    upsert(conn, "award_voting", rows, ["season", "award", "player_id"])
    return {"promoted": len(rows), "unresolved_bref": sorted(unresolved_bref)}


# -----------------------------------------------------------------------------
# Orchestration + report
# -----------------------------------------------------------------------------

def resolve(db_path: str) -> dict:
    conn = connect(db_path)
    try:
        aliases = load_aliases()
        nba_players = load_nba_players(conn)
        bref_players = load_bref_players(conn)

        recon = reconcile(nba_players, bref_players)
        maps = build_players(conn, nba_players, bref_players, recon)
        record_source_resolutions(conn, maps, nba_players, bref_players)
        pm_counts = resolve_pm_candidates(conn, maps, aliases)
        promo = promote_award_voting(conn, maps["bref_to_pid"])

        # End-of-run report = Royce's manual-review worklist (DO #9).
        total_names = conn.execute(
            "SELECT COUNT(*) FROM name_resolution"
        ).fetchone()[0]
        auto = conn.execute(
            "SELECT COUNT(*) FROM name_resolution "
            "WHERE resolved_player_id IS NOT NULL AND reviewed_flag = 1"
        ).fetchone()[0]
        pending = conn.execute(
            "SELECT COUNT(*) FROM name_resolution "
            "WHERE reviewed_flag = 0 AND resolved_player_id IS NOT NULL"
        ).fetchone()[0]
        unresolved = conn.execute(
            "SELECT COUNT(*) FROM name_resolution WHERE resolved_player_id IS NULL"
        ).fetchone()[0]

        n_players = conn.execute("SELECT COUNT(*) FROM players").fetchone()[0]
        n_synth = conn.execute(
            "SELECT COUNT(*) FROM players WHERE is_synthetic = 1"
        ).fetchone()[0]

        report = {
            "players_total": n_players,
            "players_synthetic": n_synth,
            "reconcile_matches": len(recon["matches"]),
            "reconcile_review": len(recon["review"]),
            "bref_only": len(recon["bref_only"]),
            "names_total": total_names,
            "names_auto_resolved": auto,
            "names_pending_review": pending,
            "names_unresolved": unresolved,
            "pm_candidate_counts": pm_counts,
            "award_voting_promoted": promo["promoted"],
            "award_voting_unresolved_bref": promo["unresolved_bref"],
        }
        return report
    finally:
        conn.close()


def _print_report(report: dict) -> None:
    log.info("identity resolution complete")
    log.info("players: %d total (%d synthetic / bref-only)",
             report["players_total"], report["players_synthetic"])
    log.info("reconcile: %d confident, %d to review, %d bref-only",
             report["reconcile_matches"], report["reconcile_review"],
             report["bref_only"])
    log.info("names: %d total | %d auto-resolved | %d pending review | %d unresolved",
             report["names_total"], report["names_auto_resolved"],
             report["names_pending_review"], report["names_unresolved"])
    log.info("pm candidates: %s", report["pm_candidate_counts"])
    log.info("award_voting promoted: %d rows", report["award_voting_promoted"])
    if report["award_voting_unresolved_bref"]:
        log.warning("award_voting: %d bref_ids not yet resolved (promote on a "
                    "later run): %s",
                    len(report["award_voting_unresolved_bref"]),
                    report["award_voting_unresolved_bref"][:20])


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(
        description="Reconcile player identity across nba_api/bref/Polymarket."
    )
    p.add_argument("--db", default="data/awards.db")
    args = p.parse_args(argv)
    report = resolve(args.db)
    _print_report(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
