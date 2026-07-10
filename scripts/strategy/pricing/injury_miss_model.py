"""Miss-count distribution: how many more games an injured player will miss.

This is the object that feeds the eligibility avail mixture. For a player who is
currently on the injury report in state (status, category), we want a
distribution over m, the number of additional team games he will miss, so the
eligibility factor integrates the binomial over it:

    P_elig = mean over sampled m of  binom.sf(needed - 1, max(0, avail - m), p)

rather than committing to a single availability guess. Season-ending is the
degenerate tail (m consumes all remaining games); a day-to-day designation is a
spike near zero; a murky multi-week injury is a broad middle. The distribution
is estimated empirically from the populated injuries table joined to the game
logs, not assumed.

DEFINITIONS (all from the game logs; a game-log row = an appearance, no row = a
true absence):
  m       forward consecutive team games missed from the report date until the
          player's next appearance, or to season end.
  censored True if he never reappears that season. This, not m, is the
          season-ending signal: an ACL observed with 5 games left has a small m
          but is still censored, and at serve time a censored sample means "miss
          all remaining" (miss = avail), so it collapses eligibility correctly
          regardless of when in the season it was observed.
  elapsed consecutive team games missed immediately BEFORE the report date (the
          current spell's duration so far). Recorded for diagnostics and as an
          optional serve-time conditioning refinement; a player deep into an
          absence has a different remaining-duration profile from one just down.

LABELLING vs LEAKAGE: m and censored use games AFTER the report date. That is
correct for TRAINING the distribution (supervised base rates from realised
history). At serve time the eligibility factor never reads them; it looks up the
distribution for the current (status, category) and integrates. The two are not
the same operation.

CORRELATED OBSERVATIONS: injuries is day-level, so one absence spell of N days
contributes N correlated observations (m decreasing each day). The censored flag
is spell-invariant so season-ending detection is unaffected, but per-cell n
overstates the independent sample size, and the uncensored m is a spell-position
marginal. The diagnostic can restrict to spell onset (elapsed <= 1) for the
fresh-injury view. Conditioning on elapsed is the v1.1 sharpening.

SEALED 2025: this learns generic injury-duration base rates, not award labels,
so including 2025 is defensible (Royce's call), but --exclude-seasons is exposed
and the season set is recorded in the artefact metadata for auditability.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sqlite3
import statistics
from pathlib import Path

from scipy.stats import binom

try:
    from scripts.common.injury_categories import classify_reason, CATEGORY_ORDER
except ImportError:
    from injury_categories import classify_reason, CATEGORY_ORDER

log = logging.getLogger("injury_miss_model")

DEFAULT_ARTEFACT = "data/injury_miss_distribution.json"
MIN_CELL_SAMPLES = 30

# Not injuries and not a recurring availability regime: the 2021-22 COVID
# "Health and Safety Protocols" tag, roster-ineligibility, and blank/placeholder
# reasons. Excluded from the miss-count model entirely (tracked in metadata) so a
# one-off historical tag never contaminates the injury cells.
import re as _re
_NON_INJURY = _re.compile(
    r"health\s+and\s+safety|protocol|ineligible|^\s*-?\s*$|not\s+available",
    _re.IGNORECASE)


def _is_non_injury(desc) -> bool:
    return bool(_NON_INJURY.search((desc or "").strip()))


def season_of(date_iso: str) -> int:
    """NBA season as STARTING year: Oct-Dec -> that year, Jan-Jun -> year - 1."""
    d = dt.date.fromisoformat(str(date_iso)[:10])
    return d.year if d.month >= 9 else d.year - 1


# -----------------------------------------------------------------------------
# In-memory game-log views (preloaded once; the estimation is otherwise O(rows))
# -----------------------------------------------------------------------------

def _load_schedules(conn, seasons):
    """(season, team_id) -> sorted list of (game_date, game_id)."""
    sched = {}
    q = ("SELECT DISTINCT season, team_id, game_date, game_id "
         "FROM stg_nba_player_game_logs "
         "WHERE game_date IS NOT NULL AND team_id IS NOT NULL")
    for season, team_id, gdate, gid in conn.execute(q):
        if seasons and int(season) not in seasons:
            continue
        sched.setdefault((int(season), int(team_id)), []).append((gdate, gid))
    for key in sched:
        sched[key].sort()
    return sched


def _load_appearances(conn, seasons):
    """nba_api_id -> season -> {game_id appeared} and a sorted (date, team_id)
    appearance list for current-team resolution."""
    appear, atrail = {}, {}
    q = ("SELECT nba_api_id, season, game_id, game_date, team_id "
         "FROM stg_nba_player_game_logs "
         "WHERE game_date IS NOT NULL AND minutes IS NOT NULL")
    for nid, season, gid, gdate, team_id in conn.execute(q):
        if seasons and int(season) not in seasons:
            continue
        nid, season = int(nid), int(season)
        appear.setdefault(nid, {}).setdefault(season, set()).add(gid)
        atrail.setdefault(nid, {}).setdefault(season, []).append((gdate, team_id))
    for nid in atrail:
        for season in atrail[nid]:
            atrail[nid][season].sort()
    return appear, atrail


def _current_team(atrail_ns, snapshot_date):
    """Team of the most recent appearance on or before the snapshot, or None."""
    team = None
    for gdate, team_id in atrail_ns:
        if gdate <= snapshot_date:
            team = int(team_id) if team_id is not None else team
        else:
            break
    return team


def _forward_missed(team_games, appeared, snapshot_date):
    """(m, censored): consecutive team games from snapshot_date with no
    appearance until the next appearance; censored if none ever reappears."""
    future = [(d, g) for (d, g) in team_games if d >= snapshot_date]
    m = 0
    for (d, gid) in future:
        if gid in appeared:
            return m, False
        m += 1
    return m, True


def _elapsed_missed(team_games, appeared, snapshot_date):
    past = [(d, g) for (d, g) in team_games if d < snapshot_date]
    e = 0
    for (d, gid) in reversed(past):
        if gid in appeared:
            break
        e += 1
    return e


# -----------------------------------------------------------------------------
# Estimation
# -----------------------------------------------------------------------------

def build_distribution(conn, seasons=None) -> dict:
    """Build the (status, category) -> miss-count sample distribution from the
    injuries table joined to the game logs. seasons is an optional set of
    starting-year ints to include (None = all)."""
    pid_to_nba = {int(pid): (int(nid) if nid is not None else None)
                  for pid, nid in conn.execute(
                      "SELECT player_id, nba_api_id FROM players")}
    sched = _load_schedules(conn, seasons)
    appear, atrail = _load_appearances(conn, seasons)

    cells, n_obs, skipped = {}, 0, 0
    n_non_injury = 0
    seen_seasons = set()
    rows = conn.execute(
        "SELECT player_id, snapshot_date, status, description FROM injuries")
    for pid, snap, status, desc in rows:
        if _is_non_injury(desc):
            n_non_injury += 1
            continue
        season = season_of(snap)
        if seasons and season not in seasons:
            continue
        nid = pid_to_nba.get(int(pid))
        if nid is None:
            skipped += 1
            continue
        atrail_ns = atrail.get(nid, {}).get(season)
        if not atrail_ns:
            skipped += 1
            continue
        team_id = _current_team(atrail_ns, snap)
        if team_id is None:
            skipped += 1
            continue
        team_games = sched.get((season, team_id))
        if not team_games:
            skipped += 1
            continue
        appeared = appear.get(nid, {}).get(season, set())
        m, censored = _forward_missed(team_games, appeared, snap)
        elapsed = _elapsed_missed(team_games, appeared, snap)
        cat = classify_reason(desc)
        st = (status or "unknown").strip().lower()
        key = f"{st}|{cat}"
        cells.setdefault(key, []).append([int(m), int(censored), int(elapsed)])
        n_obs += 1
        seen_seasons.add(season)

    summary_cells = {}
    for key, samples in cells.items():
        ms = [s[0] for s in samples]
        cens = [s[1] for s in samples]
        summary_cells[key] = {
            "n": len(samples),
            "censor_rate": round(sum(cens) / len(samples), 4),
            "m_mean": round(statistics.mean(ms), 3),
            "m_p50": int(statistics.median(ms)),
            "m_p90": int(sorted(ms)[max(0, int(0.9 * len(ms)) - 1)]),
            "samples": samples,
        }
    return {
        "meta": {
            "generated": dt.datetime.now().isoformat(timespec="seconds"),
            "seasons": sorted(seen_seasons),
            "n_obs": n_obs,
            "n_skipped_no_schedule": skipped,
            "n_excluded_non_injury": n_non_injury,
            "min_cell_samples": MIN_CELL_SAMPLES,
        },
        "cells": summary_cells,
    }


def save_distribution(dist: dict, path: str = DEFAULT_ARTEFACT):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dist, indent=2))
    return str(p)


def load_distribution(path: str = DEFAULT_ARTEFACT) -> dict:
    return json.loads(Path(path).read_text())


# -----------------------------------------------------------------------------
# Serve-time mixture (consumed by eligibility)
# -----------------------------------------------------------------------------

def _cell_samples(dist, status, category, min_samples=MIN_CELL_SAMPLES):
    """Samples for (status, category), falling back to the category marginal
    across statuses, then to all injured samples, when a cell is too thin."""
    cells = dist["cells"]
    key = f"{status}|{category}"
    cell = cells.get(key)
    if cell and cell["n"] >= min_samples:
        return cell["samples"]
    cat_samples = [s for k, c in cells.items()
                   if k.endswith(f"|{category}") for s in c["samples"]]
    if len(cat_samples) >= min_samples:
        return cat_samples
    return [s for c in cells.values() for s in c["samples"]]


def p_elig_mixture(dist, status, category, avail, needed, p,
                   min_samples=MIN_CELL_SAMPLES):
    """Eligibility probability as the miss-count mixture over the binomial.

    For each historical sample, a censored spell means the player misses all
    remaining games (miss = avail); an uncensored spell missed an absolute count
    truncated at avail. The binomial then runs on the surviving active games."""
    if needed <= 0:
        return 1.0
    if avail <= 0:
        return 0.0
    samples = _cell_samples(dist, str(status).lower(), category, min_samples)
    if not samples:
        return float(binom.sf(needed - 1, avail, p))
    total = 0.0
    for s in samples:
        m, censored = int(s[0]), int(s[1])
        miss = avail if censored else min(m, avail)
        eff = max(0, avail - miss)
        total += float(binom.sf(needed - 1, eff, p)) if eff > 0 else 0.0
    return total / len(samples)


# -----------------------------------------------------------------------------
# Diagnostic
# -----------------------------------------------------------------------------

def dump_category(conn, category: str):
    """Print every distinct reason string that classifies into `category`, with
    its row count, most common first. The fastest way to audit misclassification
    directly (e.g. confirm no Achilles-soreness sits in season_ending)."""
    from collections import Counter
    c = Counter()
    for (desc,) in conn.execute("SELECT description FROM injuries"):
        if classify_reason(desc) == category:
            c[desc or "(null)"] += 1
    print(f"\nreason strings classified as '{category}' "
          f"({len(c)} distinct, {sum(c.values())} rows):")
    for desc, n in c.most_common():
        print(f"{n:>6}  {desc}")


def diagnose(dist: dict, onset_only: bool = False):
    """Print per-cell counts, censor rate, and m summary so the categories can be
    validated (season_ending should show high censor_rate and high m; minor low
    on both). onset_only restricts to spell onset (elapsed <= 1), the fresh-injury
    view that is not diluted by mid-spell daily observations."""
    print(f"\ninjury miss-count distribution  seasons={dist['meta']['seasons']}  "
          f"n_obs={dist['meta']['n_obs']}  skipped={dist['meta']['n_skipped_no_schedule']}")
    print(f"{'status|category':<28} {'n':>6} {'censor':>7} {'m_mean':>7} "
          f"{'m_p50':>6} {'m_p90':>6}")
    rows = []
    for key, c in dist["cells"].items():
        if onset_only:
            ms = [s[0] for s in c["samples"] if s[2] <= 1]
            cens = [s[1] for s in c["samples"] if s[2] <= 1]
            if not ms:
                continue
            n, cr = len(ms), sum(cens) / len(cens)
            mm, p50 = statistics.mean(ms), statistics.median(ms)
            p90 = sorted(ms)[max(0, int(0.9 * len(ms)) - 1)]
        else:
            n, cr, mm, p50, p90 = (c["n"], c["censor_rate"], c["m_mean"],
                                   c["m_p50"], c["m_p90"])
        st, cat = key.split("|", 1)
        order = CATEGORY_ORDER.index(cat) if cat in CATEGORY_ORDER else 99
        rows.append((st, order, key, n, cr, mm, p50, p90))
    for _, _, key, n, cr, mm, p50, p90 in sorted(rows, key=lambda r: (r[0], r[1])):
        print(f"{key:<28} {n:>6} {cr:>7.2f} {mm:>7.2f} {p50:>6} {p90:>6}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(description="injury miss-count distribution")
    ap.add_argument("--db", default="data/awards.db")
    ap.add_argument("--out", default=DEFAULT_ARTEFACT)
    ap.add_argument("--exclude-seasons", default="", help="comma-sep starting years")
    ap.add_argument("--diagnose", action="store_true")
    ap.add_argument("--onset-only", action="store_true",
                    help="diagnostic restricted to spell onset (elapsed<=1)")
    ap.add_argument("--dump-category", default=None,
                    help="print distinct reason strings feeding a category, then exit")
    args = ap.parse_args(argv)

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row

    if args.dump_category:
        dump_category(conn, args.dump_category)
        conn.close()
        return 0

    all_seasons = {int(r[0]) for r in conn.execute(
        "SELECT DISTINCT season FROM stg_nba_player_game_logs WHERE season IS NOT NULL")}
    exclude = {int(x) for x in args.exclude_seasons.split(",") if x.strip()}
    seasons = (all_seasons - exclude) if exclude else None

    dist = build_distribution(conn, seasons=seasons)
    path = save_distribution(dist, args.out)
    log.info("wrote %s  (%d cells, %d obs)", path, len(dist["cells"]),
             dist["meta"]["n_obs"])
    if args.diagnose:
        diagnose(dist, onset_only=args.onset_only)
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
