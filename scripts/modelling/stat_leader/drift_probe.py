"""Direct-observation drift probe for the stat-leader volume nodes.

The mu question before any leading-indicator build: is the forward-rate drift
HETEROGENEOUS enough across contenders to reorder the leader, and is it
concentrated in CLOSE races (where getting mu wrong is costly)? A drift shared
across the field cancels in the argmax (the reb_env / soft-root lesson), so a
uniform bias, however real, buys per-player Brier and never who-leads.

This is MC-free. It works in STAT space (points/rebounds/assists/steals/blocks),
which is what the market resolves on. For each weekly snapshot it holds
availability at its realised value and swaps only the per-minute RATE leg between
a naive banked-rate extrapolation (the model's mean, which does not beat pace
extrapolation) and the realised remaining rate (the oracle). Because availability
is realised, eff_oracle == realised_eff, so the leader-flip fraction is a hard,
zero-new-data ceiling on what any mu forecast could change.

Outputs per stat:
  - realised leaderboard for --show-season (the corrected label; run on 2022 for STL);
  - signed drift by season stage (reproduces the 8.4 sign; + = over-projection);
  - leader-flip fraction overall, by stage, and close vs wide races (the ceiling);
  - share of flips that sit in close races (the point-5 costly zone);
  - normalised within-race drift dispersion (the heterogeneity that permits a flip);
  - drift vs banked sample size (is early small-sample over-projection the driver).

The variance channel (how an overlay's own uncertainty propagates into the traded
P) is NOT answered here; that is intrinsically an MC/oracle question and belongs to
the oracle arm. This probe answers the necessary condition (heterogeneous,
as-of-plausible mu drift that reorders the leader), cheaply, first.

Seal discipline: defaults to 2013-2023 and refuses 2024/2025 unless --allow-sealed
is passed for an explicitly non-sealed book (never for a sealed price season).

Run:
  uv run python3 -m scripts.modelling.stat_leader.drift_probe \
      --stat all --seasons 2013-2023 --show-season 2022
"""
from __future__ import annotations

import argparse
import math
import statistics as st
import sys

try:
    from scripts.common.db import connect
    from scripts.features.stat_leader import nodes as N
    from scripts.modelling.stat_leader import mc as MC
except ImportError:  # pragma: no cover
    from db import connect  # type: ignore
    import nodes as N  # type: ignore
    import mc as MC  # type: ignore

QUAL_FRAC = 0.70
FIELD_N = 30
SEALED = {"pts": {2025}, "reb": {2025}, "ast": {2025},
          "stl": {2024, 2025}, "blk": {2024, 2025}}


def _pts(d):
    return 2.0 * (d.get("fg2m") or 0.0) + 3.0 * (d.get("fg3m") or 0.0) + (d.get("ftm") or 0.0)


STAT_COUNT = {
    "reb": lambda d: d.get("reb") or 0.0,
    "pts": _pts,
    "ast": lambda d: d.get("ast") or 0.0,
    "stl": lambda d: d.get("stl") or 0.0,
    "blk": lambda d: d.get("blk") or 0.0,
}
STAT_FLOOR = {"pts": 1997, "reb": 1997, "ast": 2013, "stl": 1997, "blk": 1997}


def _mn(d):
    return d.get("min_asof") or 0.0


def _gp(d):
    return d.get("gp_played_asof") or 0.0


def _names(conn):
    try:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(players)")]
        namecol = next((c for c in ("display_name", "full_name", "name", "player_name")
                        if c in cols), None)
        idcol = "player_id" if "player_id" in cols else (
            "nba_api_id" if "nba_api_id" in cols else cols[0])
        if not namecol:
            return {}
        return {r[0]: r[1] for r in conn.execute(f"SELECT {idcol}, {namecol} FROM players")}
    except Exception:
        return {}


def _stage(frac_banked):
    if frac_banked < 1.0 / 3:
        return "early"
    if frac_banked < 2.0 / 3:
        return "mid"
    return "late"


def _season_field(counts, finals, season, snap, stat):
    """Top FIELD_N by banked per-game stat at this snapshot, gp>=1."""
    rows = []
    for (s, sn, pid), d in counts.items():
        if s != season or sn != snap:
            continue
        gp = _gp(d)
        if gp < 1 or (season, pid) not in finals:
            continue
        rows.append((STAT_COUNT[stat](d) / gp, pid, d))
    rows.sort(reverse=True)
    return [(pid, d) for _, pid, d in rows[:FIELD_N]]


def _race(counts, finals, season, snap, stat, qual, sc):
    """Return per-race records or None if no projectable remainder."""
    recs = []
    for pid, d in _season_field(counts, finals, season, snap, stat):
        fd = finals[(season, pid)]
        asof_stat, asof_min, asof_gp = sc(d), _mn(d), _gp(d)
        fin_stat, fin_min, fin_gp = sc(fd), _mn(fd), _gp(fd)
        rem_min = fin_min - asof_min
        if rem_min <= 0 or asof_min <= 0 or fin_gp < 1:
            continue
        banked_rate = asof_stat / asof_min
        proj_rem = banked_rate * rem_min
        rem_real = fin_stat - asof_stat
        drift = proj_rem - rem_real
        denom = max(fin_gp, qual)
        recs.append({
            "pid": pid, "asof_gp": asof_gp, "fin_gp": fin_gp,
            "banked_rate": banked_rate, "drift": drift,
            "eff_model": (asof_stat + proj_rem) / denom,
            "eff_oracle": fin_stat / denom,
            "stage": _stage(asof_gp / fin_gp),
        })
    return recs or None


def _leader(recs, key):
    return max(recs, key=lambda r: r[key])["pid"]


def probe_stat(conn, counts, finals, ftg_by_season, stat, seasons, show_season):
    sc = STAT_COUNT[stat]
    by_stage_drift = {"early": [], "mid": [], "late": []}
    by_stage_flip = {"early": [], "mid": [], "late": []}
    flips_close, flips_wide, races_close, races_wide = 0, 0, 0, 0
    het_close, het_wide = [], []
    small_drift, large_drift = [], []  # gp<20 vs >=20 banked, |drift|/rate

    for season in seasons:
        ftg = ftg_by_season[season]
        if not ftg:
            continue
        qual = math.ceil(QUAL_FRAC * max(ftg.values()))
        snaps = sorted({sn for (s, sn, _) in counts if s == season})
        for snap in snaps:
            recs = _race(counts, finals, season, snap, stat, qual, sc)
            if not recs or len(recs) < 2:
                continue
            oracle_sorted = sorted(recs, key=lambda r: r["eff_oracle"], reverse=True)
            top1, top2 = oracle_sorted[0]["eff_oracle"], oracle_sorted[1]["eff_oracle"]
            margin = (top1 - top2) / top1 if top1 > 0 else 1.0
            close = margin < 0.05
            flip = _leader(recs, "eff_model") != _leader(recs, "eff_oracle")

            for r in recs:
                by_stage_drift[r["stage"]].append(r["drift"])
                by_stage_flip[r["stage"]].append(0)
                nd = abs(r["drift"]) / (r["banked_rate"] or 1e-9)
                (small_drift if r["asof_gp"] < 20 else large_drift).append(nd)
            # stamp the flip onto the race's stage bucket (leader's stage)
            lead_stage = next(r["stage"] for r in recs if r["pid"] == _leader(recs, "eff_oracle"))
            by_stage_flip[lead_stage].append(1 if flip else 0)

            drifts = [r["drift"] for r in recs]
            rates = [r["banked_rate"] for r in recs if r["banked_rate"] > 0]
            het = (st.pstdev(drifts) / (sum(rates) / len(rates))) if len(drifts) > 1 and rates else 0.0
            if close:
                races_close += 1
                het_close.append(het)
                if flip:
                    flips_close += 1
            else:
                races_wide += 1
                het_wide.append(het)
                if flip:
                    flips_wide += 1

    tot_races = races_close + races_wide
    tot_flips = flips_close + flips_wide
    print("\n" + "=" * 66)
    print(f"stat={stat}  seasons={seasons[0]}-{seasons[-1]}  races={tot_races}")
    print("-" * 66)
    print("signed drift by stage (+ = over-projection):")
    for s in ("early", "mid", "late"):
        v = by_stage_drift[s]
        if v:
            over = 100.0 * sum(1 for x in v if x > 0) / len(v)
            print(f"  {s:<5}  mean {st.mean(v):+.3f}   over-proj {over:4.1f}%   n={len(v)}")
    print("leader-flip fraction (ceiling on any mu-forecast reorder):")
    if tot_races:
        print(f"  overall     {tot_flips/tot_races:5.1%}   ({tot_flips}/{tot_races})")
    if races_close:
        print(f"  close (<5%) {flips_close/races_close:5.1%}   ({flips_close}/{races_close})")
    if races_wide:
        print(f"  wide        {flips_wide/races_wide:5.1%}   ({flips_wide}/{races_wide})")
    if tot_flips:
        print(f"  share of flips in close races: {flips_close/tot_flips:5.1%}   "
              f"(the costly zone)")
    print("normalised within-race drift dispersion (heterogeneity => flips possible):")
    if het_close:
        print(f"  close races  {st.mean(het_close):.3f}")
    if het_wide:
        print(f"  wide races   {st.mean(het_wide):.3f}")
    if small_drift and large_drift:
        print("|drift|/rate by banked sample (is small-sample over-projection the driver):")
        print(f"  gp<20   {st.mean(small_drift):.3f}   n={len(small_drift)}")
        print(f"  gp>=20  {st.mean(large_drift):.3f}   n={len(large_drift)}")

    if show_season and show_season in seasons:
        ftg = ftg_by_season[show_season]
        if ftg:
            qual = math.ceil(QUAL_FRAC * max(ftg.values()))
            names = _names(conn)
            board = []
            for (s, pid), fd in finals.items():
                if s != show_season or _gp(fd) < 1 or pid not in ftg:
                    continue
                board.append((sc(fd) / max(_gp(fd), qual), pid))
            board.sort(reverse=True)
            print(f"\n  realised leaderboard {stat} {show_season}  (qual q={qual}):")
            for eff, pid in board[:10]:
                print(f"    {names.get(pid, pid)!s:<24} {eff:.3f}   (id={pid})")


def _parse_seasons(s):
    if "-" in s:
        lo, hi = s.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in s.split(",")]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/awards.db")
    ap.add_argument("--stat", default="all",
                    choices=["reb", "pts", "ast", "stl", "blk", "pra", "all"])
    ap.add_argument("--seasons", default="2013-2023")
    ap.add_argument("--show-season", type=int, default=None)
    ap.add_argument("--allow-sealed", action="store_true")
    a = ap.parse_args(argv)

    stats = {"all": ["pts", "reb", "ast", "stl", "blk"],
             "pra": ["pts", "reb", "ast"]}.get(a.stat, [a.stat])
    seasons = _parse_seasons(a.seasons)

    conn = connect(a.db)
    for stat in stats:
        yrs = [y for y in seasons if y >= STAT_FLOOR[stat]]
        bad = [y for y in yrs if y in SEALED[stat]]
        if bad and not a.allow_sealed:
            print(f"REFUSE stat={stat}: sealed seasons {bad} in range. "
                  f"Drop them or pass --allow-sealed for a non-price book only.")
            continue
        if not yrs:
            continue
        counts, finals, _, _ = N._load(conn, yrs)
        ftg_by_season = {y: MC._load_ftg(conn, y) for y in yrs}
        probe_stat(conn, counts, finals, ftg_by_season, stat, yrs, a.show_season)
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
