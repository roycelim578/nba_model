"""Lever ablation diagnostic for the sizer / no-trade-band redesign.

Runs the three levers in every on/off combination on the burned 2024 development
season, for the voter arm (MVP DPOY ROTY) and the stat arm (PTS REB AST), reads
each run's pooled result, and decomposes the PnL and risk-adjusted impact of each
lever plus their interactions. Combo 000 is the control (all levers off) and must
reproduce the stored 2024 control from the previous chat; the table surfaces it so
the match can be checked at a glance.

The levers are toggled by environment variable, read by the orchestrator:
  bit 0  carry   CARRY_R=0.02          opportunity-cost hurdle in the sizer
  bit 1  conc    CONC_FORM=saturating  concave concentration cap
  bit 2  band    REGION_BAND_MODE=psig price-space no-trade band

Each cell is an independent single-pass subprocess (its own internal book pool),
writing its own out/ directory, so trade logs, position logs and model_eval are
kept per cell. Cells are fanned across processes up to --max-parallel; each cell
uses up to three cores internally, so keep max-parallel * 3 at or below the core
count to avoid oversubscription.

The gate is NOT run here. Run it separately as a bit-identical confirmation of the
levers-off path once the results are in hand.

Usage from repo root:
  caffeinate -i uv run python diagnose_levers.py --max-parallel 4
  uv run python diagnose_levers.py --dry-run            # print the job matrix only
  uv run python diagnose_levers.py --combos 000,100,010,001,111   # screen design

British English.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as _dt
import json
import math
import os
import subprocess
import sys
from pathlib import Path

ARMS = {"voter": ["MVP", "DPOY", "ROTY"], "stat": ["PTS", "REB", "AST"]}

VOTER_FOLD_2024 = {
    "MVP": "models/folds/MVP_monotone_2024.pkl",
    "DPOY": "models/folds/DPOY_monotone_2024.pkl",
    "ROTY": "models/folds/ROTY_vs_2024.pkl",
}

LEVER_ENV = {
    0: ("carry", {"CARRY_R": "0.02"}, {"CARRY_R": "0.0"}),
    1: ("conc", {"CONC_FORM": "saturating"}, {"CONC_FORM": "hump"}),
    2: ("band", {"REGION_BAND_MODE": "psig"}, {"REGION_BAND_MODE": "curvature"}),
}

BASE_REGION_ENV = {
    "USE_REGION": "1",
    "REGION_CONFIRM": "2",
    "REGION_HYST": "1.0",
    "REGION_BAND_FLOOR": "0.02",
    "ASYM_TRIM": "1",
}

FACTORIAL = ["000", "100", "010", "001", "110", "101", "011", "111"]


def combo_env(combo):
    env = {}
    for bit, (_name, on, off) in LEVER_ENV.items():
        env.update(on if combo[bit] == "1" else off)
    return env


def combo_label(combo):
    on = [LEVER_ENV[b][0] for b in range(3) if combo[b] == "1"]
    return "+".join(on) if on else "control"


def build_jobs(arms, combos, season, out_root):
    jobs = []
    for arm in arms:
        for combo in combos:
            outdir = out_root / f"{arm}_{combo}_{season}"
            jobs.append(dict(arm=arm, combo=combo, awards=ARMS[arm], outdir=outdir))
    return jobs


def run_cell(job, root, season, budget, python, force=False):
    outdir = job["outdir"]
    if not force and (outdir / f"pooled_{season}.json").exists():
        return job, 0, True
    outdir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    for k, v in BASE_REGION_ENV.items():
        env.setdefault(k, v)
    env.update(combo_env(job["combo"]))
    env["ALLOC_BUDGETS_JSON"] = json.dumps({aw: float(budget) for aw in job["awards"]})
    if job["arm"] == "voter" and int(season) == 2024:
        env["FINAL_MODEL_OVERRIDE_JSON"] = json.dumps(
            {aw: VOTER_FOLD_2024[aw] for aw in job["awards"]})
    cmd = [python, "-m", "scripts.backtest.engine.backtest_singlepass",
           "--season", str(season), "--awards", *job["awards"],
           "--budget", str(budget), "--out", str(outdir)]
    log = outdir / "run.log"
    with open(log, "w") as fh:
        proc = subprocess.run(cmd, cwd=str(root), env=env, stdout=fh,
                              stderr=subprocess.STDOUT, text=True)
    return job, proc.returncode, False


def read_metrics(outdir, season):
    pooled_p = outdir / f"pooled_{season}.json"
    if not pooled_p.exists():
        return None
    pooled = json.loads(pooled_p.read_text())
    mtm = pooled.get("mtm", {}) or {}
    m = dict(
        pnl=float(pooled.get("realised_pnl_total", float("nan"))),
        n_txn=int(pooled.get("n_transactions_total", 0)),
        sharpe=float(mtm.get("sharpe", float("nan"))),
        sortino=float(mtm.get("sortino", float("nan"))),
        maxdd=float(mtm.get("max_drawdown_pct", float("nan"))),
        ret_pct=float(pooled.get("return_pct_static", float("nan"))),
    )
    bs_p = outdir / f"book_summary_{season}.json"
    if bs_p.exists():
        books = json.loads(bs_p.read_text())
        m["per_book"] = {b.get("award", b.get("book", "?")):
                         round(float(b.get("realised_pnl", float("nan"))), 1) for b in books}
    return m


def _fmt(x, nd=1):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "  n/a"
    return f"{x:.{nd}f}"


def print_arm_table(arm, combos, results):
    print(f"\n=== {arm} arm, 2024 ===")
    hdr = f"{'combo':>6} {'levers':>16} {'PnL':>9} {'ret%':>7} {'Sharpe':>7} {'Sortino':>8} {'maxDD%':>8} {'nTxn':>6}"
    print(hdr)
    print("-" * len(hdr))
    for combo in combos:
        m = results.get((arm, combo))
        if m is None:
            print(f"{combo:>6} {combo_label(combo):>16} {'FAILED (see run.log)':>40}")
            continue
        print(f"{combo:>6} {combo_label(combo):>16} {_fmt(m['pnl']):>9} {_fmt(m['ret_pct']):>7} "
              f"{_fmt(m['sharpe'],3):>7} {_fmt(m['sortino'],3):>8} {_fmt(m['maxdd']):>8} {m['n_txn']:>6}")


def decompose(arm, combos, results, metric):
    def val(combo):
        m = results.get((arm, combo))
        return None if m is None else m[metric]
    have = {c: val(c) for c in combos if val(c) is not None}
    out = {}
    base = have.get("000")
    if base is not None:
        for bit, (name, _on, _off) in LEVER_ENV.items():
            single = "".join("1" if i == bit else "0" for i in range(3))
            if single in have:
                out[f"{name} alone vs control"] = have[single] - base
        if "111" in have:
            out["all three vs control"] = have["111"] - base
            singles = [have.get("".join("1" if i == bit else "0" for i in range(3)))
                       for bit in range(3)]
            if all(s is not None for s in singles):
                out["interaction residual"] = (have["111"] - base) - sum(s - base for s in singles)
    if len(have) == len(FACTORIAL):
        for bit, (name, _on, _off) in LEVER_ENV.items():
            on = [have[c] for c in FACTORIAL if c[bit] == "1"]
            off = [have[c] for c in FACTORIAL if c[bit] == "0"]
            out[f"{name} main effect (factorial)"] = sum(on) / len(on) - sum(off) / len(off)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Lever ablation diagnostic (2024).")
    ap.add_argument("--root", default=".", help="repo root")
    ap.add_argument("--season", type=int, default=2024)
    ap.add_argument("--budget", type=float, default=1000.0)
    ap.add_argument("--arms", default="voter,stat")
    ap.add_argument("--combos", default=",".join(FACTORIAL),
                    help="comma list of 3-bit codes [carry,conc,band]")
    ap.add_argument("--out", default="out/lever_diag")
    ap.add_argument("--max-parallel", type=int, default=4)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--force", action="store_true",
                    help="recompute cells even if pooled json already exists")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    root = Path(args.root).expanduser().resolve()
    out_root = (root / args.out) if not os.path.isabs(args.out) else Path(args.out)
    arms = [a for a in args.arms.split(",") if a in ARMS]
    combos = [c.strip() for c in args.combos.split(",") if c.strip()]
    bad = [c for c in combos if len(c) != 3 or any(ch not in "01" for ch in c)]
    if bad:
        print(f"bad combo codes: {bad}")
        return 1

    jobs = build_jobs(arms, combos, args.season, out_root)
    print(f"root={root}")
    print(f"cells={len(jobs)} ({len(arms)} arms x {len(combos)} combos)  "
          f"max_parallel={args.max_parallel}  out={out_root}")
    for j in jobs:
        e = combo_env(j["combo"])
        print(f"  {j['arm']:>5} {j['combo']} [{combo_label(j['combo']):>16}]  "
              f"{' '.join(f'{k}={v}' for k, v in sorted(e.items()))}")
    if args.dry_run:
        print("\ndry run only; nothing launched")
        return 0

    t0 = _dt.datetime.now()
    failures = []
    with cf.ThreadPoolExecutor(max_workers=args.max_parallel) as ex:
        futs = {ex.submit(run_cell, j, root, args.season, args.budget, args.python,
                          args.force): j for j in jobs}
        for fut in cf.as_completed(futs):
            j = futs[fut]
            skipped = False
            try:
                _job, rc, skipped = fut.result()
            except Exception as exc:  # noqa: BLE001
                rc, exc_txt = 1, repr(exc)
            else:
                exc_txt = ""
            tag = f"{j['arm']}_{j['combo']}"
            if skipped:
                print(f"skip  {tag} (pooled json present; --force to rerun)")
            elif rc == 0:
                print(f"done  {tag}")
            else:
                print(f"FAIL  {tag} rc={rc} {exc_txt}")
                failures.append(tag)

    results = {}
    for j in jobs:
        m = read_metrics(j["outdir"], args.season)
        if m is not None:
            results[(j["arm"], j["combo"])] = m

    for arm in arms:
        print_arm_table(arm, combos, results)
        dec = decompose(arm, combos, results, "pnl")
        decs = decompose(arm, combos, results, "sharpe")
        print(f"  -- {arm} PnL decomposition --")
        for k, v in dec.items():
            print(f"     {k:<34} {v:+.1f}")
        print(f"  -- {arm} Sharpe decomposition --")
        for k, v in decs.items():
            print(f"     {k:<34} {v:+.3f}")

    summary = dict(
        season=args.season, arms=arms, combos=combos,
        generated=_dt.datetime.now().isoformat(timespec="seconds"),
        base_env=BASE_REGION_ENV, budget=args.budget,
        cells={f"{a}_{c}": results.get((a, c)) for a in arms for c in combos},
        pnl_decomposition={a: decompose(a, combos, results, "pnl") for a in arms},
        sharpe_decomposition={a: decompose(a, combos, results, "sharpe") for a in arms},
        failures=failures,
    )
    sp = out_root / f"summary_{args.season}.json"
    sp.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nsummary -> {sp}")
    print(f"trade logs kept per cell at {out_root}/<arm>_<combo>_{args.season}/trade_log_{args.season}.csv")
    print(f"elapsed {(_dt.datetime.now() - t0)}")
    if failures:
        print(f"\n{len(failures)} cell(s) failed: {failures}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
