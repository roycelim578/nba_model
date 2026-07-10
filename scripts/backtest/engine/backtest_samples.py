"""
Joint-sample producer for the sized backtest. For each tradeable 2024 snapshot,
score the candidate set through the persisted K=200 boosters (reusing score_fold's
EXACT load/predict path: manifest feat_cols + assemble_matrix + factorise_groups),
add eta forward-noise (fit on <=2023 only; the seal), softmax within group, and
return the joint winner draws for the sizer.

Structural points:
  - eta drawn PER CANDIDATE independently (shared factor F dead), added to each
    booster score, then softmax the whole group per (booster, eta-draw).
  - eta scale = exp(slope*frac + inter), slope/inter fit on <=2023 OOF forward errors
    ONLY (walk-forward seal). df from per-award constants (ROTY 4.0 fixed).
  - S = K*M joint samples per snapshot; each names one winner (argmax).
  - The scoring path is byte-identical to score_fold, so per-candidate mean-score
    p_win reproduces model_predictions (the housekeeping identity proof holds).

British English. Reuses score_fold internals; does not reimplement scoring.
"""
from __future__ import annotations
import sys, glob, json
from dataclasses import dataclass
import numpy as np
from scipy import stats

sys.path.insert(0, "src")

DF_HI = {"DPOY": 12.0, "MVP": 12.0, "ROTY": 4.0}
DF_LO = {"DPOY": 6.0, "MVP": 4.0, "ROTY": 4.0}
STRIP = {"DPOY": False, "MVP": False, "ROTY": True}
TRUST_MAX_FRAC = 0.6

def _add_frac(df):
    pos = df[df["week_index"] >= 0]
    smax = pos.groupby("season")["week_index"].max().to_dict()
    wi = df["week_index"].to_numpy(float); sm = df["season"].map(smax).to_numpy(float)
    df = df.copy()
    df["frac"] = np.where(sm > 0, np.where(wi < 0, 0.0, wi/sm), 0.0)
    df["frac"] = df["frac"].clip(0, 1)
    return df

def _with_fwd(df):
    df = _add_frac(df)
    finals = df.sort_values("week_index").groupby(["season", "nba_api_id"]).tail(1)
    fs = finals.set_index(["season", "nba_api_id"])["stage1_score"]
    df["final_score"] = [fs.get(k, np.nan) for k in zip(df["season"], df["nba_api_id"])]
    df["fwd_err"] = df["stage1_score"] - df["final_score"]
    df["pop_var"] = df["raw_scores"].apply(lambda v: float(np.var(np.asarray(v), ddof=1)))
    df["epi_var_at50"] = df["pop_var"] / 50.0
    return df[df["fwd_err"].notna() & (df["frac"] < 1.0)].copy()

def _fit_slope_trust(df, strip):
    import pandas as pd
    bins = np.linspace(0, 1, 11); d = df.copy()
    d["fb"] = pd.cut(d["frac"], bins=bins, include_lowest=True)
    c, lg = [], []
    for iv, g in d.groupby("fb", observed=True):
        if iv.mid > TRUST_MAX_FRAC:
            continue
        x = g["fwd_err"].values
        if strip and len(x) > 20:
            cap = np.percentile(np.abs(x), 97.5); k = np.abs(x) <= cap
            tot = np.var(x[k]); epi = g["epi_var_at50"].values[k].mean()
        else:
            tot = np.var(x); epi = g["epi_var_at50"].values.mean()
        c.append(iv.mid); lg.append(0.5*np.log(max(tot-epi, 1e-6)))
    slope, inter = np.polyfit(np.array(c), np.array(lg), 1)
    return float(slope), float(inter)

@dataclass
class JointSamples:
    date: str
    player_ids: list
    winner_idx: np.ndarray
    p_win: np.ndarray
    vote_share_pred: np.ndarray
    raw_scores: np.ndarray
    frac: float
    sizing_weights: np.ndarray = None
    sim: np.ndarray = None

def build_samples(conn, award, season, snapshot_dates, oof_bundle_le2023,
                  final_model_path, M=200, seed=20260701):
    import lightgbm as lgb
    import pandas as pd
    from scripts.modelling.score.score_fold import assemble_matrix, factorise_groups, grouped_softmax

    rng = np.random.default_rng(seed)
    fwd = _with_fwd(oof_bundle_le2023)
    slope, inter = _fit_slope_trust(fwd, strip=STRIP[award])
    df_hi, df_lo = DF_HI[award], DF_LO[award]

    d = pd.read_pickle(final_model_path)
    booster_strings = d["boosters"]["booster_strings"]
    boosters = [lgb.Booster(model_str=s) for s in booster_strings]
    manifest = json.load(open(final_model_path + ".manifest.json"))
    if season in set(manifest.get("train_seasons", [])):
        raise SystemExit(f"SEAL: season {season} is in train_seasons; would be in-sample")
    feat_cols = list(manifest["features"])
    K = len(boosters)

    df = assemble_matrix(conn, award, feat_cols, seasons=[season], model_version=None)
    if df.empty:
        raise SystemExit(f"no scoring rows for {award} {season}")
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"missing trained cols: {missing[:8]}")
    df = _add_frac(df)

    out = {}
    for snap in snapshot_dates:
        g = df[df["snapshot_date"] == snap].sort_values("player_id", kind="mergesort").reset_index(drop=True)
        if len(g) < 2:
            continue
        X = g[feat_cols].to_numpy(float)
        ens = np.vstack([b.predict(X) for b in boosters])       # (K, ncand)
        R = ens.T                                               # (ncand, K)
        gids = factorise_groups(g)
        mu = ens.mean(axis=0)
        vsp = grouped_softmax(mu, gids)                         # matches model_predictions
        fr = float(np.clip(g["frac"].iloc[0], 0, 1))
        scale = np.exp(slope * fr + inter)
        dfree = max(df_hi + (df_lo - df_hi) * fr, 2.1)
        ncand = R.shape[0]
        eta = stats.t.rvs(dfree, size=(ncand, K, M), random_state=rng) * scale
        sim = (R[:, :, None] + eta).reshape(ncand, K * M)
        winners = np.asarray(sim).argmax(axis=0)
        pw = np.array([(winners == i).mean() for i in range(ncand)])
        sm = np.exp(sim - sim.max(axis=0))
        sm = sm / sm.sum(axis=0)
        q_size = sm.mean(axis=1)                                 # mean_of_softmax (sizing)
        out[snap] = JointSamples(
            date=snap, player_ids=g["player_id"].astype(int).tolist(),
            winner_idx=winners, p_win=pw, vote_share_pred=np.asarray(vsp),
            raw_scores=R, frac=fr, sizing_weights=q_size, sim=sim)
    return out

def _report(award="MVP", season=2024):
    import pandas as pd
    from scripts.common import db
    conn = db.connect("data/awards.db")
    ob = [p for p in sorted(glob.glob(f"out/oof_cache/oofraw_{award}_*.pkl"))
          if "_archive" not in p and "_backup" not in p][-1]
    bundle_le = pd.read_pickle(ob)
    bundle_le = bundle_le[bundle_le["season"] <= 2023].copy()
    fm = sorted(glob.glob(f"out/final_models/final_{award}_*K200*.pkl"))
    fm = [p for p in fm if not p.endswith(".json")][-1]
    from scripts.backtest.engine.backtest_pricejoin import load_tradeable
    fr = load_tradeable(conn, award, season)
    snaps = list(fr.keys())
    js = build_samples(conn, award, season, snaps, bundle_le, fm, M=200)
    print(f"{award} {season}: sampled {len(js)} snapshots (eta fit on <=2023)")
    for snap, s in list(js.items())[:6] + list(js.items())[-3:]:
        order = np.argsort(-s.p_win)[:3]
        top = ", ".join(f"{s.player_ids[i]}:{s.p_win[i]:.2f}(vs{s.vote_share_pred[i]:.2f})" for i in order)
        print(f"  {snap} frac={s.frac:.2f} n={len(s.player_ids)} S={s.winner_idx.size} | {top}")
    conn.close()

if __name__ == "__main__":
    _report()
