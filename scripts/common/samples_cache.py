"""Content-addressed cache for build_samples output. Key = pinned artefact paths
and mtimes, OOF cap, M, seed, DB mtime, the snap set, and the fingerprint of every
scoring-path source file. A change to any of these re-scores; nothing else does.
British English."""
from __future__ import annotations
import hashlib, os, pickle, pathlib

CACHE_DIR = pathlib.Path("out/_samples_cache")

_SCORE_PATHS = [
    "scripts/features",
    "scripts/modelling/score",
    "scripts/backtest/engine/backtest_samples.py",
]


def _mtime(p):
    try:
        return int(os.stat(p).st_mtime)
    except OSError:
        return 0


def _code_fingerprint():
    items = []
    for root in _SCORE_PATHS:
        rp = pathlib.Path(root)
        if rp.is_file():
            items.append((str(rp), _mtime(rp)))
        elif rp.is_dir():
            for f in sorted(rp.rglob("*.py")):
                items.append((str(f), _mtime(f)))
    return items


def _key(award, season, snaps, fm, oof_path, cap, M, seed):
    h = hashlib.sha1()
    parts = [
        award, str(season), fm, str(_mtime(fm)), oof_path, str(_mtime(oof_path)),
        str(cap), str(M), str(seed),
        repr(sorted(str(s) for s in snaps)), repr(_code_fingerprint()),
    ]
    h.update("|".join(parts).encode())
    return h.hexdigest()[:16]


def load_or_build(award, season, snaps, fm, oof_path, cap, M, seed, builder):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = _key(award, season, snaps, fm, oof_path, cap, M, seed)
    path = CACHE_DIR / f"{award}_{season}_{key}.pkl"
    if path.exists():
        print(f"[cache] HIT  {award} {season}")
        with open(path, "rb") as fh:
            return pickle.load(fh)
    print(f"[cache] MISS {award} {season} (scoring)")
    result = builder()
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(result, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    return result
