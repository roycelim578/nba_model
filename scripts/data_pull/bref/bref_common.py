"""Shared Basketball-Reference fetch layer for the bref scrapers.

Responsibilities:
- Hard caching of raw HTML to ``data/cache/bref/`` (gitignored). bref is
  rate-strict and will block aggressive scraping; for dev iteration we must
  never re-hit the network for a page we already have.
- Polite rate limiting (~1 request / 3 seconds) on actual network fetches.
  Cache hits are not rate limited.
- ``tenacity`` retries with exponential backoff on transient failures.
- Unwrapping bref's commented-out tables. bref hides many stat tables inside
  HTML comments (``<!-- ... <table> ... -->``) as an anti-scrape / lazy-render
  measure; a naive BeautifulSoup parse misses them. We strip the comment
  markers before parsing so both the commented and uncommented layouts work.

This module is bref-specific and owned by Chat B. It does NOT touch the DB.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

# bref sits behind Cloudflare, which blocks on TLS/JA3 fingerprint, not just
# headers. Python's `requests` has a TLS handshake signature Cloudflare
# recognises as non-browser and 403s regardless of how perfect the headers are.
# curl_cffi performs the request with a real Chrome TLS fingerprint
# (impersonate="chrome"), which is what actually clears the block. Its API is a
# drop-in for requests (Session, .get, .status_code, .text), so nothing else in
# this module changes.
from curl_cffi import requests as cffi_requests
from curl_cffi.requests.exceptions import RequestException as CffiRequestException
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

BASE_URL = "https://www.basketball-reference.com"
CACHE_DIR = Path("data/cache/bref")

# Full browser header set. With curl_cffi's impersonate the TLS layer already
# matches Chrome; these headers keep the application layer consistent with it.
_HEADERS = {
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,image/apng,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
}

# Persistent session with Chrome impersonation: matches Chrome's TLS/JA3
# fingerprint AND its default header order, retains Cloudflare cookies across
# requests. impersonate is the load-bearing argument here, not the headers.
_SESSION = cffi_requests.Session(impersonate="chrome", headers=_HEADERS)

_MIN_INTERVAL_S = 3.0  # ~1 request / 3 seconds on network fetches
_last_fetch_ts: float = 0.0


class FetchError(RuntimeError):
    """Raised when a page cannot be fetched after retries."""


def _cache_path(slug: str) -> Path:
    """Map a page slug (e.g. 'awards/awards_2025') to a cache file path."""
    safe = slug.strip("/").replace("/", "__")
    return CACHE_DIR / f"{safe}.html"


def _respect_rate_limit() -> None:
    global _last_fetch_ts
    elapsed = time.monotonic() - _last_fetch_ts
    if elapsed < _MIN_INTERVAL_S:
        time.sleep(_MIN_INTERVAL_S - elapsed)
    _last_fetch_ts = time.monotonic()


@retry(
    retry=retry_if_exception_type((CffiRequestException, FetchError)),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    reraise=True,
)
def _network_get(url: str) -> str:
    _respect_rate_limit()
    resp = _SESSION.get(url, timeout=30)
    # 429/5xx are transient; raise so tenacity retries. 404 is NOT retried
    # (the page genuinely does not exist, e.g. a below-floor award season).
    if resp.status_code == 404:
        raise FileNotFoundError(f"404 Not Found: {url}")
    if resp.status_code >= 400:
        raise FetchError(f"HTTP {resp.status_code} for {url}")
    return resp.text


def fetch_html(slug: str, *, force_refresh: bool = False) -> str:
    """Return raw HTML for a bref page slug, using the on-disk cache.

    ``slug`` is the path under the bref domain WITHOUT leading slash or
    ``.html``, e.g. ``"awards/awards_2025"`` or ``"players/j/jokicni01"``.

    Raises ``FileNotFoundError`` (un-retried) if the page 404s, so callers can
    distinguish a genuinely absent page (below-floor season) from a transient
    failure. Caches successful fetches hard.
    """
    cache_file = _cache_path(slug)
    if cache_file.exists() and not force_refresh:
        return cache_file.read_text(encoding="utf-8")

    url = f"{BASE_URL}/{slug.strip('/')}.html"
    html = _network_get(url)  # may raise FileNotFoundError on 404

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(html, encoding="utf-8")
    return html


# bref wraps many tables in HTML comments. Stripping the comment delimiters
# exposes the inner <table> to the parser. We only strip comment markers that
# appear to wrap a table, to avoid disturbing genuine comments.
_COMMENT_RE = re.compile(r"<!--(.*?)-->", re.DOTALL)


def uncomment_tables(html: str) -> str:
    """Return HTML with comment-wrapped tables exposed for parsing.

    Replaces any ``<!-- ... -->`` block that contains a ``<table`` with its
    inner content. Leaves non-table comments intact.
    """

    def _sub(match: re.Match[str]) -> str:
        inner = match.group(1)
        return inner if "<table" in inner else match.group(0)

    return _COMMENT_RE.sub(_sub, html)