import re
from typing import NamedTuple
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup

from app.models.enums import SourceType
from app.models.source import Source
from app.scraping.rate_limit import DomainRateLimiter
from app.scraping.robots import can_fetch

USER_AGENT = "LoreTraceBot/0.1 (+https://loretrace.pages.dev)"

GUTENBERG_START_MARKER = "*** START OF"
GUTENBERG_END_MARKER = "*** END OF"
GUTENBERG_TEXT_SUFFIXES = (".txt", ".txt.utf-8")
GUTENBERG_TITLE_PATTERN = re.compile(r"^Title:\s*(.+)$", re.MULTILINE)

_rate_limiter = DomainRateLimiter(min_interval_seconds=2.0)


class RobotsDisallowedError(Exception):
    pass


class NotModifiedError(Exception):
    """Server confirmed the source is unchanged since the last successful fetch."""


class FetchResult(NamedTuple):
    text: str
    etag: str | None
    last_modified: str | None
    title: str | None


async def fetch_source_text(client: httpx.AsyncClient, source: Source) -> FetchResult:
    if source.source_type not in (
        SourceType.GUTENBERG_TEXT,
        SourceType.WIKISOURCE,
        SourceType.WIKIPEDIA,
    ):
        raise NotImplementedError(f"no scraper for source_type={source.source_type}")

    url = source.url
    if source.source_type == SourceType.GUTENBERG_TEXT:
        url = await _resolve_gutenberg_text_url(client, url)

    if not await can_fetch(client, url, USER_AGENT):
        raise RobotsDisallowedError(f"robots.txt disallows fetching {url}")

    await _rate_limiter.wait(url)

    headers = {"User-Agent": USER_AGENT}
    if source.etag:
        headers["If-None-Match"] = source.etag
    if source.last_modified:
        headers["If-Modified-Since"] = source.last_modified

    response = await client.get(url, headers=headers, timeout=30, follow_redirects=True)
    if response.status_code == 304:
        raise NotModifiedError(f"{source.url} unchanged since last fetch")
    response.raise_for_status()

    if source.source_type == SourceType.GUTENBERG_TEXT:
        text = _extract_gutenberg_text(response.text)
        title = _extract_gutenberg_title(response.text)
    else:
        text = _extract_mediawiki_text(response.text)
        title = _extract_mediawiki_title(response.text)

    return FetchResult(
        text=text,
        etag=response.headers.get("ETag"),
        last_modified=response.headers.get("Last-Modified"),
        title=title,
    )


async def _resolve_gutenberg_text_url(client: httpx.AsyncClient, url: str) -> str:
    """Resolve a Gutenberg ebook page (e.g. /ebooks/{id}) to its plain-text download link.

    Gutenberg's /ebooks/{id} page is a catalog/landing page, not the book text,
    and actively discourages being scraped directly. If the given url isn't
    already a direct text link, fetch the catalog page and follow the actual
    "Plain Text" download link it advertises rather than guessing a filename
    pattern that may not hold for every book.
    """
    if url.rstrip("/").endswith(GUTENBERG_TEXT_SUFFIXES):
        return url

    if not await can_fetch(client, url, USER_AGENT):
        raise RobotsDisallowedError(f"robots.txt disallows fetching {url}")
    await _rate_limiter.wait(url)

    response = await client.get(
        url, headers={"User-Agent": USER_AGENT}, timeout=15, follow_redirects=True
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    link = soup.find("a", href=lambda href: href and ".txt" in href)
    if link is None:
        raise ValueError(f"could not find a plain text download link on {url}")
    return urljoin(str(response.url), link["href"])


def _extract_gutenberg_text(raw: str) -> str:
    start = raw.find(GUTENBERG_START_MARKER)
    start = raw.find("\n", start) + 1 if start != -1 else 0

    end = raw.find(GUTENBERG_END_MARKER)
    if end == -1:
        end = len(raw)

    return raw[start:end].strip()


def _extract_gutenberg_title(raw: str) -> str | None:
    """Parse the "Title:" line from Gutenberg's own standard metadata header
    (before *** START OF ***), rather than the noisier catalog-page <title>
    tag or a guessed filename pattern. Only the title's first line is kept —
    a wrapped subtitle continuation line, if present, is dropped in favor of
    a short, citable name."""
    header_end = raw.find(GUTENBERG_START_MARKER)
    header = raw if header_end == -1 else raw[:header_end]
    match = GUTENBERG_TITLE_PATTERN.search(header)
    if match is None:
        return None
    title = match.group(1).strip()
    return title or None


def _extract_mediawiki_title(html: str) -> str | None:
    soup = BeautifulSoup(html, "html.parser")
    heading = soup.select_one("#firstHeading")
    if heading is None:
        return None
    title = heading.get_text(strip=True)
    return title or None


def _extract_mediawiki_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    content = soup.select_one("#mw-content-text")
    if content is None:
        raise ValueError("could not find MediaWiki content div")

    noise_selector = (
        "style, script, table, sup.reference, "
        ".navbox, .mw-editsection, .ws-noexport, .noprint, .printfooter"
    )
    for tag in content.select(noise_selector):
        tag.decompose()

    lines = (line.strip() for line in content.get_text(separator="\n").splitlines())
    return "\n".join(line for line in lines if line)
