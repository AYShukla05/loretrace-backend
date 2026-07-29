from typing import NamedTuple

import httpx
from bs4 import BeautifulSoup

from app.models.enums import SourceType
from app.models.source import Source
from app.scraping.rate_limit import DomainRateLimiter
from app.scraping.robots import can_fetch

USER_AGENT = "LoreTraceBot/0.1 (+https://loretrace.pages.dev)"

GUTENBERG_START_MARKER = "*** START OF"
GUTENBERG_END_MARKER = "*** END OF"

_rate_limiter = DomainRateLimiter(min_interval_seconds=2.0)


class RobotsDisallowedError(Exception):
    pass


class NotModifiedError(Exception):
    """Server confirmed the source is unchanged since the last successful fetch."""


class FetchResult(NamedTuple):
    text: str
    etag: str | None
    last_modified: str | None


async def fetch_source_text(client: httpx.AsyncClient, source: Source) -> FetchResult:
    if source.source_type not in (
        SourceType.GUTENBERG_TEXT,
        SourceType.WIKISOURCE,
        SourceType.WIKIPEDIA,
    ):
        raise NotImplementedError(f"no scraper for source_type={source.source_type}")

    if not await can_fetch(client, source.url, USER_AGENT):
        raise RobotsDisallowedError(f"robots.txt disallows fetching {source.url}")

    await _rate_limiter.wait(source.url)

    headers = {"User-Agent": USER_AGENT}
    if source.etag:
        headers["If-None-Match"] = source.etag
    if source.last_modified:
        headers["If-Modified-Since"] = source.last_modified

    response = await client.get(source.url, headers=headers, timeout=30)
    if response.status_code == 304:
        raise NotModifiedError(f"{source.url} unchanged since last fetch")
    response.raise_for_status()

    if source.source_type == SourceType.GUTENBERG_TEXT:
        text = _extract_gutenberg_text(response.text)
    else:
        text = _extract_mediawiki_text(response.text)

    return FetchResult(
        text=text,
        etag=response.headers.get("ETag"),
        last_modified=response.headers.get("Last-Modified"),
    )


def _extract_gutenberg_text(raw: str) -> str:
    start = raw.find(GUTENBERG_START_MARKER)
    start = raw.find("\n", start) + 1 if start != -1 else 0

    end = raw.find(GUTENBERG_END_MARKER)
    if end == -1:
        end = len(raw)

    return raw[start:end].strip()


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
