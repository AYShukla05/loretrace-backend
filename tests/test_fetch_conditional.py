import asyncio

import httpx
import pytest

from app.models.enums import SourceType
from app.models.source import Source
from app.scraping import robots
from app.scraping.fetch import NotModifiedError, fetch_source_text

GUTENBERG_BODY = (
    "*** START OF THIS PROJECT GUTENBERG EBOOK ***\n"
    "Once upon a time.\n"
    "*** END OF THIS PROJECT GUTENBERG EBOOK ***"
)


def make_source(url: str, **overrides) -> Source:
    return Source(
        url=url,
        source_type=SourceType.GUTENBERG_TEXT,
        tradition="greek",
        added_by=1,
        **overrides,
    )


def run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def clear_robots_cache():
    robots._robots_cache.clear()
    yield
    robots._robots_cache.clear()


def test_sends_conditional_headers_when_source_has_cached_validators():
    seen_headers = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="")
        seen_headers.update(request.headers)
        return httpx.Response(200, text=GUTENBERG_BODY)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    source = make_source(
        "https://example.com/one/book.txt",
        etag='"abc123"',
        last_modified="Tue, 01 Jul 2026 00:00:00 GMT",
    )

    run(fetch_source_text(client, source))

    assert seen_headers["if-none-match"] == '"abc123"'
    assert seen_headers["if-modified-since"] == "Tue, 01 Jul 2026 00:00:00 GMT"


def test_omits_conditional_headers_when_source_has_no_cached_validators():
    seen_headers = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="")
        seen_headers.update(request.headers)
        return httpx.Response(200, text=GUTENBERG_BODY)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    source = make_source("https://example.com/two/book.txt")

    run(fetch_source_text(client, source))

    assert "if-none-match" not in seen_headers
    assert "if-modified-since" not in seen_headers


def test_raises_not_modified_on_304():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="")
        return httpx.Response(304)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    source = make_source(
        "https://example.com/three/book.txt",
        etag='"abc123"',
        last_modified="Tue, 01 Jul 2026 00:00:00 GMT",
    )

    with pytest.raises(NotModifiedError):
        run(fetch_source_text(client, source))


def test_returns_new_validators_from_response_on_200():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="")
        return httpx.Response(
            200,
            text=GUTENBERG_BODY,
            headers={"ETag": '"newetag"', "Last-Modified": "Wed, 02 Jul 2026 00:00:00 GMT"},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    source = make_source("https://example.com/four/book.txt")

    result = run(fetch_source_text(client, source))

    assert result.etag == '"newetag"'
    assert result.last_modified == "Wed, 02 Jul 2026 00:00:00 GMT"
    assert "Once upon a time." in result.text
