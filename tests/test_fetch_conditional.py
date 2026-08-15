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

GUTENBERG_BODY_WITH_TITLE = (
    "The Project Gutenberg eBook of The Poetic Edda\n\n"
    "Title: The Poetic Edda\n"
    "        Translated from the Icelandic with an introduction and notes\n\n"
    "Author: Various\n\n" + GUTENBERG_BODY
)

MEDIAWIKI_BODY = (
    "<html><body>"
    '<h1 id="firstHeading"><span lang="en"><span>Prose Edda</span></span></h1>'
    '<div id="mw-content-text"><p>Once upon a time.</p></div>'
    "</body></html>"
)


def make_source(url: str, **overrides) -> Source:
    defaults = {"source_type": SourceType.GUTENBERG_TEXT, "tradition": "greek", "added_by": 1}
    defaults.update(overrides)
    return Source(url=url, **defaults)


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


def test_resolves_gutenberg_catalog_page_to_its_text_link():
    catalog_html = (
        '<html><body><a href="/ebooks/999.txt.utf-8">Plain Text (accessible)</a>' "</body></html>"
    )
    requested_paths = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_paths.append(request.url.path)
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="")
        if request.url.path == "/ebooks/999":
            return httpx.Response(200, text=catalog_html)
        if request.url.path == "/ebooks/999.txt.utf-8":
            return httpx.Response(200, text=GUTENBERG_BODY)
        return httpx.Response(404)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    source = make_source("https://example.com/ebooks/999")

    result = run(fetch_source_text(client, source))

    assert "Once upon a time." in result.text
    assert "/ebooks/999.txt.utf-8" in requested_paths


def test_extracts_title_from_gutenberg_metadata_header():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="")
        return httpx.Response(200, text=GUTENBERG_BODY_WITH_TITLE)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    source = make_source("https://example.com/six/book.txt")

    result = run(fetch_source_text(client, source))

    assert result.title == "The Poetic Edda"


def test_gutenberg_title_is_none_without_a_title_line():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="")
        return httpx.Response(200, text=GUTENBERG_BODY)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    source = make_source("https://example.com/seven/book.txt")

    result = run(fetch_source_text(client, source))

    assert result.title is None


def test_extracts_title_from_mediawiki_first_heading():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="")
        return httpx.Response(200, text=MEDIAWIKI_BODY)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    source = make_source("https://example.com/eight/page", source_type=SourceType.WIKISOURCE)

    result = run(fetch_source_text(client, source))

    assert result.title == "Prose Edda"


def test_gutenberg_direct_text_url_skips_catalog_resolution():
    requested_paths = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_paths.append(request.url.path)
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="")
        return httpx.Response(200, text=GUTENBERG_BODY)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    source = make_source("https://example.com/five/book.txt")

    run(fetch_source_text(client, source))

    assert "/five/book.txt" in requested_paths
    assert "/ebooks" not in "".join(requested_paths)
