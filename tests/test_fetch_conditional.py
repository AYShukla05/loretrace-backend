import asyncio

import httpx
import pytest

from app.models.enums import SourceType
from app.models.source import Source
from app.scraping import robots
from app.scraping.fetch import (
    NotModifiedError,
    _extract_gutenberg_text,
    _rate_limiter,
    fetch_source_text,
)

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


GUTENBERG_BODY_WITH_FOOTNOTES = (
    "*** START OF THIS PROJECT GUTENBERG EBOOK ***\n"
    "The Autonoeian[26] hero took to flight, and {when} no voice followed\n"
    "he groaned.\n\n"
    "    [Footnote 26: _Autonoeian._--Ver. 198. Actaeon was the son of\n"
    "    Autonoe, the daughter of Cadmus.]\n\n"
    "    [Footnote 31: _Hyperion._--Ver. 192. He was the father of the\n"
    "    Sun.]\n\n"
    "And men call her the foam-born goddess.\n"
    "*** END OF THIS PROJECT GUTENBERG EBOOK ***"
)


def test_strips_gutenberg_footnote_blocks_and_inline_markers():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="")
        return httpx.Response(200, text=GUTENBERG_BODY_WITH_FOOTNOTES)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    source = make_source("https://example.com/ovid/book.txt")

    text = run(fetch_source_text(client, source)).text

    assert "[Footnote" not in text
    assert "[26]" not in text
    assert "Ver. 198" not in text
    # the translation itself, including its {clarifying} interpolations, stays
    assert "The Autonoeian hero took to flight" in text
    assert "{when} no voice followed" in text
    assert "And men call her the foam-born goddess." in text


def test_extract_gutenberg_text_drops_multiline_footnote_but_keeps_narrative():
    raw = (
        "*** START OF THE PROJECT GUTENBERG EBOOK ***\n"
        "Narrative line one.\n\n"
        "[Footnote 5: _Some place._--Ver. 12. A long note that runs\n"
        "across several lines and ends much later.]\n\n"
        "Narrative line two.\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK ***"
    )

    text = _extract_gutenberg_text(raw)

    assert text == "Narrative line one.\n\nNarrative line two."


def test_extract_gutenberg_text_drops_unnumbered_footnote_form():
    # Bulfinch's edition writes "[Footnote: ...]" with no number
    raw = (
        "*** START OF THE PROJECT GUTENBERG EBOOK ***\n"
        "Jupiter [Footnote: The names in parentheses are the Greek ones.]\n"
        "was called the father of gods and men.\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK ***"
    )

    assert _extract_gutenberg_text(raw) == ("Jupiter\nwas called the father of gods and men.")


def test_extract_gutenberg_text_is_a_no_op_without_footnotes():
    raw = (
        "*** START OF THE PROJECT GUTENBERG EBOOK ***\n"
        "Just a plain line.\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK ***"
    )

    assert _extract_gutenberg_text(raw) == "Just a plain line."


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


WIKISOURCE_INDEX_BODY = (
    "<html><body>"
    '<h1 id="firstHeading"><span>Nihongi</span></h1>'
    '<div id="mw-content-text">'
    "<p>Chronicles of Japan. Translated by W. G. Aston.</p>"
    "<ul>"
    '<li><a href="/wiki/Nihongi/Introduction">Introduction</a></li>'
    '<li><a href="/wiki/Nihongi/Book_I">Book I</a></li>'
    '<li><a href="/wiki/Nihongi/Book_II">Book II</a></li>'
    "</ul>"
    '<p>See <a href="/wiki/Author:William_George_Aston">the translator</a>, '
    '<a href="/wiki/Nihongi/Book_I#Age_of_the_Gods">a repeat link</a>, and '
    '<a href="/wiki/Nihongi/Book_I/Notes">a deeper page</a>.</p>'
    "</div></body></html>"
)


def _wiki_child_body(text: str) -> str:
    return f'<html><body><div id="mw-content-text"><p>{text}</p></div></body></html>'


def wikisource_index_handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    bodies = {
        "/robots.txt": "",
        "/wiki/Nihongi": WIKISOURCE_INDEX_BODY,
        "/wiki/Nihongi/Introduction": _wiki_child_body("Translator's introduction."),
        "/wiki/Nihongi/Book_I": _wiki_child_body("In the beginning, Izanagi."),
        "/wiki/Nihongi/Book_II": _wiki_child_body("The gods descended."),
        "/wiki/Nihongi/Book_I/Notes": _wiki_child_body("SHOULD NOT APPEAR."),
    }
    if path in bodies:
        return httpx.Response(200, text=bodies[path])
    return httpx.Response(404)


@pytest.fixture
def no_rate_limit():
    original = _rate_limiter._min_interval
    _rate_limiter._min_interval = 0.0
    _rate_limiter._last_request.clear()
    yield
    _rate_limiter._min_interval = original


def test_wikisource_index_walks_child_subpages_in_document_order(no_rate_limit):
    client = httpx.AsyncClient(transport=httpx.MockTransport(wikisource_index_handler))
    source = make_source(
        "https://en.wikisource.org/wiki/Nihongi", source_type=SourceType.WIKI_INDEX
    )

    result = run(fetch_source_text(client, source))

    assert result.text == (
        "Translator's introduction.\n\nIn the beginning, Izanagi.\n\nThe gods descended."
    )
    assert "SHOULD NOT APPEAR" not in result.text
    assert result.title == "Nihongi"


def test_wikisource_index_ignores_cached_validators(no_rate_limit):
    seen_headers = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path != "/robots.txt":
            seen_headers.append({k.lower() for k in request.headers})
        return wikisource_index_handler(request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    source = make_source(
        "https://en.wikisource.org/wiki/Nihongi",
        source_type=SourceType.WIKI_INDEX,
        etag='"abc123"',
        last_modified="Tue, 01 Jul 2026 00:00:00 GMT",
    )

    result = run(fetch_source_text(client, source))

    assert result.etag is None
    assert result.last_modified is None
    assert all("if-none-match" not in headers for headers in seen_headers)
    assert all("if-modified-since" not in headers for headers in seen_headers)


def test_wikisource_index_raises_when_no_child_subpages(no_rate_limit):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="")
        return httpx.Response(
            200,
            text=(
                '<html><body><h1 id="firstHeading">Empty</h1>'
                '<div id="mw-content-text"><p>No subpage links here.</p></div></body></html>'
            ),
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    source = make_source(
        "https://en.wikisource.org/wiki/Empty", source_type=SourceType.WIKI_INDEX
    )

    with pytest.raises(ValueError):
        run(fetch_source_text(client, source))


def test_strips_mediawiki_reference_list_from_content():
    body = (
        '<html><body><h1 id="firstHeading">Section 1</h1>'
        '<div id="mw-content-text">'
        '<p>The narrative text.<sup class="reference">[1]</sup></p>'
        '<ol class="references"><li>A long philological footnote citing Motowori.</li></ol>'
        "</div></body></html>"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="")
        return httpx.Response(200, text=body)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    source = make_source("https://en.wikisource.org/wiki/Page", source_type=SourceType.WIKISOURCE)

    result = run(fetch_source_text(client, source))

    assert result.text == "The narrative text."
    assert "philological footnote" not in result.text
