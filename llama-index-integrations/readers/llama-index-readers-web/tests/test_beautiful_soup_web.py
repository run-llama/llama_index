import requests
from bs4 import BeautifulSoup

from llama_index.readers.web.beautiful_soup_web.base import (
    _gitbook_reader,
    _readthedocs_reader,
    _substack_reader,
)

SUBSTACK_HTML = """<html><body>
<h1 class="post-title">My Post</h1>
<h3 class="subtitle">A subtitle</h3>
<span class="byline-names">Jane Doe</span>
<div class="available-content"><p>Body text.</p></div>
</body></html>"""


class _FakeResponse:
    def __init__(self, text):
        self.text = text


def test_substack_reader_full_markup() -> None:
    soup = BeautifulSoup(SUBSTACK_HTML, "html.parser")
    text, extra_info = _substack_reader(soup)
    assert "Body text." in text
    assert extra_info["Title of this Substack post"] == "My Post"
    assert extra_info["Subtitle"] == "A subtitle"
    assert extra_info["Author"] == "Jane Doe"


def test_substack_reader_without_subtitle() -> None:
    html = SUBSTACK_HTML.replace('<h3 class="subtitle">A subtitle</h3>', "")
    soup = BeautifulSoup(html, "html.parser")
    text, extra_info = _substack_reader(soup)
    assert "Body text." in text
    assert "Subtitle" not in extra_info
    assert extra_info["Title of this Substack post"] == "My Post"


def test_substack_reader_without_content_container_falls_back_to_page_text() -> None:
    html = "<html><body><h1 class='post-title'>T</h1><p>Loose text.</p></body></html>"
    soup = BeautifulSoup(html, "html.parser")
    text, extra_info = _substack_reader(soup)
    assert "Loose text." in text


def test_readthedocs_reader_extracts_main(monkeypatch) -> None:
    monkeypatch.setattr(
        requests,
        "get",
        lambda *a, **k: _FakeResponse(
            '<html><body><div role="main"><p>Doc body.</p></div></body></html>'
        ),
    )
    soup = BeautifulSoup(
        '<html><body><a class="reference internal" href="http://example.com/page">p</a></body></html>',
        "html.parser",
    )
    text, _ = _readthedocs_reader(soup, "http://example.com/")
    assert "Doc body." in text


def test_readthedocs_reader_skips_page_without_main(monkeypatch) -> None:
    monkeypatch.setattr(
        requests,
        "get",
        lambda *a, **k: _FakeResponse("<html><body><div>no main</div></body></html>"),
    )
    soup = BeautifulSoup(
        '<html><body><a class="reference internal" href="http://example.com/page">p</a></body></html>',
        "html.parser",
    )
    text, _ = _readthedocs_reader(soup, "http://example.com/")
    assert text == ""


def test_gitbook_reader_skips_page_without_main(monkeypatch) -> None:
    monkeypatch.setattr(
        requests,
        "get",
        lambda *a, **k: _FakeResponse("<html><body><div>plain</div></body></html>"),
    )
    soup = BeautifulSoup(
        '<html><body><a href="/docs/page">p</a></body></html>', "html.parser"
    )
    text, _ = _gitbook_reader(soup, "http://example.com/")
    assert text == ""


def test_readthedocs_reader_ignores_anchor_without_href(monkeypatch) -> None:
    monkeypatch.setattr(
        requests,
        "get",
        lambda *a, **k: _FakeResponse(
            '<html><body><div role="main"><p>Doc body.</p></div></body></html>'
        ),
    )
    soup = BeautifulSoup(
        '<html><body><a class="reference internal" name="sec1">anchor</a>'
        '<a class="reference internal" href="http://example.com/page">p</a></body></html>',
        "html.parser",
    )
    text, _ = _readthedocs_reader(soup, "http://example.com/")
    assert "Doc body." in text


def test_gitbook_reader_ignores_anchor_without_href(monkeypatch) -> None:
    monkeypatch.setattr(
        requests,
        "get",
        lambda *a, **k: _FakeResponse(
            "<html><body><main><p>Gb body.</p></main></body></html>"
        ),
    )
    soup = BeautifulSoup(
        '<html><body><a name="toc">t</a><a href="/docs/page">p</a></body></html>',
        "html.parser",
    )
    text, _ = _gitbook_reader(soup, "http://example.com/")
    assert "Gb body." in text
