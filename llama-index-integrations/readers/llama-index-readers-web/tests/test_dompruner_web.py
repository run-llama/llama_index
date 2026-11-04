"""Tests for DomPrunerWebReader (all network I/O is mocked)."""

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from llama_index.readers.web import DomPrunerWebReader
from llama_index.readers.web.dompruner_web.base import (
    _estimate_tokens,
    _split_sections,
)

URL = "https://example.com/docs"

FLAT_MARKDOWN = "Just a paragraph with no headings at all."

HEADING_MARKDOWN = """# Page Title
intro body

## Section One
body one

## Section Two
body two

## Section Three
body three
"""

FENCE_MARKDOWN = """# Code Page
intro

## Example
```python
# heading-looking line inside a fence
def handler(timeout):
    return timeout
```
"""


def make_result(url: str = URL, markdown: str = FLAT_MARKDOWN, **overrides):
    base = {
        "url": url,
        "render_type": "SSR",
        "markdown": markdown,
        "original_tokens": 15965,
        "refined_tokens": 1368,
        "reduction_ratio": 0.914,
        "bm25_confidence": 4.2,
        "meta": {"title": "Docs", "lang": "en", "description": "API reference"},
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def patch_pipeline(mocker, markdown: str = FLAT_MARKDOWN, **overrides):
    async def fake_run_pipeline(url, query=""):
        return make_result(url=url, markdown=markdown, **overrides)

    return mocker.patch(
        "dompruner.run_pipeline", new=AsyncMock(side_effect=fake_run_pipeline)
    )


def test_init_defaults():
    reader = DomPrunerWebReader()
    assert reader.query is None
    assert reader.token_budget == 1500


def test_load_data_requires_list_of_urls():
    reader = DomPrunerWebReader()
    with pytest.raises(ValueError, match="urls must be a list of strings"):
        reader.load_data("https://example.com")


def test_async_load_data_requires_list_of_urls():
    reader = DomPrunerWebReader()
    with pytest.raises(ValueError, match="urls must be a list of strings"):
        asyncio.run(reader.async_load_data(None))


def test_load_data_without_headings_returns_single_document(mocker):
    mock = patch_pipeline(mocker, FLAT_MARKDOWN)
    documents = DomPrunerWebReader().load_data([URL])

    mock.assert_awaited_with(URL, "")
    assert len(documents) == 1
    assert documents[0].text == FLAT_MARKDOWN
    assert documents[0].metadata["heading_path"] == []
    assert documents[0].metadata["section_title"] is None


def test_heading_chain_preserved_per_section(mocker):
    patch_pipeline(mocker, HEADING_MARKDOWN)
    documents = DomPrunerWebReader().load_data([URL])

    assert [doc.metadata["section_title"] for doc in documents] == [
        "Page Title",
        "Section One",
        "Section Two",
        "Section Three",
    ]
    assert [doc.metadata["heading_path"] for doc in documents] == [
        ["Page Title"],
        ["Page Title", "Section One"],
        ["Page Title", "Section Two"],
        ["Page Title", "Section Three"],
    ]
    assert documents[3].text == "## Section Three\nbody three"


def test_heading_inside_code_fence_does_not_split_section(mocker):
    patch_pipeline(mocker, FENCE_MARKDOWN)
    documents = DomPrunerWebReader().load_data([URL])

    assert [doc.metadata["section_title"] for doc in documents] == [
        "Code Page",
        "Example",
    ]
    assert "# heading-looking line inside a fence" in documents[1].text


def test_query_is_forwarded_to_pipeline(mocker):
    mock = patch_pipeline(mocker, HEADING_MARKDOWN, bm25_confidence=7.5)
    DomPrunerWebReader(query="body two").load_data([URL])

    mock.assert_awaited_with(URL, "body two")


def test_token_budget_prunes_sections_when_query_set(mocker):
    patch_pipeline(mocker, HEADING_MARKDOWN)
    sections = _split_sections(HEADING_MARKDOWN)
    budget = _estimate_tokens(sections[0].text)

    reader = DomPrunerWebReader(query="body", token_budget=budget)
    documents = reader.load_data([URL])

    assert len(documents) == 1
    assert documents[0].metadata["section_title"] == "Page Title"


def test_no_query_skips_pruning_even_with_small_budget(mocker):
    patch_pipeline(mocker, HEADING_MARKDOWN)
    documents = DomPrunerWebReader(token_budget=1).load_data([URL])

    assert len(documents) == 4


def test_token_budget_none_keeps_all_sections(mocker):
    patch_pipeline(mocker, HEADING_MARKDOWN)
    reader = DomPrunerWebReader(query="body", token_budget=None)
    documents = reader.load_data([URL])

    assert len(documents) == 4


def test_oversized_section_is_kept_whole_not_split(mocker):
    big_fence = "# Big\n```python\n" + ("x = 1\n" * 200) + "```\n"
    patch_pipeline(mocker, big_fence)
    documents = DomPrunerWebReader(query="x", token_budget=10).load_data([URL])

    assert len(documents) == 1
    assert documents[0].text.endswith("```")
    assert "x = 1\n" * 200 in documents[0].text


def test_token_stats_and_page_meta_surfaced_in_metadata(mocker):
    patch_pipeline(mocker, FLAT_MARKDOWN)
    document = DomPrunerWebReader().load_data([URL])[0]
    metadata = document.metadata

    assert metadata["url"] == URL
    assert metadata["render_type"] == "SSR"
    assert metadata["original_tokens"] == 15965
    assert metadata["refined_tokens"] == 1368
    assert metadata["reduction_ratio"] == 0.914
    assert metadata["bm25_confidence"] == 4.2
    assert metadata["title"] == "Docs"
    assert metadata["lang"] == "en"
    assert metadata["description"] == "API reference"


def test_multiple_urls_are_all_processed(mocker):
    mock = patch_pipeline(mocker, FLAT_MARKDOWN)
    urls = ["https://example.com/a", "https://example.com/b"]
    documents = DomPrunerWebReader().load_data(urls)

    assert [call.args[0] for call in mock.await_args_list] == urls
    assert [doc.metadata["url"] for doc in documents] == urls


def test_async_load_data(mocker):
    patch_pipeline(mocker, HEADING_MARKDOWN)
    documents = asyncio.run(DomPrunerWebReader().async_load_data([URL]))

    assert len(documents) == 4


def test_load_data_inside_running_event_loop(mocker):
    patch_pipeline(mocker, FLAT_MARKDOWN)

    async def scenario():
        return DomPrunerWebReader().load_data([URL])

    documents = asyncio.run(scenario())
    assert len(documents) == 1


def test_missing_dompruner_raises_clear_error(mocker):
    mocker.patch.dict(sys.modules, {"dompruner": None})
    reader = DomPrunerWebReader()

    with pytest.raises(ImportError, match="pip install dompruner"):
        reader.load_data([URL])


def test_split_sections_without_markdown():
    assert _split_sections("") == []
