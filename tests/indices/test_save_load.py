"""Test keyword table index."""
import json
import tempfile
from pathlib import Path
from typing import Any, List
from unittest.mock import patch

import pytest

from gpt_index.indices.keyword_table.simple_base import GPTSimpleKeywordTableIndex
from gpt_index.readers.schema.base import Document
from tests.mock_utils.mock_decorator import patch_common
from tests.mock_utils.mock_utils import mock_extract_keywords


@pytest.fixture
def documents() -> List[Document]:
    """Get documents."""
    # NOTE: one document for now
    doc_text = "á"
    return [Document(doc_text)]


@patch_common
@patch(
    "gpt_index.indices.keyword_table.simple_base.simple_extract_keywords",
    mock_extract_keywords,
)
def test_write_ascii(
    _mock_init: Any,
    _mock_predict: Any,
    _mock_total_tokens_used: Any,
    _mock_split_text_overlap: Any,
    _mock_split_text: Any,
    documents: List[Document],
) -> None:
    """Test build table."""
    # test simple keyword table
    # NOTE: here the keyword extraction isn't mocked because we're using
    # the regex-based keyword extractor, not GPT
    table = GPTSimpleKeywordTableIndex(documents)
    table_chunks = {n.text for n in table.index_struct.text_chunks.values()}
    assert len(table_chunks) == 1
    assert "á" in table_chunks

    # test that expected keys are present in table
    # NOTE: in mock keyword extractor, stopwords are not filtered
    assert table.index_struct.table.keys() == {"á"}

    with tempfile.TemporaryDirectory() as tmpdir:
        write_path = Path(Path(tmpdir) / "test.json")
        table.save_to_disk(str(write_path))

        with open(write_path, "r", encoding="ascii") as f:
            # Escape unicode characters, for testing, since there is no way to turn
            # off json.loads unicode parsing
            # https://stackoverflow.com/q/50252251

            escaped = f.read().replace("\\", "\\\\")
            data = json.loads(escaped)

            docs = data["docstore"]["docs"]
            doc_key = list(docs.keys())[0]
            text_chunk_id = list(docs[doc_key]["text_chunks"].keys())[0]

            # "\u00e1" is the escaped unicode character for "á"
            assert docs[doc_key]["text_chunks"][text_chunk_id]["text"] == "\\u00e1"
            assert docs[doc_key]["text_chunks"][text_chunk_id]["text"] != "á"


@patch_common
@patch(
    "gpt_index.indices.keyword_table.simple_base.simple_extract_keywords",
    mock_extract_keywords,
)
def test_write_utf8(
    _mock_init: Any,
    _mock_predict: Any,
    _mock_total_tokens_used: Any,
    _mock_split_text_overlap: Any,
    _mock_split_text: Any,
    documents: List[Document],
) -> None:
    """Test build table."""
    # test simple keyword table
    # NOTE: here the keyword extraction isn't mocked because we're using
    # the regex-based keyword extractor, not GPT
    table = GPTSimpleKeywordTableIndex(documents)
    table_chunks = {n.text for n in table.index_struct.text_chunks.values()}
    assert len(table_chunks) == 1
    assert "á" in table_chunks

    # test that expected keys are present in table
    # NOTE: in mock keyword extractor, stopwords are not filtered
    assert table.index_struct.table.keys() == {"á"}

    with tempfile.TemporaryDirectory() as tmpdir:
        write_path = Path(Path(tmpdir) / "test.json")
        table.save_to_disk(str(write_path), encoding="utf8", ensure_ascii=False)

        with open(write_path, "r", encoding="utf8") as f:
            # Escape unicode characters, for testing, since there is no way to turn
            # off json.loads unicode parsing
            # https://stackoverflow.com/q/50252251

            escaped = f.read().replace("\\", "\\\\")
            data = json.loads(escaped)

            docs = data["docstore"]["docs"]
            doc_key = list(docs.keys())[0]
            text_chunk_id = list(docs[doc_key]["text_chunks"].keys())[0]

            # "\u00e1" is the escaped unicode character for "á"
            assert docs[doc_key]["text_chunks"][text_chunk_id]["text"] != "\\u00e1"
            assert docs[doc_key]["text_chunks"][text_chunk_id]["text"] == "á"
