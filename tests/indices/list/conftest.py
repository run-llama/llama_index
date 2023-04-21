from typing import Any, List
import pytest
from gpt_index.indices.list.base import GPTListIndex

from gpt_index.readers.schema.base import Document
from tests.mock_utils.mock_decorator import patch_common


@pytest.fixture
def documents() -> List[Document]:
    """Get documents."""
    # NOTE: one document for now
    doc_text = (
        "Hello world.\n"
        "This is a test.\n"
        "This is another test.\n"
        "This is a test v2."
    )
    return [Document(doc_text)]


@pytest.fixture
@patch_common
def list_index(
    _mock_init: Any,
    _mock_predict: Any,
    _mock_total_tokens_used: Any,
    _mock_split_text_overlap: Any,
    _mock_split_text: Any,
    documents: List[Document],
) -> GPTListIndex:
    return GPTListIndex.from_documents(documents)
