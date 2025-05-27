"""Conftest."""

from typing import List

import pytest
from llama_index.core.schema import Document


@pytest.fixture()
def documents() -> List[Document]:
    """Get documents."""
    # NOTE: one document for now
    doc_text = (
        "Hello world.\nThis is a test.\nThis is another test.\nThis is a test v2."
    )
    return [Document(text=doc_text)]
