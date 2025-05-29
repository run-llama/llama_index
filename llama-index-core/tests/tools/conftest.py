"""Conftest."""

from typing import List

import pytest
from llama_index.core.schema import Document


@pytest.fixture()
def documents() -> List[Document]:
    """Get documents."""
    # NOTE: one document for now
    doc_text = (
        "Hello world.\n"
        "This is a test.\n"
        "This is another test.\n"
        "This is a test v2."
    )
    return [Document(text=doc_text)]
