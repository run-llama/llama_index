from typing import Any, Dict, List, Tuple
import pytest

from gpt_index.readers.schema.base import Document


@pytest.fixture
def struct_kwargs() -> Tuple[Dict, Dict]:
    """Index kwargs."""
    index_kwargs: Dict[str, Any] = {}
    retrieval_kwargs: Dict[str, Any] = {
        "similarity_top_k": 1,
    }
    return index_kwargs, retrieval_kwargs


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
