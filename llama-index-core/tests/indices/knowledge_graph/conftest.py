from typing import List

import pytest
from llama_index.core.schema import Document


@pytest.fixture()
def documents() -> List[Document]:
    """Get documents."""
    # NOTE: one document for now
    # NOTE: in this unit test, document text == triplets
    doc_text = "(foo, is, bar)\n(hello, is not, world)\n(Jane, is mother of, Bob)"
    return [Document(text=doc_text)]


@pytest.fixture()
def doc_triplets_with_text_around() -> List[str]:
    """Get triplets returned from LLM with text around triplet."""
    # NOTE: the first two triplets below are returned by LLM 'solar'.
    # NOTE: in general it's good to be more relaxed when parsing triplet response. illustrated by the third triplet.
    # NOTE: one document for now
    # NOTE: in this unit test, document text == triplets
    doc_text = (
        "1. (foo, is, bar)\n"
        "2. (hello, is not, world)\n"
        "Third triplet is (Jane, is mother of, Bob) according to your query"
    )
    return [Document(text=doc_text)]
