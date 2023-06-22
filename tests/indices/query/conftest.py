from typing import Dict, List

import pytest
from llama_index.data_structs.struct_type import IndexStructType
from llama_index.readers.schema.base import Document

from tests.mock_utils.mock_prompts import (
    MOCK_INSERT_PROMPT,
    MOCK_KEYWORD_EXTRACT_PROMPT,
    MOCK_QUERY_KEYWORD_EXTRACT_PROMPT,
    MOCK_QUERY_PROMPT,
    MOCK_REFINE_PROMPT,
    MOCK_SUMMARY_PROMPT,
    MOCK_TEXT_QA_PROMPT,
)


@pytest.fixture
def index_kwargs() -> Dict:
    """Index kwargs."""
    return {
        "tree": {
            "summary_template": MOCK_SUMMARY_PROMPT,
            "insert_prompt": MOCK_INSERT_PROMPT,
            "num_children": 2,
        },
        "list": {},
        "table": {
            "keyword_extract_template": MOCK_KEYWORD_EXTRACT_PROMPT,
        },
        "vector": {},
        "pinecone": {},
    }


@pytest.fixture
def retriever_kwargs() -> Dict:
    return {
        IndexStructType.TREE: {
            "query_template": MOCK_QUERY_PROMPT,
            "text_qa_template": MOCK_TEXT_QA_PROMPT,
            "refine_template": MOCK_REFINE_PROMPT,
        },
        IndexStructType.LIST: {},
        IndexStructType.KEYWORD_TABLE: {
            "query_keyword_extract_template": MOCK_QUERY_KEYWORD_EXTRACT_PROMPT,
        },
        IndexStructType.DICT: {
            "similarity_top_k": 1,
        },
        IndexStructType.PINECONE: {
            "similarity_top_k": 1,
        },
    }


@pytest.fixture
def documents() -> List[Document]:
    """Get documents."""
    docs = [
        Document(text="This is a test v2."),
        Document(text="This is another test."),
        Document(text="This is a test."),
        Document(text="Hello world."),
        Document(text="Hello world."),
        Document(text="This is a test."),
        Document(text="This is another test."),
        Document(text="This is a test v2."),
    ]
    return docs
