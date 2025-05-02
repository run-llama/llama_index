from typing import List

import pytest
from llama_index.core.indices.document_summary.base import DocumentSummaryIndex
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.schema import Document
from tests.mock_utils.mock_prompts import MOCK_REFINE_PROMPT, MOCK_TEXT_QA_PROMPT


@pytest.fixture()
def docs() -> List[Document]:
    return [
        Document(text="This is a test v2.", id_="doc_1"),
        Document(text="This is another test.", id_="doc_2"),
        Document(text="This is a test.", id_="doc_3"),
        Document(text="Hello world.", id_="doc_4"),
    ]


@pytest.fixture()
def index(
    docs: List[Document], patch_llm_predictor, mock_embed_model
) -> DocumentSummaryIndex:
    response_synthesizer = get_response_synthesizer(
        text_qa_template=MOCK_TEXT_QA_PROMPT,
        refine_template=MOCK_REFINE_PROMPT,
    )
    return DocumentSummaryIndex.from_documents(
        docs,
        response_synthesizer=response_synthesizer,
        summary_query="summary_query",
        embed_model=mock_embed_model,
    )
