"""Test document summary index.""" ""
from typing import List
import pytest
from llama_index.indices.service_context import ServiceContext
from llama_index.indices.document_summary.base import DocumentSummaryIndex
from llama_index.schema import Document
from llama_index.response_synthesizers import get_response_synthesizer
from tests.mock_utils.mock_prompts import MOCK_TEXT_QA_PROMPT, MOCK_REFINE_PROMPT


@pytest.fixture
def docs() -> List[Document]:
    return [
        Document(text="This is a test v2.", id_="doc_1"),
        Document(text="This is another test.", id_="doc_2"),
        Document(text="This is a test.", id_="doc_3"),
        Document(text="Hello world.", id_="doc_4"),
    ]


@pytest.fixture
def index(
    docs: List[Document], mock_service_context: ServiceContext
) -> DocumentSummaryIndex:
    response_synthesizer = get_response_synthesizer(
        text_qa_template=MOCK_TEXT_QA_PROMPT,
        refine_template=MOCK_REFINE_PROMPT,
        callback_manager=mock_service_context.callback_manager,
    )
    index = DocumentSummaryIndex.from_documents(
        docs,
        service_context=mock_service_context,
        response_synthesizer=response_synthesizer,
        summary_query="summary_query",
    )
    return index


def test_build_index(
    docs: List[Document],
    index: DocumentSummaryIndex,
) -> None:
    """Test build tree."""
    test = index.get_document_summary("doc_1")
    assert test == "summary_query:This is a test v2."
    test4 = index.get_document_summary("doc_4")
    assert test4 == "summary_query:Hello world."

    all_ref_doc_info = index.ref_doc_info
    for idx, (doc_id, ref_doc_info) in enumerate(all_ref_doc_info.items()):
        assert docs[idx].doc_id == doc_id
        assert len(ref_doc_info.node_ids) == 2
