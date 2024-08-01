from typing import Dict, List, Optional
from dataclasses import dataclass

from llama_index.packs.docugami_kg_rag.helpers.reports import ReportDetails
from llama_index.core.readers import Document
from llama_index.packs.docugami_kg_rag.config import (
    MAX_CHUNK_TEXT_LENGTH,
    LARGE_CONTEXT_INSTRUCT_LLM,
)
import re
from llama_index.packs.docugami_kg_rag.helpers.prompts import (
    CREATE_DIRECT_RETRIEVAL_TOOL_DESCRIPTION_QUERY_PROMPT,
    CREATE_DIRECT_RETRIEVAL_TOOL_SYSTEM_PROMPT,
)
from llama_index.core.vector_stores.types import VectorStoreQueryMode

from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.tools import BaseTool, ToolMetadata, QueryEngineTool

from llama_index.packs.docugami_kg_rag.helpers.vector_store import get_vector_store
from llama_index.packs.docugami_kg_rag.helpers.fused_summary_retriever import (
    FusedSummaryRetriever,
)


@dataclass
class LocalIndexState:
    full_doc_summaries_by_id: Dict[str, Document]
    """Mapping of ID to full document summaries."""

    chunks_by_id: Dict[str, Document]
    """Mapping of ID to chunks."""

    retrieval_tool_function_name: str
    """Function name for retrieval tool e.g. "search_earnings_calls."""

    retrieval_tool_description: str
    """Description of retrieval tool e.g. Searches for and returns chunks from earnings call documents."""

    reports: List[ReportDetails]
    """Details about any reports for this docset."""


def docset_name_to_direct_retriever_tool_function_name(name: str) -> str:
    """
    Converts a docset name to a direct retriever tool function name.

    Direct retriever tool function names follow these conventions:
    1. Retrieval tool function names always start with "search_".
    2. The rest of the name should be a lowercased string, with underscores for whitespace.
    3. Exclude any characters other than a-z (lowercase) from the function name, replacing them with underscores.
    4. The final function name should not have more than one underscore together.

    >>> docset_name_to_direct_retriever_tool_function_name('Earnings Calls')
    'search_earnings_calls'
    >>> docset_name_to_direct_retriever_tool_function_name('COVID-19   Statistics')
    'search_covid_19_statistics'
    >>> docset_name_to_direct_retriever_tool_function_name('2023 Market Report!!!')
    'search_2023_market_report'
    """
    # Replace non-letter characters with underscores and remove extra whitespaces
    name = re.sub(r"[^a-z\d]", "_", name.lower())
    # Replace whitespace with underscores and remove consecutive underscores
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"_{2,}", "_", name)
    name = name.strip("_")

    return f"search_{name}"


def chunks_to_direct_retriever_tool_description(name: str, chunks: List[Document]):
    """
    Converts a set of chunks to a direct retriever tool description.
    """
    texts = [c.text for c in chunks[:100]]
    document = "\n".join(texts)[:MAX_CHUNK_TEXT_LENGTH]

    chat_messages = [
        ChatMessage(
            role=MessageRole.SYSTEM,
            content=CREATE_DIRECT_RETRIEVAL_TOOL_SYSTEM_PROMPT,
        ),
        ChatMessage(
            role=MessageRole.USER,
            content=CREATE_DIRECT_RETRIEVAL_TOOL_DESCRIPTION_QUERY_PROMPT.format(
                docset_name=name, document=document
            ),
        ),
    ]

    summary = LARGE_CONTEXT_INSTRUCT_LLM.chat(chat_messages).message.content

    return f"Given a single input 'query' parameter, searches for and returns chunks from {name} documents. {summary}"


def get_retrieval_tool_for_docset(
    docset_id: str, docset_state: LocalIndexState
) -> Optional[BaseTool]:
    """
    Gets a retrieval tool for an agent.
    """
    chunk_vectorstore = get_vector_store(docset_id)

    if not chunk_vectorstore:
        return None

    retriever = FusedSummaryRetriever(
        vectorstore=chunk_vectorstore,
        parent_doc_store=docset_state.chunks_by_id,
        full_doc_summary_store=docset_state.full_doc_summaries_by_id,
        search_type=VectorStoreQueryMode.MMR,
    )

    if not retriever:
        return None

    query_engine = RetrieverQueryEngine(retriever=retriever)

    return QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name=docset_state.retrieval_tool_function_name,
            description=docset_state.retrieval_tool_description,
        ),
    )
