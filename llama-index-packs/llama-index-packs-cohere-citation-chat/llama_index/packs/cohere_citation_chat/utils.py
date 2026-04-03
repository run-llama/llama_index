from dataclasses import asdict
from typing import List

from llama_index.core.base.llms.types import ChatResponse
from llama_index.core.schema import NodeWithScore

from .types import Document, Citation, CitationsSettings


def convert_nodes_to_documents_list(nodes: List[NodeWithScore]) -> List[Document]:
    if nodes:
        return [
            asdict(Document(id=node.node_id, text=node.get_text())) for node in nodes
        ]
    return []


def convert_chat_response_to_citations(
    chat_response: ChatResponse, citations_settings: CitationsSettings
) -> List[Citation]:
    if (
        chat_response
        and chat_response.raw
        and chat_response.raw.get(citations_settings.citations_response_field, None)
    ):
        return [
            Citation(
                text=citation.get("text"),
                start=citation.get("start"),
                end=citation.get("end"),
                document_ids=citation.get("document_ids"),
            )
            for citation in chat_response.raw[
                citations_settings.citations_response_field
            ]
        ]
    return []


def convert_chat_response_to_documents(
    chat_response: ChatResponse, citations_settings: CitationsSettings
) -> List[Document]:
    if (
        chat_response
        and chat_response.raw
        and chat_response.raw.get(citations_settings.documents_response_field, None)
    ):
        return [
            Document(
                id=document.get("id"),
                text=document.get("text"),
            )
            for document in chat_response.raw[
                citations_settings.documents_response_field
            ]
        ]
    return []
