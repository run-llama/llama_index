from dataclasses import asdict
from typing import List

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    MessageRole,
    CitationsSettings,
)
from llama_index.core.schema import NodeWithScore
from llama_index.core.types import TokenGen

from llama_index.core.base.llms.types import Document, Citation


def response_gen_from_query_engine(response_gen: TokenGen) -> ChatResponseGen:
    response_str = ""
    for token in response_gen:
        response_str += token
        yield ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content=response_str),
            delta=token,
        )


def convert_nodes_to_documents_list(nodes: List[NodeWithScore]) -> List[Document]:
    return [asdict(Document(id=node.node_id, text=node.get_text())) for node in nodes]


def convert_chat_response_to_citations(
    chat_response: ChatResponse, citations_settings: CitationsSettings
) -> List[Citation]:
    if (
        chat_response.raw
        and citations_settings.citations_response_field in chat_response.raw
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
        chat_response.raw
        and citations_settings.documents_response_field in chat_response.raw
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
