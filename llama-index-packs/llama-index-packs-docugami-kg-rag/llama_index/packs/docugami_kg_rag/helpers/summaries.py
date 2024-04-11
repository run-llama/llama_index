import hashlib
from typing import Dict

from tqdm import tqdm

from llama_index.llms.openai import OpenAI

from llama_index.packs.docugami_kg_rag.config import (
    LARGE_CONTEXT_INSTRUCT_LLM,
    MAX_CHUNK_TEXT_LENGTH,
    INCLUDE_XML_TAGS,
    MIN_LENGTH_TO_SUMMARIZE,
    MAX_FULL_DOCUMENT_TEXT_LENGTH,
    SMALL_CONTEXT_INSTRUCT_LLM,
)
from llama_index.core.readers import Document

from llama_index.packs.docugami_kg_rag.helpers.prompts import (
    CREATE_FULL_DOCUMENT_SUMMARY_QUERY_PROMPT,
    CREATE_FULL_DOCUMENT_SUMMARY_SYSTEM_PROMPT,
    CREATE_CHUNK_SUMMARY_QUERY_PROMPT,
    CREATE_CHUNK_SUMMARY_SYSTEM_PROMPT,
)
from llama_index.packs.docugami_kg_rag.config import PARENT_DOC_ID_KEY
from llama_index.core.llms import ChatMessage, MessageRole

FORMAT = (
    "text"
    if not INCLUDE_XML_TAGS
    else "semantic XML without any namespaces or attributes"
)


def _build_summary_mappings(
    docs_by_id: Dict[str, Document],
    system_message: str,
    prompt_template: str,
    llm: OpenAI,
    min_length_to_summarize=MIN_LENGTH_TO_SUMMARIZE,
    max_length_cutoff=MAX_CHUNK_TEXT_LENGTH,
    label="summaries",
) -> Dict[str, Document]:
    """
    Build summaries for all the given documents.
    """
    summaries: Dict[str, Document] = {}

    for id, doc in tqdm(docs_by_id.items(), desc=f"Building {label}", unit="chunks"):
        text_content = doc.text[:max_length_cutoff]

        query_str = prompt_template.format(format=FORMAT, document=text_content)

        chat_messages = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=system_message,
            ),
            ChatMessage(role=MessageRole.USER, content=query_str),
        ]
        # Only summarize when content is longer than min_length_to_summarize
        summary_txt = (
            llm.chat(chat_messages).message.content
            if len(text_content) < min_length_to_summarize
            else text_content
        )

        # Create new hashed id for the summary and add original id as parent doc id
        summaries[id] = summary_txt
        summary_id = hashlib.md5(summary_txt.encode()).hexdigest()
        meta = doc.metadata
        meta["id"] = summary_id
        meta[PARENT_DOC_ID_KEY] = id

        summaries[id] = Document(
            text=summary_txt,
            metadata=meta,
        )

    return summaries


def build_full_doc_summary_mappings(
    docs_by_id: Dict[str, Document]
) -> Dict[str, Document]:
    """
    Build summaries for all the given full documents.
    """
    return _build_summary_mappings(
        docs_by_id=docs_by_id,
        system_message=CREATE_FULL_DOCUMENT_SUMMARY_SYSTEM_PROMPT,
        prompt_template=CREATE_FULL_DOCUMENT_SUMMARY_QUERY_PROMPT,
        llm=LARGE_CONTEXT_INSTRUCT_LLM,
        min_length_to_summarize=MIN_LENGTH_TO_SUMMARIZE,
        max_length_cutoff=MAX_FULL_DOCUMENT_TEXT_LENGTH,
        label="full document summaries",
    )


def build_chunk_summary_mappings(
    docs_by_id: Dict[str, Document]
) -> Dict[str, Document]:
    """
    Build summaries for all the given chunks.
    """
    return _build_summary_mappings(
        docs_by_id=docs_by_id,
        system_message=CREATE_CHUNK_SUMMARY_SYSTEM_PROMPT,
        prompt_template=CREATE_CHUNK_SUMMARY_QUERY_PROMPT,
        llm=SMALL_CONTEXT_INSTRUCT_LLM,
        min_length_to_summarize=MIN_LENGTH_TO_SUMMARIZE,
        max_length_cutoff=MAX_CHUNK_TEXT_LENGTH,
        label="chunk summaries",
    )
