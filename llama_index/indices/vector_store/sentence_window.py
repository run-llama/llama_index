"""Sentence window vector index.

An index that that is built on top of an existing vector store.

Retrieves based on sentence-level embeddings with an adaptive window size.

"""

from typing import Any, Callable, List, Optional, Sequence, Type

from llama_index.data_structs.data_structs import IndexDict
from llama_index.indices.vector_store.base import VectorStoreIndex
from llama_index.indices.base_retriever import BaseRetriever
from llama_index.indices.service_context import ServiceContext
from llama_index.node_parser.node_utils import build_nodes_from_splits
from llama_index.schema import BaseNode, Document
from llama_index.storage.storage_context import StorageContext
from llama_index.text_splitter.utils import split_by_sentence_tokenizer


def build_window_nodes_from_documents(
    documents: Sequence[Document],
    sentence_splitter: Callable[[str], List[str]],
    window_size: int = 5,
    window_metadata_key: str = "window",
    original_text_metadata_key: str = "original_text",
) -> List[BaseNode]:
    """Build window nodes from documents."""
    all_nodes: List[BaseNode] = []
    for doc in documents:
        text = doc.text
        text_splits = sentence_splitter(text)
        nodes = build_nodes_from_splits(text_splits, doc, include_prev_next_rel=True)

        # add window to each node
        for i, node in enumerate(nodes):
            window_nodes = nodes[
                max(0, i - window_size) : min(i + window_size, len(nodes))
            ]

            node.metadata[window_metadata_key] = " ".join(
                [n.text for n in window_nodes]
            )
            node.metadata[original_text_metadata_key] = node.text

            # exclude window metadata from embed and llm
            node.excluded_embed_metadata_keys.extend(
                [window_metadata_key, original_text_metadata_key]
            )
            node.excluded_llm_metadata_keys.extend(
                [window_metadata_key, original_text_metadata_key]
            )

        all_nodes.extend(nodes)

    return all_nodes


class SentenceWindowVectorIndex(VectorStoreIndex):
    """[BETA] Sentence Window Vector Index.

    Note: This is a BETA feature, and is subject to change in future releases.
    Some languages and text formats may not work well with this index.

    Args:
        use_async (bool): Whether to use asynchronous calls. Defaults to False.
        show_progress (bool): Whether to show tqdm progress bars. Defaults to False.
        store_nodes_override (bool): set to True to always store Node objects in index
            store and document store even if vector store keeps text. Defaults to False
    """

    index_struct_cls = IndexDict
    window_metadata_key = "window"

    def __init__(
        self,
        nodes: Optional[Sequence[BaseNode]] = None,
        index_struct: Optional[IndexDict] = None,
        service_context: Optional[ServiceContext] = None,
        storage_context: Optional[StorageContext] = None,
        sentence_splitter: Optional[Callable[[str], List[str]]] = None,
        window_size: int = 5,
        use_async: bool = False,
        store_nodes_override: bool = False,
        show_progress: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        self._sentence_splitter = sentence_splitter or split_by_sentence_tokenizer()
        self._window_size = window_size

        super().__init__(
            nodes=nodes,
            index_struct=index_struct,
            service_context=service_context,
            storage_context=storage_context,
            show_progress=show_progress,
            store_nodes_override=store_nodes_override,
            use_async=use_async,
            **kwargs,
        )

    @classmethod
    def from_documents(
        cls: Type["SentenceWindowVectorIndex"],
        documents: Sequence[Document],
        storage_context: Optional[StorageContext] = None,
        service_context: Optional[ServiceContext] = None,
        show_progress: bool = False,
        sentence_splitter: Optional[Callable[[str], List[str]]] = None,
        window_size: int = 5,
        **kwargs: Any,
    ) -> "SentenceWindowVectorIndex":
        """Create index from documents.

        Args:
            documents (Optional[Sequence[BaseDocument]]): List of documents to
                build the index from.

        """
        storage_context = storage_context or StorageContext.from_defaults()
        service_context = service_context or ServiceContext.from_defaults()
        docstore = storage_context.docstore

        sentence_splitter = sentence_splitter or split_by_sentence_tokenizer()

        with service_context.callback_manager.as_trace("index_construction"):
            for doc in documents:
                docstore.set_document_hash(doc.get_doc_id(), doc.hash)
            nodes = build_window_nodes_from_documents(
                documents,
                sentence_splitter,
                window_size=window_size,
                window_metadata_key=cls.window_metadata_key,
            )

            return cls(
                nodes=nodes,
                storage_context=storage_context,
                service_context=service_context,
                show_progress=show_progress,
                sentence_splitter=sentence_splitter,
                window_size=window_size,
                **kwargs,
            )

    def as_retriever(self, **kwargs: Any) -> BaseRetriever:
        # NOTE: lazy import
        from llama_index.indices.vector_store.retrievers import (
            SentenceWindowVectorRetriever,
        )

        return SentenceWindowVectorRetriever(
            self,
            node_ids=list(self.index_struct.nodes_dict.values()),
            **kwargs,
        )

    def insert(self, document: Document, **insert_kwargs: Any) -> None:
        """Insert a document."""
        with self._service_context.callback_manager.as_trace("insert"):
            nodes = build_window_nodes_from_documents(
                [document],
                self._sentence_splitter,
                window_size=self._window_size,
                window_metadata_key=self.window_metadata_key,
            )
            self.insert_nodes(nodes, **insert_kwargs)
            self.docstore.set_document_hash(document.get_doc_id(), document.hash)


GPTVectorStoreIndex = VectorStoreIndex
