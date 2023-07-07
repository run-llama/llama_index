"""Simple node parser."""
from typing import List, Optional, Sequence

from llama_index.callbacks.base import CallbackManager
from llama_index.callbacks.schema import CBEventType, EventPayload
from llama_index.constants import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from llama_index.langchain_helpers.text_splitter import TextSplitter, TokenTextSplitter
from llama_index.node_parser.interface import NodeParser, NodeParserPostProcessor
from llama_index.node_parser.node_utils import get_nodes_from_document
from llama_index.utils import get_tqdm_iterable
from llama_index.schema import Document
from llama_index.schema import BaseNode


class SimpleNodeParser(NodeParser):
    """Simple node parser.

    Splits a document into Nodes using a TextSplitter.

    Args:
        text_splitter (Optional[TextSplitter]): text splitter
        include_metadata (bool): whether to include metadata in nodes
        include_prev_next_rel (bool): whether to include prev/next relationships

    """

    def __init__(
        self,
        text_splitter: Optional[TextSplitter] = None,
        include_metadata: bool = True,
        include_prev_next_rel: bool = True,
        callback_manager: Optional[CallbackManager] = None,
        post_processors: Optional[List[NodeParserPostProcessor]] = None,
    ) -> None:
        """Init params."""
        self.callback_manager = callback_manager or CallbackManager([])
        self._text_splitter = text_splitter or TokenTextSplitter(
            callback_manager=self.callback_manager
        )
        self._include_metadata = include_metadata
        self._include_prev_next_rel = include_prev_next_rel
        self._post_processors = post_processors or []

    @classmethod
    def from_defaults(
        cls,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        include_metadata: bool = True,
        include_prev_next_rel: bool = True,
        post_processors: Optional[List[NodeParserPostProcessor]] = None,
        callback_manager: Optional[CallbackManager] = None,
    ) -> "SimpleNodeParser":
        callback_manager = callback_manager or CallbackManager([])
        chunk_size = chunk_size or DEFAULT_CHUNK_SIZE
        chunk_overlap = (
            chunk_overlap if chunk_overlap is not None else DEFAULT_CHUNK_OVERLAP
        )

        token_text_splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            callback_manager=callback_manager,
        )
        return cls(
            text_splitter=token_text_splitter,
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel,
            callback_manager=callback_manager,
        )

    def get_nodes_from_documents(
        self,
        documents: Sequence[Document],
        show_progress: bool = False,
    ) -> List[BaseNode]:
        """Parse document into nodes.

        Args:
            documents (Sequence[Document]): documents to parse
            include_metadata (bool): whether to include metadata in nodes

        """
        event_id = self.callback_manager.on_event_start(
            CBEventType.NODE_PARSING, payload={EventPayload.DOCUMENTS: documents}
        )

        all_nodes: List[BaseNode] = []
        documents_with_progress = get_tqdm_iterable(
            documents, show_progress, "Parsing documents into nodes"
        )

        for document in documents_with_progress:
            nodes = get_nodes_from_document(
                document,
                self._text_splitter,
                self._include_metadata,
                include_prev_next_rel=self._include_prev_next_rel,
            )
            all_nodes.extend(nodes)

        for post_processor in self._post_processors:
            post_processor.post_process_nodes(all_nodes)

        self.callback_manager.on_event_end(
            CBEventType.NODE_PARSING,
            payload={EventPayload.NODES: all_nodes},
            event_id=event_id,
        )
        return all_nodes
