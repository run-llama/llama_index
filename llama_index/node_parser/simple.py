"""Simple node parser."""
from typing import List, Optional, Sequence

from llama_index.callbacks.base import CallbackManager
from llama_index.callbacks.schema import CBEventType, EventPayload
from llama_index.node_parser.extractors.metadata_extractors import MetadataExtractor
from llama_index.node_parser.interface import NodeParser
from llama_index.node_parser.node_utils import get_nodes_from_document
from llama_index.schema import BaseNode, Document
from llama_index.text_splitter import TextSplitter
from llama_index.text_splitter.utils import get_default_text_splitter
from llama_index.utils import get_tqdm_iterable


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
        metadata_extractor: Optional[MetadataExtractor] = None,
    ) -> None:
        """Init params."""
        self.callback_manager = callback_manager or CallbackManager([])
        self._text_splitter = text_splitter or get_default_text_splitter(
            callback_manager=self.callback_manager
        )
        self._include_metadata = include_metadata
        self._include_prev_next_rel = include_prev_next_rel
        self._metadata_extractor = metadata_extractor

    @classmethod
    def from_defaults(
        cls,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        include_metadata: bool = True,
        include_prev_next_rel: bool = True,
        callback_manager: Optional[CallbackManager] = None,
        metadata_extractor: Optional[MetadataExtractor] = None,
    ) -> "SimpleNodeParser":
        callback_manager = callback_manager or CallbackManager([])

        text_splitter = get_default_text_splitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            callback_manager=callback_manager,
        )
        return cls(
            text_splitter=text_splitter,
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel,
            callback_manager=callback_manager,
            metadata_extractor=metadata_extractor,
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
        with self.callback_manager.event(
            CBEventType.NODE_PARSING, payload={EventPayload.DOCUMENTS: documents}
        ) as event:
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

            if self._metadata_extractor is not None:
                self._metadata_extractor.process_nodes(all_nodes)

            event.on_end(payload={EventPayload.NODES: all_nodes})

        return all_nodes
