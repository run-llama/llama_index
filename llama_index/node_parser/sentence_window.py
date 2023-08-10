"""Simple node parser."""
from typing import List, Callable, Optional, Sequence

from llama_index.callbacks.base import CallbackManager
from llama_index.callbacks.schema import CBEventType, EventPayload
from llama_index.node_parser.extractors.metadata_extractors import MetadataExtractor
from llama_index.node_parser.interface import NodeParser
from llama_index.node_parser.node_utils import build_nodes_from_splits
from llama_index.schema import BaseNode, Document
from llama_index.text_splitter.utils import split_by_sentence_tokenizer
from llama_index.utils import get_tqdm_iterable


DEFAULT_WINDOW_SIZE = 5
DEFAULT_WINDOW_METADATA_KEY = "window"
DEFAULT_OG_TEXT_METADATA_KEY = "original_text"


class SentenceWinowNodeParser(NodeParser):
    """Sentence window node parser.

    Splits a document into Nodes, with each node being a sentence.
    Each node contains a window from the surrounding sentences in the metadata.

    Args:
        sentence_splitter (Optional[Callable]): splits text into sentences
        include_metadata (bool): whether to include metadata in nodes
        include_prev_next_rel (bool): whether to include prev/next relationships
    """

    def __init__(
        self,
        sentence_splitter: Optional[Callable[[str], List[str]]] = None,
        window_size: int = DEFAULT_WINDOW_SIZE,
        window_metadata_key: str = DEFAULT_WINDOW_METADATA_KEY,
        original_text_metadata_key: str = DEFAULT_OG_TEXT_METADATA_KEY,
        include_metadata: bool = True,
        include_prev_next_rel: bool = True,
        callback_manager: Optional[CallbackManager] = None,
        metadata_extractor: Optional[MetadataExtractor] = None,
    ) -> None:
        """Init params."""
        self.callback_manager = callback_manager or CallbackManager([])
        self._sentence_splitter = sentence_splitter or split_by_sentence_tokenizer()
        self._window_size = window_size
        self._window_metadata_key = window_metadata_key
        self._original_text_metadata_key = original_text_metadata_key

        self._include_metadata = include_metadata
        self._include_prev_next_rel = include_prev_next_rel
        self._metadata_extractor = metadata_extractor

    @classmethod
    def from_defaults(
        cls,
        sentence_splitter: Optional[Callable[[str], List[str]]] = None,
        window_size: int = DEFAULT_WINDOW_SIZE,
        window_metadata_key: str = DEFAULT_WINDOW_METADATA_KEY,
        original_text_metadata_key: str = DEFAULT_OG_TEXT_METADATA_KEY,
        include_metadata: bool = True,
        include_prev_next_rel: bool = True,
        callback_manager: Optional[CallbackManager] = None,
        metadata_extractor: Optional[MetadataExtractor] = None,
    ) -> "SentenceWinowNodeParser":
        callback_manager = callback_manager or CallbackManager([])

        sentence_splitter = sentence_splitter or split_by_sentence_tokenizer()

        return cls(
            sentence_splitter=sentence_splitter,
            window_size=window_size,
            window_metadata_key=window_metadata_key,
            original_text_metadata_key=original_text_metadata_key,
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
                self._sentence_splitter(document.text)
                nodes = self.build_window_nodes_from_documents([document])
                all_nodes.extend(nodes)

            if self._metadata_extractor is not None:
                self._metadata_extractor.process_nodes(all_nodes)

            event.on_end(payload={EventPayload.NODES: all_nodes})

        return all_nodes

    def build_window_nodes_from_documents(
        self, documents: Sequence[Document]
    ) -> List[BaseNode]:
        """Build window nodes from documents."""
        all_nodes: List[BaseNode] = []
        for doc in documents:
            text = doc.text
            text_splits = self._sentence_splitter(text)
            nodes = build_nodes_from_splits(
                text_splits, doc, include_prev_next_rel=True
            )

            # add window to each node
            for i, node in enumerate(nodes):
                window_nodes = nodes[
                    max(0, i - self._window_size) : min(
                        i + self._window_size, len(nodes)
                    )
                ]

                node.metadata[self._window_metadata_key] = " ".join(
                    [n.text for n in window_nodes]
                )
                node.metadata[self._original_text_metadata_key] = node.text

                # exclude window metadata from embed and llm
                node.excluded_embed_metadata_keys.extend(
                    [self._window_metadata_key, self._original_text_metadata_key]
                )
                node.excluded_llm_metadata_keys.extend(
                    [self._window_metadata_key, self._original_text_metadata_key]
                )

            all_nodes.extend(nodes)

        return all_nodes
