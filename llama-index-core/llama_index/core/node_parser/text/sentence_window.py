"""Simple node parser."""

from typing import Any, Callable, List, Optional, Sequence

from llama_index.core.bridge.pydantic import Field
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.node_parser.interface import NodeParser
from llama_index.core.node_parser.node_utils import (
    build_nodes_from_splits,
    default_id_func,
)
from llama_index.core.node_parser.text.utils import split_by_sentence_tokenizer
from llama_index.core.schema import BaseNode, Document
from llama_index.core.utils import get_tqdm_iterable

DEFAULT_WINDOW_SIZE = 3
DEFAULT_WINDOW_METADATA_KEY = "window"
DEFAULT_OG_TEXT_METADATA_KEY = "original_text"


class SentenceWindowNodeParser(NodeParser):
    """
    Sentence window node parser.

    Splits a document into Nodes, with each node being a sentence.
    Each node contains a window from the surrounding sentences in the metadata.

    Args:
        sentence_splitter (Optional[Callable]): splits text into sentences
        include_metadata (bool): whether to include metadata in nodes
        include_prev_next_rel (bool): whether to include prev/next relationships

    """

    sentence_splitter: Callable[[str], List[str]] = Field(
        default_factory=split_by_sentence_tokenizer,
        description="The text splitter to use when splitting documents.",
        exclude=True,
    )
    window_size: int = Field(
        default=DEFAULT_WINDOW_SIZE,
        description="The number of sentences on each side of a sentence to capture.",
        gt=0,
    )
    window_metadata_key: str = Field(
        default=DEFAULT_WINDOW_METADATA_KEY,
        description="The metadata key to store the sentence window under.",
    )
    original_text_metadata_key: str = Field(
        default=DEFAULT_OG_TEXT_METADATA_KEY,
        description="The metadata key to store the original sentence in.",
    )

    @classmethod
    def class_name(cls) -> str:
        return "SentenceWindowNodeParser"

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
        id_func: Optional[Callable[[int, Document], str]] = None,
    ) -> "SentenceWindowNodeParser":
        callback_manager = callback_manager or CallbackManager([])

        sentence_splitter = sentence_splitter or split_by_sentence_tokenizer()

        id_func = id_func or default_id_func

        return cls(
            sentence_splitter=sentence_splitter,
            window_size=window_size,
            window_metadata_key=window_metadata_key,
            original_text_metadata_key=original_text_metadata_key,
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel,
            callback_manager=callback_manager,
            id_func=id_func,
        )

    def _parse_nodes(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[BaseNode]:
        """Parse document into nodes."""
        all_nodes: List[BaseNode] = []
        nodes_with_progress = get_tqdm_iterable(nodes, show_progress, "Parsing nodes")

        for node in nodes_with_progress:
            nodes = self.build_window_nodes_from_documents([node])
            all_nodes.extend(nodes)

        return all_nodes

    def build_window_nodes_from_documents(
        self, documents: Sequence[Document]
    ) -> List[BaseNode]:
        """Build window nodes from documents."""
        all_nodes: List[BaseNode] = []
        for doc in documents:
            text = doc.text
            text_splits = self.sentence_splitter(text)
            nodes = build_nodes_from_splits(
                text_splits,
                doc,
                id_func=self.id_func,
            )

            # add window to each node
            for i, node in enumerate(nodes):
                window_nodes = nodes[
                    max(0, i - self.window_size) : min(
                        i + self.window_size + 1, len(nodes)
                    )
                ]

                node.metadata[self.window_metadata_key] = " ".join(
                    [n.text for n in window_nodes]
                )
                node.metadata[self.original_text_metadata_key] = node.text

                # exclude window metadata from embed and llm
                node.excluded_embed_metadata_keys.extend(
                    [self.window_metadata_key, self.original_text_metadata_key]
                )
                node.excluded_llm_metadata_keys.extend(
                    [self.window_metadata_key, self.original_text_metadata_key]
                )

            all_nodes.extend(nodes)

        return all_nodes
