"""Markdown node parser."""
import re
from typing import Dict, List, Optional, Sequence

from llama_index.bridge.pydantic import Field
from llama_index.callbacks.base import CallbackManager
from llama_index.callbacks.schema import CBEventType, EventPayload
from llama_index.node_parser.interface import NodeParser
from llama_index.node_parser.node_utils import build_nodes_from_splits
from llama_index.schema import BaseNode, MetadataMode, TextNode
from llama_index.utils import get_tqdm_iterable


class MarkdownNodeParser(NodeParser):
    """Markdown node parser.

    Splits a document into Nodes using custom Markdown splitting logic.

    Args:
        include_metadata (bool): whether to include metadata in nodes
        include_prev_next_rel (bool): whether to include prev/next relationships

    """

    include_metadata: bool = Field(
        default=True, description="Whether or not to consider metadata when splitting."
    )
    include_prev_next_rel: bool = Field(
        default=True, description="Include prev/next node relationships."
    )
    callback_manager: CallbackManager = Field(
        default_factory=CallbackManager, exclude=True
    )

    @classmethod
    def from_defaults(
        cls,
        include_metadata: bool = True,
        include_prev_next_rel: bool = True,
        callback_manager: Optional[CallbackManager] = None,
    ) -> "MarkdownNodeParser":
        callback_manager = callback_manager or CallbackManager([])

        return cls(
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel,
            callback_manager=callback_manager,
        )

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "MarkdownNodeParser"

    def get_nodes_from_documents(
        self,
        documents: Sequence[TextNode],
        show_progress: bool = False,
    ) -> List[BaseNode]:
        """Parse document into nodes.

        Args:
            documents (Sequence[TextNode]): TextNodes or Documents to parse

        """
        with self.callback_manager.event(
            CBEventType.NODE_PARSING, payload={EventPayload.DOCUMENTS: documents}
        ) as event:
            all_nodes: List[BaseNode] = []
            documents_with_progress = get_tqdm_iterable(
                documents, show_progress, "Parsing documents into nodes"
            )

            for document in documents_with_progress:
                nodes = self.get_nodes_from_node(document)
                all_nodes.extend(nodes)

            event.on_end(payload={EventPayload.NODES: all_nodes})

        return all_nodes

    def get_nodes_from_node(self, node: BaseNode) -> List[TextNode]:
        """Get nodes from document."""
        text = node.get_content(metadata_mode=MetadataMode.NONE)
        markdown_nodes = []
        lines = text.split("\n")
        metadata: Dict[str, str] = {}
        code_block = False
        current_section = ""

        for line in lines:
            if line.startswith("```"):
                code_block = not code_block
            header_match = re.match(r"^(#+)\s(.*)", line)
            if header_match and not code_block:
                if current_section != "":
                    markdown_nodes.append(
                        self._build_node_from_split(
                            current_section.strip(), node, metadata
                        )
                    )
                metadata = self._update_metadata(
                    metadata, header_match.group(2), len(header_match.group(1).strip())
                )
                current_section = f"{header_match.group(2)}\n"
            else:
                current_section += line + "\n"

        markdown_nodes.append(
            self._build_node_from_split(current_section.strip(), node, metadata)
        )

        return markdown_nodes

    def _update_metadata(
        self, headers_metadata: dict, new_header: str, new_header_level: int
    ) -> dict:
        """Update the markdown headers for metadata.

        Removes all headers that are equal or less than the level
        of the newly found header
        """
        updated_headers = {}

        for i in range(1, new_header_level):
            key = f"Header {i}"
            if key in headers_metadata:
                updated_headers[key] = headers_metadata[key]

        updated_headers[f"Header {new_header_level}"] = new_header
        return updated_headers

    def _build_node_from_split(
        self,
        text_split: str,
        node: BaseNode,
        metadata: dict,
    ) -> TextNode:
        """Build node from single text split."""
        node = build_nodes_from_splits(
            [text_split], node, self.include_metadata, self.include_prev_next_rel
        )[0]

        if self.include_metadata:
            node.metadata = {**node.metadata, **metadata}

        return node
