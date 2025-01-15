"""Markdown node parser."""

import re
from typing import Any, Dict, List, Optional, Sequence

from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.node_parser.interface import NodeParser
from llama_index.core.node_parser.node_utils import build_nodes_from_splits
from llama_index.core.schema import BaseNode, MetadataMode, TextNode
from llama_index.core.utils import get_tqdm_iterable
from pydantic import Field


class MarkdownNodeParserWithOptions(NodeParser):
    """Markdown node parser.

    Splits a document into Nodes using custom Markdown splitting logic.
    It allows for setting a maximum node length. If the contents belonging to a
    header exceed the maximum node length, the contents are split into multiple parts.
    The header title is repeated in the next node, and the contents flow over into the
    new node, without overlap.

    Args:
        include_metadata (bool): whether to include metadata in nodes
        include_prev_next_rel (bool): whether to include prev/next relationships
        max_node_length (int): desired maximum characters in a node
    """

    max_node_length: int | None = Field(
        default=None, description="desired maximum characters in a node"
    )

    @classmethod
    def from_defaults(
        cls,
        include_metadata: bool = True,
        include_prev_next_rel: bool = True,
        callback_manager: Optional[CallbackManager] = None,
        max_node_length=max_node_length,
    ) -> "MarkdownNodeParserWithOptions":
        callback_manager = callback_manager or CallbackManager([])

        return cls(
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel,
            callback_manager=callback_manager,
            max_node_length=max_node_length,
        )

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "MarkdownNodeParserWithOptions"

    def _parse_nodes(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[BaseNode]:
        all_nodes: List[BaseNode] = []
        nodes_with_progress = get_tqdm_iterable(nodes, show_progress, "Parsing nodes")

        for node in nodes_with_progress:
            nodes = self.get_nodes_from_node(node)
            all_nodes.extend(nodes)

        return all_nodes

    def get_nodes_from_node(self, node: BaseNode) -> List[TextNode]:
        """Get nodes from document."""
        text = node.get_content(metadata_mode=MetadataMode.NONE)
        markdown_nodes = []
        paragraphs = text.split("\n")
        metadata: Dict[str, str] = {}
        current_section = ""
        header_regex = r"^(#+)\s(.*)"
        last_header_match = re.match(header_regex, "# New section")

        for par in [p.strip() for p in paragraphs]:
            header_match = re.match(header_regex, par)

            if header_match:
                if current_section != last_header_match.group(2) + "\n":
                    # Only add the node if the current section has contents
                    markdown_nodes.append(
                        self._build_node_from_split(
                            current_section.strip(), node, metadata
                        )
                    )

                last_header_match = header_match
                header_size = header_match.group(1)
                header_title = header_match.group(2)
                # Update metadata to new header
                metadata = self._update_metadata(
                    metadata, header_title, len(header_size)
                )

                # Start with next node
                current_section = f"{header_title}\n"
            else:
                current_section += par + "\n"

                if not self.max_node_length:
                    continue

                length_exceeded = len(current_section) >= self.max_node_length
                if length_exceeded:
                    # if the length is exceeded, we add the current paragraph and then
                    # commit the node. The next document will receive the same
                    # header as the one when the length was exceeded.
                    header_title = last_header_match.group(2)

                    markdown_nodes.append(
                        self._build_node_from_split(
                            current_section.strip(), node, metadata
                        )
                    )
                    # initiate new section
                    current_section = f"{header_title}\n"

        # commit the final node
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
        node = build_nodes_from_splits([text_split], node, id_func=self.id_func)[0]

        if self.include_metadata:
            node.metadata = {**node.metadata, **metadata}

        return node
