"""HTML node parser."""

from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Union

from llama_index.core.bridge.pydantic import Field
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.node_parser.interface import NodeParser
from llama_index.core.node_parser.node_utils import build_nodes_from_splits
from llama_index.core.schema import BaseNode, MetadataMode, TextNode
from llama_index.core.utils import get_tqdm_iterable

if TYPE_CHECKING:
    from bs4 import Tag, PageElement, NavigableString

DEFAULT_TAGS = ["p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "b", "i", "u", "section"]


class HTMLNodeParser(NodeParser):
    """
    HTML node parser.

    Splits a document into Nodes using custom HTML splitting logic.

    Args:
        include_metadata (bool): whether to include metadata in nodes
        include_prev_next_rel (bool): whether to include prev/next relationships

    """

    tags: List[str] = Field(
        default=DEFAULT_TAGS, description="HTML tags to extract text from."
    )

    @classmethod
    def from_defaults(
        cls,
        include_metadata: bool = True,
        include_prev_next_rel: bool = True,
        callback_manager: Optional[CallbackManager] = None,
        tags: Optional[List[str]] = DEFAULT_TAGS,
    ) -> "HTMLNodeParser":
        callback_manager = callback_manager or CallbackManager([])

        return cls(
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel,
            callback_manager=callback_manager,
            tags=tags,
        )

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "HTMLNodeParser"

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
        try:
            from bs4 import BeautifulSoup, Tag
        except ImportError:
            raise ImportError("bs4 is required to read HTML files.")

        text = node.get_content(metadata_mode=MetadataMode.NONE)
        soup = BeautifulSoup(text, "html.parser")
        # (section text, metadata) pairs, turned into nodes in one batch below
        sections: List[tuple[str, dict]] = []
        last_tag = None
        current_section = ""

        tags = soup.find_all(self.tags)
        for tag in tags:
            tag_text = self._extract_text_from_tag(tag)
            if isinstance(tag, Tag) and (tag.name == last_tag or last_tag is None):
                last_tag = tag.name
                current_section += f"{tag_text.strip()}\n"
            else:
                # Skip empty sections (e.g. a container tag whose only children
                # are themselves extracted tags) to avoid emitting blank nodes.
                if current_section.strip():
                    sections.append((current_section.strip(), {"tag": last_tag}))
                if isinstance(tag, Tag):
                    last_tag = tag.name
                current_section = f"{tag_text}\n"

        if current_section.strip():
            sections.append((current_section.strip(), {"tag": last_tag}))

        return self._build_nodes_from_sections(sections, node)

    def _extract_text_from_tag(
        self, tag: Union["Tag", "NavigableString", "PageElement"]
    ) -> str:
        from bs4 import NavigableString, Tag, PageElement

        texts = []
        if isinstance(tag, Tag):
            for elem in tag.children:
                if isinstance(elem, NavigableString):
                    if elem.strip():
                        texts.append(elem.strip())
                elif isinstance(elem, Tag):
                    if elem.name in self.tags:
                        continue
                    else:
                        texts.append(elem.get_text().strip())
                elif isinstance(elem, PageElement):
                    texts.append(elem.get_text().strip())
        else:
            texts.append(tag.get_text().strip())
        return "\n".join(texts)

    def _build_node_from_split(
        self,
        text_split: str,
        node: BaseNode,
        metadata: dict,
    ) -> TextNode:
        """Build node from single text split."""
        return self._build_nodes_from_sections([(text_split, metadata)], node)[0]

    def _build_nodes_from_sections(
        self,
        sections: List[tuple[str, dict]],
        node: BaseNode,
    ) -> List[TextNode]:
        """
        Build nodes from (text, metadata) sections.

        All nodes are built in a single call so that the source node's hash is
        computed once instead of once per section.
        """
        if type(self)._build_node_from_split is not HTMLNodeParser._build_node_from_split:
            return [
                self._build_node_from_split(text, node, metadata)
                for text, metadata in sections
            ]

        nodes = build_nodes_from_splits(
            [text for text, _ in sections], node, id_func=self.id_func
        )

        if self.include_metadata:
            for built_node, (_, metadata) in zip(nodes, sections):
                built_node.metadata = {**built_node.metadata, **metadata}

        return nodes
