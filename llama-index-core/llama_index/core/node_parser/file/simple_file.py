"""Simple file node parser."""
from typing import Any, Dict, List, Optional, Sequence, Type

from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.node_parser.node_utils import build_nodes_from_splits
from llama_index.core.node_parser.file.html import HTMLNodeParser
from llama_index.core.node_parser.file.json import JSONNodeParser
from llama_index.core.node_parser.file.markdown import MarkdownNodeParser
from llama_index.core.node_parser.interface import NodeParser
from llama_index.core.schema import BaseNode, MetadataMode
from llama_index.core.utils import get_tqdm_iterable

FILE_NODE_PARSERS: Dict[str, Type[NodeParser]] = {
    ".md": MarkdownNodeParser,
    ".html": HTMLNodeParser,
    ".json": JSONNodeParser,
}


class SimpleFileNodeParser(NodeParser):
    """Simple file node parser.

    Splits a document loaded from a file into Nodes using logic based on the file type
    automatically detects the NodeParser to use based on file type

    Args:
        include_metadata (bool): whether to include metadata in nodes
        include_prev_next_rel (bool): whether to include prev/next relationships

    """

    @classmethod
    def from_defaults(
        cls,
        include_metadata: bool = True,
        include_prev_next_rel: bool = True,
        callback_manager: Optional[CallbackManager] = None,
    ) -> "SimpleFileNodeParser":
        callback_manager = callback_manager or CallbackManager([])

        return cls(
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel,
            callback_manager=callback_manager,
        )

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "SimpleFileNodeParser"

    def _parse_nodes(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[BaseNode]:
        """Parse document into nodes.

        Args:
            nodes (Sequence[BaseNode]): nodes to parse
        """
        all_nodes: List[BaseNode] = []
        documents_with_progress = get_tqdm_iterable(
            nodes, show_progress, "Parsing documents into nodes"
        )

        for document in documents_with_progress:
            ext = document.metadata.get("extension", "None")
            if ext in FILE_NODE_PARSERS:
                parser = FILE_NODE_PARSERS[ext](
                    include_metadata=self.include_metadata,
                    include_prev_next_rel=self.include_prev_next_rel,
                    callback_manager=self.callback_manager,
                )

                nodes = parser.get_nodes_from_documents([document], show_progress)
                all_nodes.extend(nodes)
            else:
                # What to do when file type isn't supported yet?
                all_nodes.extend(
                    # build node from document
                    build_nodes_from_splits(
                        [document.get_content(metadata_mode=MetadataMode.NONE)],
                        document,
                        id_func=self.id_func,
                    )
                )

        return all_nodes
