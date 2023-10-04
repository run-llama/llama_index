"""Simple file node parser."""
from typing import Dict, List, Optional, Sequence, Type

from llama_index.bridge.pydantic import Field
from llama_index.callbacks.base import CallbackManager
from llama_index.callbacks.schema import CBEventType, EventPayload
from llama_index.node_parser.file.html import HTMLNodeParser
from llama_index.node_parser.file.json import JSONNodeParser
from llama_index.node_parser.file.markdown import MarkdownNodeParser
from llama_index.node_parser.interface import NodeParser
from llama_index.schema import BaseNode, Document
from llama_index.utils import get_tqdm_iterable

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

    def get_nodes_from_documents(
        self,
        documents: Sequence[Document],
        show_progress: bool = False,
    ) -> List[BaseNode]:
        """Parse document into nodes.

        Args:
            documents (Sequence[Document]): documents to parse
        """
        with self.callback_manager.event(
            CBEventType.NODE_PARSING, payload={EventPayload.DOCUMENTS: documents}
        ) as event:
            all_nodes: List[BaseNode] = []
            documents_with_progress = get_tqdm_iterable(
                documents, show_progress, "Parsing documents into nodes"
            )

            for document in documents_with_progress:
                ext = document.metadata["extension"]
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
                    all_nodes.extend(document)

            event.on_end(payload={EventPayload.NODES: all_nodes})

        return all_nodes
