"""Simple node parser."""
from typing import List, Optional, Sequence

from llama_index.data_structs.node import Node
from llama_index.langchain_helpers.text_splitter import TextSplitter, TokenTextSplitter
from llama_index.node_parser.node_utils import get_nodes_from_document
from llama_index.readers.schema.base import Document
from llama_index.node_parser.interface import NodeParser


class SimpleNodeParser(NodeParser):
    """Simple node parser.

    Splits a document into Nodes using a TextSplitter.

    Args:
        text_splitter (Optional[TextSplitter]): text splitter
        include_extra_info (bool): whether to include extra info in nodes
        include_prev_next_rel (bool): whether to include prev/next relationships

    """

    def __init__(
        self,
        text_splitter: Optional[TextSplitter] = None,
        include_extra_info: bool = True,
        include_prev_next_rel: bool = True,
    ) -> None:
        """Init params."""
        self._text_splitter = text_splitter or TokenTextSplitter()
        self._include_extra_info = include_extra_info
        self._include_prev_next_rel = include_prev_next_rel

    def get_nodes_from_documents(
        self,
        documents: Sequence[Document],
    ) -> List[Node]:
        """Parse document into nodes.

        Args:
            documents (Sequence[Document]): documents to parse
            include_extra_info (bool): whether to include extra info in nodes

        """
        all_nodes: List[Node] = []
        for document in documents:
            nodes = get_nodes_from_document(
                document,
                self._text_splitter,
                self._include_extra_info,
                include_prev_next_rel=self._include_prev_next_rel,
            )
            all_nodes.extend(nodes)
        return all_nodes
