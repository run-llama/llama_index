"""Simple node parser."""
from typing import List, Optional, Sequence

from gpt_index.data_structs.node_v2 import Node
from gpt_index.langchain_helpers.text_splitter import TextSplitter, TokenTextSplitter
from gpt_index.node_parser.node_utils import get_nodes_from_document
from gpt_index.readers.schema.base import Document


class SimpleNodeParser:
    """Simple node parser.

    Splits a document into Nodes using a TextSplitter.

    Args:
        text_splitter (Optional[TextSplitter]): text splitter
        include_extra_info (bool): whether to include extra info in nodes

    """

    def __init__(
        self,
        text_splitter: Optional[TextSplitter] = None,
        include_extra_info: bool = True,
    ) -> None:
        """Init params."""
        self._text_splitter = text_splitter or TokenTextSplitter()
        self._include_extra_info = include_extra_info

    def get_nodes_from_documents(
        self,
        documents: Sequence[Document],
        include_extra_info: bool = True,
    ) -> List[Node]:
        """Parse document into nodes.

        Args:
            documents (Sequence[Document]): documents to parse
            include_extra_info (bool): whether to include extra info in nodes

        """
        all_nodes: List[Node] = []
        for document in documents:
            nodes = get_nodes_from_document(
                document, self._text_splitter, include_extra_info
            )
            all_nodes.extend(nodes)
        return all_nodes
