from typing import List, Optional, Set

from gpt_index.data_structs.node_v2 import Node
from gpt_index.langchain_helpers.text_splitter import TextSplitter, TokenTextSplitter
from gpt_index.node_parser.node_utils import get_nodes_from_document
from gpt_index.readers.schema.base import Document


class SimpleNodeParser:
    def __init__(self, text_splitter: Optional[TextSplitter] = None) -> None:
        self._text_splitter = text_splitter or TokenTextSplitter()

    def get_nodes_from_documents(
        self,
        documents: Set[Document],
        include_extra_info: bool = True,
    ) -> List[Node]:
        all_nodes = []
        for document in documents:
            nodes = self.get_nodes_from_document(
                document, include_extra_info=include_extra_info
            )
            all_nodes.extend(nodes)
        return all_nodes

    def get_nodes_from_document(
        self,
        document: Document,
        include_extra_info: bool = True,
    ) -> List[Node]:
        return get_nodes_from_document(
            document, self._text_splitter, include_extra_info
        )
