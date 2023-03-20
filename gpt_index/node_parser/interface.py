
from typing import List, Protocol, Set

from gpt_index.data_structs.data_structs import Node
from gpt_index.readers.schema.base import Document


class NodeParser(Protocol):
    def get_nodes_from_documents(
        self,
        documents: Set[Document],
        include_extra_info: bool = True,
    ) -> List[Node]:
        ...

    def get_nodes_from_document(
        self,
        document: Set[Document],
        include_extra_info: bool = True,
    ) -> List[Node]:
        ...
