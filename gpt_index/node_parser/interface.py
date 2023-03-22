from typing import List, Protocol, Sequence

from gpt_index.data_structs.node_v2 import Node
from gpt_index.readers.schema.base import Document


class NodeParser(Protocol):
    def get_nodes_from_documents(
        self,
        documents: Sequence[Document],
        include_extra_info: bool = True,
    ) -> List[Node]:
        ...