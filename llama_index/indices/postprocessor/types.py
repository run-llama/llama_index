from abc import ABC, abstractmethod
from typing import List, Optional

from llama_index.data_structs.node import NodeWithScore
from llama_index.indices.query.schema import QueryBundle


class BaseNodePostprocessor(ABC):
    @abstractmethod
    def postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """Postprocess nodes."""
        pass
