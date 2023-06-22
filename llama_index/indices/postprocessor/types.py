from abc import ABC, abstractmethod
from typing import List, Optional

from llama_index.indices.query.schema import QueryBundle
from llama_index.schema import NodeWithScore


class BaseNodePostprocessor(ABC):
    @abstractmethod
    def postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """Postprocess nodes."""
        pass
