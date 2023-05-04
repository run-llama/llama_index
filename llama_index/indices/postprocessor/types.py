

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from llama_index.data_structs.node import NodeWithScore


class BaseNodePostprocessor(ABC):
    @abstractmethod
    def postprocess_nodes(
        self, nodes: List[NodeWithScore], extra_info: Optional[Dict] = None
    ) -> List[NodeWithScore]:
        """Postprocess nodes."""
        pass