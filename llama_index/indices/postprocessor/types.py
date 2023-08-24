import asyncio
from abc import ABC, abstractmethod
from typing import List, Optional

from llama_index.indices.query.schema import QueryBundle
from llama_index.schema import NodeWithScore


class BaseNodePostprocessor(ABC):

    def postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """Postprocess nodes."""
        return asyncio.get_running_loop().run_until_complete(
            self.apostprocess_nodes(nodes, query_bundle)
        )

    @abstractmethod
    async def apostprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """Postprocess nodes."""
        pass
