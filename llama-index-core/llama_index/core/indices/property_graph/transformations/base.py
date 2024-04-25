from typing import Any, List

from llama_index.core.schema import TransformComponent, BaseNode


class BaseKGTransformation(TransformComponent):
    def run(self, nodes: List[BaseNode], **kwargs: Any) -> List[BaseNode]:
        nodes = self.__call__(nodes, **kwargs)
        for node in nodes:
            if "edges" not in node.metadata or "triplets" not in node.metadata:
                raise ValueError(
                    "KGTransformation must add 'edges' and/or 'triplets' to node metadata"
                )

        return nodes

    async def arun(self, nodes: List[BaseNode], **kwargs: Any) -> List[BaseNode]:
        nodes = await self.acall(nodes, **kwargs)
        for node in nodes:
            if "edges" not in node.metadata or "triplets" not in node.metadata:
                raise ValueError(
                    "KGTransformation must add 'edges' and/or 'triplets' to node metadata"
                )

        return nodes
