from typing import Optional, List, Dict, Any, Type
from llama_index.core.schema import (
    NodeWithScore,
    TextNode,
    BaseNode,
    IndexNode,
    ImageNode,
)
from llama_index.core.base.response.schema import Response
from llama_index.core.bridge.pydantic import BaseModel, Field

NODE_REGISTRY: Dict[str, Type[BaseNode]] = {
    "TextNode": TextNode,
    "IndexNode": IndexNode,
    "ImageNode": ImageNode,
}


class ContributorQueryRequest(BaseModel):
    query: str


class ContributorQueryResponse(BaseModel):
    response: Optional[str] = Field(default=None)
    score: Optional[float] = Field(default=None)

    def __str__(self) -> str:
        """Convert to string representation."""
        return self.response or "None"

    def to_response(self) -> Response:
        """Convert to Response type."""
        return Response(response=self.response, metadata={"score": self.score})


class ContributorRetrieverRequest(BaseModel):
    query: str


class ContributorRetrieverResponse(BaseModel):
    nodes_dict: Optional[List[Dict[str, Any]]] = Field(default=None)

    def get_nodes(self) -> List[NodeWithScore]:
        """Build list of nodes with score."""
        nodes = []
        for d in self.nodes_dict:
            node_dict = d["node"]
            try:
                node_cls = NODE_REGISTRY[node_dict["class_name"]]
            except KeyError:
                node_cls = NODE_REGISTRY["TextNode"]
            node = node_cls.model_validate(node_dict)
            nodes.append(NodeWithScore(node=node, score=d["score"]))
        return nodes
