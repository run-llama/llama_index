from typing import Optional, List, Dict, Any, Type
from llama_index.core.schema import (
    NodeWithScore,
    TextNode,
    BaseNode,
    IndexNode,
    ImageNode,
)
from llama_index.core.base.response.schema import Response
from llama_index.core.bridge.pydantic import BaseModel
from pydantic import BaseModel as V2BaseModel

NODE_REGISTRY: Dict[str, Type[BaseNode]] = {
    "TextNode": TextNode,
    "IndexNode": IndexNode,
    "ImageNode": ImageNode,
}


class ContributorQueryRequest(V2BaseModel):
    query: str


class ContributorQueryResponse(BaseModel):
    response: Optional[str]
    score: Optional[float]

    def __str__(self) -> str:
        """Convert to string representation."""
        return self.response or "None"

    def to_response(self) -> Response:
        """Convert to Response type."""
        return Response(response=self.response, metadata={"score": self.score})


class ContributorRetrieverRequest(V2BaseModel):
    query: str


class ContributorRetrieverResponse(BaseModel):
    nodes_dict: Optional[List[Dict[str, Any]]]

    def get_nodes(self) -> List[NodeWithScore]:
        """Build list of nodes with score."""
        nodes = []
        for d in self.nodes_dict:
            node_dict = d["node"]
            try:
                node_cls = NODE_REGISTRY[node_dict["class_name"]]
            except KeyError:
                node_cls = NODE_REGISTRY["TextNode"]
            node = node_cls.parse_obj(node_dict)
            nodes.append(NodeWithScore(node=node, score=d["score"]))
        return nodes
