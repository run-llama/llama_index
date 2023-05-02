from typing import Any, Dict, List, Optional
from gpt_index.data_structs.node_v2 import Node
from gpt_index.vector_stores.types import VectorStore, NodeEmbeddingResult, VectorStoreQuery, VectorStoreQueryResult


class MetalVectorStore(VectorStore):
    def __init__(self, api_key: str, client_id: str, index_id: str):
        """Init params."""
        import_err_msg = (
            "`metal_sdk` package not found, please run `pip install metal_sdk`"
        )
        try:
            import metal_sdk  # noqa: F401
        except ImportError:
            raise ImportError(import_err_msg)
        from metal_sdk import Metal


        self.metal_client = Metal(api_key, client_id, index_id)
        self.stores_text = True
        self.is_embedding_query = True

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "MetalVectorStore":
        return cls(
            api_key=config_dict["api_key"],
            client_id=config_dict["client_id"],
            index_id=config_dict["index_id"]
        )

    @property
    def client(self) -> Any:
        return self.metal_client

    @property
    def config_dict(self) -> dict:
        return {
            "api_key": self.metal_client.api_key,
            "client_id": self.metal_client.client_id,
            "index_id": self.metal_client.index_id
        }

    def add(self, embedding_results: List[NodeEmbeddingResult]) -> List[str]:
        ids = []
        for result in embedding_results:
            payload = {
                "id": result.id,
                "metadata": result.node.to_dict(),
                "text": result.node.text,  # Send text instead of embedding
            }
            response = self.metal_client.index(payload)
            ids.append(response["id"])
        return ids

    def delete(self, doc_id: str, **delete_kwargs: Any) -> None:
        self.metal_client.delete_one(doc_id)

    def query(self, query: VectorStoreQuery) -> VectorStoreQueryResult:
        # todo: add filters to query?
        payload = {
            "text": query.query_str,
            # "filters": {"ids": query.doc_ids} if query.doc_ids else None
        }
        response = self.metal_client.search(payload, limit=query.similarity_top_k)

        nodes = []
        distances = []
        ids = []

        for item in response["results"]:
            node = Node.from_dict(item["metadata"])
            nodes.append(node)
            distances.append(item["dist"])  # Use 'dist' instead of 'similarity'
            ids.append(item["id"])

        # Return distances instead of similarities
        return VectorStoreQueryResult(nodes=nodes, similarities=distances, ids=ids)
