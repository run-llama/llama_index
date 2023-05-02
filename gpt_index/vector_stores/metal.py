from typing import Any, Dict, List, Optional
from gpt_index.data_structs.node_v2 import Node
from gpt_index.vector_stores.types import VectorStore, NodeEmbeddingResult, VectorStoreQuery, VectorStoreQueryResult


class MetalVectorStore(VectorStore):
    def __init__(self, api_key: str, client_id: str, index_id: str, filters: Optional[Dict[str, Any]] = None):
        """Init params."""
        import_err_msg = (
            "`metal_sdk` package not found, please run `pip install metal_sdk`"
        )
        try:
            import metal_sdk  # noqa: F401
        except ImportError:
            raise ImportError(import_err_msg)
        from metal_sdk.metal import Metal

        self.metal_client = Metal(api_key, client_id, index_id)
        self.stores_text = True
        self.is_embedding_query = True
        self.filters = filters

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "MetalVectorStore":
        return cls(
            api_key=config_dict["api_key"],
            client_id=config_dict["client_id"],
            index_id=config_dict["index_id"],
            filters=config_dict.get("filters", None)
        )

    # ... (other methods remain the same) ...

    def query(self, query: VectorStoreQuery) -> VectorStoreQueryResult:
        payload = {
            "text": query.query_str,  # Send text for query
            "filters": self.filters,  # Pass metadata filters
        }
        response = self.metal_client.search(payload, limit=query.similarity_top_k)

        nodes = []
        distances = []
        ids = []

        for item in response["results"]:
            node = Node.from_dict(item["metadata"])
            nodes.append(node)
            distances.append(item["dist"])
            ids.append(item["id"])

        return VectorStoreQueryResult(nodes=nodes, similarities=distances, ids=ids)
