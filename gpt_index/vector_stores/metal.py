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
        from metal_sdk.metal import Metal   # noqa: F401


        self.api_key = api_key
        self.client_id = client_id
        self.index_id = index_id
        self.filters = filters

        self.metal_client = Metal(api_key, client_id, index_id)
        self.stores_text = True
        self.is_embedding_query = True

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "MetalVectorStore":
        return cls(
            api_key=config_dict["api_key"],
            client_id=config_dict["client_id"],
            index_id=config_dict["index_id"],
            filters=config_dict.get("filters", None)
        )

    def query(self, query: VectorStoreQuery) -> VectorStoreQueryResult:
        payload = {
            "text": query.query_str,  # Query Text
            "filters": self.filters,  # Metadata Filters
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

    @property
    def client(self) -> Any:
        """Return Metal client."""
        return self.metal_client

    @property
    def config_dict(self) -> dict:
        """Return config dict."""
        return {
            "api_key": self.api_key,
            "client_id": self.client_id,
            "index_id": self.index_id,
            "filters": self.filters
        }


    def add(self, embedding_results: List[NodeEmbeddingResult]) -> List[str]:
        """Add embedding results to index.

        Args
            embedding_results: List[NodeEmbeddingResult]: list of embedding results

        """
        if not self.metal_client:
            raise ValueError("metal_client not initialized")


        ids = []
        for result in embedding_results:
            ids.append(result.id)

            payload = {
                "text": result.node.get_text(),
                "metadata": result.node.extra_info or {},
            }

            if result.id:
                payload["id"] = result.id

            if result.doc_id:
                payload["metadata"]["document_id"] = result.doc_id

            self.metal_client.index(payload)

        return ids

    def delete(self, ids: List[str]) -> None:
        """Delete nodes from index.

        Args:
            ids (List[str]): list of node ids to delete

        """
        if not self.metal_client:
            raise ValueError("metal_client not initialized")

        for id in ids:
            self.metal_client.deleteOne(id)
