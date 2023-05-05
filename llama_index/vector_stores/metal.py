import json
import math
from typing import Any, Dict, List, Optional
from llama_index.data_structs.node import DocumentRelationship, Node
from llama_index.vector_stores.types import (
    NodeWithEmbedding,
    VectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)


class MetalVectorStore(VectorStore):
    def __init__(
        self,
        api_key: str,
        client_id: str,
        index_id: str,
        filters: Optional[Dict[str, Any]] = None,
    ):
        """Init params."""
        import_err_msg = (
            "`metal_sdk` package not found, please run `pip install metal_sdk`"
        )
        try:
            import metal_sdk  # noqa: F401
        except ImportError:
            raise ImportError(import_err_msg)
        from metal_sdk.metal import Metal  # noqa: F401

        self.api_key = api_key
        self.client_id = client_id
        self.index_id = index_id
        self.filters = filters

        self.metal_client = Metal(api_key, client_id, index_id)
        self.stores_text = True
        self.is_embedding_query = True

    def query(self, query: VectorStoreQuery) -> VectorStoreQueryResult:
        payload = {
            "embedding": query.query_embedding,  # Query Embedding
            "filters": self.filters,  # Metadata Filters
        }
        response = self.metal_client.search(payload, limit=query.similarity_top_k)

        nodes = []
        ids = []
        similarities = []

        for item in response["data"]:
            text = item["text"]
            metadata = item["metadata"]
            ref_doc_id = metadata["doc_id"]
            if "extra_info" in metadata:
                extra_info = json.loads(metadata["extra_info"])
            id = item["id"]
            node = Node(
                text=text,
                extra_info=extra_info,
                doc_id=id,
                relationships={DocumentRelationship.SOURCE: ref_doc_id},
            )
            nodes.append(node)
            ids.append(item["id"])

            similarity_score = 1.0 - math.exp(-item["dist"])
            similarities.append(similarity_score)

        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)

    @property
    def client(self) -> Any:
        """Return Metal client."""
        return self.metal_client

    def add(self, embedding_results: List[NodeWithEmbedding]) -> List[str]:
        """Add embedding results to index.

        Args
            embedding_results: List[NodeEmbeddingResult]: list of embedding results

        """
        if not self.metal_client:
            raise ValueError("metal_client not initialized")

        ids = []
        for result in embedding_results:
            ids.append(result.id)

            metadata = {}
            metadata["doc_id"] = result.ref_doc_id
            metadata["text"] = result.node.get_text()
            if result.node.extra_info is not None:
                metadata["extra_info"] = json.dumps(result.node.extra_info)

            payload = {
                "embedding": result.embedding,
                "metadata": metadata,
                "id": result.id,
            }

            self.metal_client.index(payload)

        return ids

    def delete(self, doc_id: str, **delete_kwargs: Any) -> None:
        """Delete nodes from index.

        Args:
            doc_id (str): document id

        """
        if not self.metal_client:
            raise ValueError("metal_client not initialized")

        self.metal_client.deleteOne(doc_id)
