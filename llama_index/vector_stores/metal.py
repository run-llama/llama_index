import json
import math
from typing import Any, Dict, List, Tuple

from llama_index.data_structs.node import DocumentRelationship, Node
from llama_index.vector_stores.types import (
    MetadataFilters,
    NodeWithEmbedding,
    VectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.vector_stores.utils import metadata_dict_to_node, node_to_metadata_dict


def _to_metal_filters(standard_filters: MetadataFilters) -> list:
    filters = []
    for filter in standard_filters.filters:
        filters.append(
            {
                "field": filter.key,
                "value": filter.value,
            }
        )
    return filters


def _legacy_metadata_dict_to_node(metadata: Dict[str, Any]) -> Tuple[dict, dict, dict]:
    if "extra_info" in metadata:
        extra_info = json.loads(metadata["extra_info"])
    else:
        extra_info = {}
    ref_doc_id = metadata["doc_id"]
    relationships = {DocumentRelationship.SOURCE: ref_doc_id}
    node_info: dict = {}
    return extra_info, node_info, relationships


class MetalVectorStore(VectorStore):
    def __init__(
        self,
        api_key: str,
        client_id: str,
        index_id: str,
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

        self.metal_client = Metal(api_key, client_id, index_id)
        self.stores_text = True
        self.is_embedding_query = True

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        if query.filters is not None:
            if "filters" in kwargs:
                raise ValueError(
                    "Cannot specify filter via both query and kwargs. "
                    "Use kwargs only for metal specific items that are "
                    "not supported via the generic query interface."
                )
            filters = _to_metal_filters(query.filters)
        else:
            filters = kwargs.get("filters", {})

        payload = {
            "embedding": query.query_embedding,  # Query Embedding
            "filters": filters,  # Metadata Filters
        }
        response = self.metal_client.search(payload, limit=query.similarity_top_k)

        nodes = []
        ids = []
        similarities = []

        for item in response["data"]:
            text = item["text"]
            id_ = item["id"]

            # load additional Node data
            try:
                extra_info, node_info, relationships = metadata_dict_to_node(
                    item["metadata"]
                )
            except Exception:
                # NOTE: deprecated legacy logic for backward compatibility
                extra_info, node_info, relationships = _legacy_metadata_dict_to_node(
                    item["metadata"]
                )

            node = Node(
                text=text,
                doc_id=id_,
                extra_info=extra_info,
                node_info=node_info,
                relationships=relationships,
            )
            nodes.append(node)
            ids.append(id_)

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
            metadata["text"] = result.node.text or ""

            additional_metadata = node_to_metadata_dict(result.node)
            metadata.update(additional_metadata)

            payload = {
                "embedding": result.embedding,
                "metadata": metadata,
                "id": result.id,
            }

            self.metal_client.index(payload)

        return ids

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The doc_id of the document to delete.

        """
        if not self.metal_client:
            raise ValueError("metal_client not initialized")

        self.metal_client.deleteOne(ref_doc_id)
