import math
from typing import Any, List

from llama_index.schema import BaseNode, MetadataMode, TextNode
from llama_index.vector_stores.types import (
    MetadataFilters,
    VectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.vector_stores.utils import (
    legacy_metadata_dict_to_node,
    metadata_dict_to_node,
    node_to_metadata_dict,
)


def _to_metal_filters(standard_filters: MetadataFilters) -> list:
    filters = []
    for filter in standard_filters.legacy_filters():
        filters.append(
            {
                "field": filter.key,
                "value": filter.value,
            }
        )
    return filters


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
            import metal_sdk  # noqa
        except ImportError:
            raise ImportError(import_err_msg)
        from metal_sdk.metal import Metal

        self.api_key = api_key
        self.client_id = client_id
        self.index_id = index_id

        self.metal_client = Metal(api_key, client_id, index_id)
        self.stores_text = True
        self.flat_metadata = False
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
                node = metadata_dict_to_node(item["metadata"])
                node.text = text
            except Exception:
                # NOTE: deprecated legacy logic for backward compatibility
                metadata, node_info, relationships = legacy_metadata_dict_to_node(
                    item["metadata"]
                )

                node = TextNode(
                    text=text,
                    id_=id_,
                    metadata=metadata,
                    start_char_idx=node_info.get("start", None),
                    end_char_idx=node_info.get("end", None),
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

    def add(self, nodes: List[BaseNode], **add_kwargs: Any) -> List[str]:
        """Add nodes to index.

        Args:
            nodes: List[BaseNode]: list of nodes with embeddings.

        """
        if not self.metal_client:
            raise ValueError("metal_client not initialized")

        ids = []
        for node in nodes:
            ids.append(node.node_id)

            metadata = {}
            metadata["text"] = node.get_content(metadata_mode=MetadataMode.NONE) or ""

            additional_metadata = node_to_metadata_dict(
                node, remove_text=True, flat_metadata=self.flat_metadata
            )
            metadata.update(additional_metadata)

            payload = {
                "embedding": node.get_embedding(),
                "metadata": metadata,
                "id": node.node_id,
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
