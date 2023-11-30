import logging
from typing import TYPE_CHECKING, Any, List

from llama_index.bridge.pydantic import PrivateAttr
from llama_index.schema import BaseNode, MetadataMode
from llama_index.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.vector_stores.utils import metadata_dict_to_node, node_to_metadata_dict

logger = logging.getLogger(__name__)
import_err_msg = (
    '`pgvecto_rs.sdk` package not found, please run `pip install "pgvecto_rs[sdk]"`'
)

if TYPE_CHECKING:
    from pgvecto_rs.sdk import PGVectoRs


class PGVectoRsStore(BasePydanticVectorStore):
    stores_text = True

    _client: "PGVectoRs" = PrivateAttr()

    def __init__(self, client: "PGVectoRs") -> None:
        try:
            from pgvecto_rs.sdk import PGVectoRs
        except ImportError:
            raise ImportError(import_err_msg)
        self._client: PGVectoRs = client
        super().__init__()

    @classmethod
    def class_name(cls) -> str:
        return "PGVectoRsStore"

    @property
    def client(self) -> Any:
        return self._client

    def add(
        self,
        nodes: List[BaseNode],
    ) -> List[str]:
        from pgvecto_rs.sdk import Record

        records = [
            Record(
                id=node.id_,
                text=node.get_content(metadata_mode=MetadataMode.NONE),
                meta=node_to_metadata_dict(node, remove_text=True),
                embedding=node.get_embedding(),
            )
            for node in nodes
        ]

        self._client.insert(records)
        return [node.id_ for node in nodes]

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        from pgvecto_rs.sdk.filters import meta_contains

        self._client.delete(meta_contains({"ref_doc_id": ref_doc_id}))

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        from pgvecto_rs.sdk.filters import meta_contains

        results = self._client.search(
            embedding=query.query_embedding,
            top_k=query.similarity_top_k,
            filter=meta_contains(
                {pair.key: pair.value for pair in query.filters.legacy_filters()}
            )
            if query.filters is not None
            else None,
        )

        nodes = [
            metadata_dict_to_node(record.meta, text=record.text)
            for record, _ in results
        ]

        return VectorStoreQueryResult(
            nodes=nodes,
            similarities=[score for _, score in results],
            ids=[str(record.id) for record, _ in results],
        )
