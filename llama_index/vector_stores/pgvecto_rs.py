import logging
from typing import Any, List

from llama_index.bridge.pydantic import PrivateAttr
from llama_index.schema import BaseNode
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


class PGVectoRsStore(BasePydanticVectorStore):
    stores_text = True

    _client = PrivateAttr()

    def __init__(self, pgvecto_rs: Any) -> None:
        try:
            from pgvecto_rs.sdk import PGVectoRs
        except ImportError:
            raise ImportError(import_err_msg)
        self._client: PGVectoRs = pgvecto_rs
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
                text="",
                meta=node_to_metadata_dict(node),
                embedding=node.get_embedding(),
            )
            for node in nodes
        ]

        self._client.insert(records)
        return [node.id_ for node in nodes]

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        self._client.delete_by_ids((ref_doc_id,))

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        from pgvecto_rs.sdk.filters import meta_contains

        results = self._client.search(
            embedding=query.query_embedding,
            top_k=query.similarity_top_k,
            filter=meta_contains(
                {pair.key: pair.value for pair in query.filters.filters}
            )
            if query.filters is not None
            else None,
        )
        return VectorStoreQueryResult(
            nodes=[metadata_dict_to_node(record.meta) for record, _score in results],
            similarities=[score for _record, score in results],
            ids=[str(record.id) for record, _score in results],
        )
