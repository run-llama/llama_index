import logging
from typing import Any, List

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.schema import BaseNode, MetadataMode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.core.vector_stores.utils import (
    metadata_dict_to_node,
    node_to_metadata_dict,
)

from pgvecto_rs.sdk import PGVectoRs, Record
from pgvecto_rs.sdk.filters import meta_contains

logger = logging.getLogger(__name__)
import_err_msg = (
    '`pgvecto_rs.sdk` package not found, please run `pip install "pgvecto_rs[sdk]"`'
)


class PGVectoRsStore(BasePydanticVectorStore):
    """
    PGVectoRs Vector Store.

    Examples:
        `pip install llama-index-vector-stores-pgvecto-rs`

        ```python
        from llama_index.vector_stores.pgvecto_rs import PGVectoRsStore

        # Setup PGVectoRs client
        from pgvecto_rs.sdk import PGVectoRs
        import os

        URL = "postgresql+psycopg://{username}:{password}@{host}:{port}/{db_name}".format(
            port=os.getenv("DB_PORT", "5432"),
            host=os.getenv("DB_HOST", "localhost"),
            username=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASS", "mysecretpassword"),
            db_name=os.getenv("DB_NAME", "postgres"),
        )

        client = PGVectoRs(
            db_url=URL,
            collection_name="example",
            dimension=1536,  # Using OpenAI’s text-embedding-ada-002
        )

        # Initialize PGVectoRsStore
        vector_store = PGVectoRsStore(client=client)
        ```

    """

    stores_text: bool = True

    _client: "PGVectoRs" = PrivateAttr()

    def __init__(self, client: "PGVectoRs") -> None:
        super().__init__()
        self._client: PGVectoRs = client

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
        self._client.delete(meta_contains({"ref_doc_id": ref_doc_id}))

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        results = self._client.search(
            embedding=query.query_embedding,
            top_k=query.similarity_top_k,
            filter=(
                meta_contains(
                    {pair.key: pair.value for pair in query.filters.legacy_filters()}
                )
                if query.filters is not None
                else None
            ),
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
