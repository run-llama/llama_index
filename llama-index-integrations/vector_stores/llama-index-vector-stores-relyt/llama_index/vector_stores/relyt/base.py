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
from sqlalchemy import text

logger = logging.getLogger(__name__)
import_err_msg = (
    '`pgvecto_rs.sdk` package not found, please run `pip install "pgvecto_rs[sdk]"`'
)


class RelytVectorStore(BasePydanticVectorStore):
    """
    Relyt Vector Store.

    Examples:
        `pip install llama-index-vector-stores-relyt`

        ```python
        from llama_index.vector_stores.relyt import RelytVectorStore

        # Setup relyt client
        from pgvecto_rs.sdk import PGVectoRs
        import os

        URL = "postgresql+psycopg://{username}:{password}@{host}:{port}/{db_name}".format(
            port=os.getenv("RELYT_PORT", "5432"),
            host=os.getenv("RELYT_HOST", "localhost"),
            username=os.getenv("RELYT_USER", "postgres"),
            password=os.getenv("RELYT_PASS", "mysecretpassword"),
            db_name=os.getenv("RELYT_NAME", "postgres"),
        )

        client = PGVectoRs(
            db_url=URL,
            collection_name="example",
            dimension=1536,  # Using OpenAI’s text-embedding-ada-002
        )

        # Initialize RelytVectorStore
        vector_store = RelytVectorStore(client=client)
        ```

    """

    stores_text: bool = True

    _client: "PGVectoRs" = PrivateAttr()
    _collection_name: str = PrivateAttr()

    def __init__(self, client: "PGVectoRs", collection_name: str) -> None:
        super().__init__()

        self._client: PGVectoRs = client
        self._collection_name = collection_name
        self.init_index()

    @classmethod
    def class_name(cls) -> str:
        return "RelytStore"

    def init_index(self):
        index_name = f"idx_{self._collection_name}_embedding"
        with self._client._engine.connect() as conn:
            with conn.begin():
                index_query = text(
                    f"""
                        SELECT 1
                        FROM pg_indexes
                        WHERE indexname = '{index_name}';
                    """
                )
                result = conn.execute(index_query).scalar()
                if not result:
                    index_statement = text(
                        f"""
                            CREATE INDEX {index_name}
                            ON collection_{self._collection_name}
                            USING vectors (embedding vector_l2_ops)
                            WITH (options = $$
                            optimizing.optimizing_threads = 30
                            segment.max_growing_segment_size = 2000
                            segment.max_sealed_segment_size = 30000000
                            [indexing.hnsw]
                            m=30
                            ef_construction=500
                            $$);
                        """
                    )
                    conn.execute(index_statement)

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

    def drop(self) -> None:
        self._client.drop()

    # TODO: the more filter type(le, ne, ge ...) will add later, after the base api supported,
    #  now only support eq filter for meta information
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
