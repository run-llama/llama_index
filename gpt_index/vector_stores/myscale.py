"""MyScale vector store.

An index that is built on top of an existing MyScale cluster.

"""
import json
import logging
from typing import Any, Dict, List, Optional, cast

from gpt_index.data_structs.node_v2 import DocumentRelationship, Node
from gpt_index.embeddings.base import BaseEmbedding
from gpt_index.indices.service_context import ServiceContext
from gpt_index.readers.myscale import (
    MyScaleSettings,
    _build_query_statement,
    _escape_str,
)
from gpt_index.utils import iter_batch
from gpt_index.vector_stores.types import (
    NodeEmbeddingResult,
    VectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)

logger = logging.getLogger(__name__)


class MyscaleVectorStore(VectorStore):
    """MyScale Vector Store.

    In this vector store, embeddings and docs are stored within an existing
    MyScale cluster.

    During query time, the index uses MyScale to query for the top
    k most similar nodes.

    Args:
        collection_name: (str): name of the Qdrant collection
        client (Optional[Any]): QdrantClient instance from `qdrant-client` package

    """

    stores_text: bool = True
    _index_existed: bool = False

    def __init__(
        self,
        table: str = "llama_index",
        database: str = "default",
        index_type: str = "IVFFLAT",
        metric: str = "cosine",
        index_params: Optional[dict] = None,
        myscale_client: Optional[Any] = None,
        service_context: Optional[ServiceContext] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        import_err_msg = "`clickhouse_connect` package not found, please run `pip install clickhouse-connect`"
        try:
            from clickhouse_connect.driver.httpclient import HttpClient  # noqa: F401
        except ImportError:
            raise ImportError(import_err_msg)

        if myscale_client is None:
            raise ValueError("Missing MyScale client!")

        self._client = cast(HttpClient, myscale_client)
        self.config = MyScaleSettings(
            table=table,
            database=database,
            index_type=index_type,
            metric=metric,
            index_params=index_params,
        )
        self._batch_size = 32

        if service_context is not None:
            service_context = cast(ServiceContext, service_context)
            dimension = len(
                service_context.embed_model.get_query_embedding("try this out")
            )
            self._create_index(dimension)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "VectorStore":
        return cls(**config_dict)

    @property
    def client(self) -> Any:
        """Get client."""
        return self._client

    @property
    def config_dict(self) -> dict:
        """Return config dict."""
        return self.config.__dict__

    def _create_index(self, dimension: int) -> None:
        index_params = (
            ", " + ",".join([f"'{k}={v}'" for k, v in self.config.index_params.items()])
            if self.config.index_params
            else ""
        )
        schema_ = f"""
            CREATE TABLE IF NOT EXISTS {self.config.database}.{self.config.table}(
                id String,
                doc_id String,
                text String,
                vector Array(Float32),
                node_info JSON,
                extra_info JSON,
                CONSTRAINT vector_length CHECK length(vector) = {dimension},
                VECTOR INDEX {self.config.table}_index vector TYPE {self.config.index_type}('metric_type={self.config.metric}'{index_params})
            ) ENGINE = MergeTree ORDER BY id
            """
        self.dim = dimension
        self._client.command("SET allow_experimental_object_type=1")
        self._client.command(schema_)
        self._index_existed = True

    def _build_insert_statement(
        self, values: List[NodeEmbeddingResult], column_names: List[str]
    ) -> str:
        columns = ",".join(column_names)
        _data = []
        for item in values:
            item_embedding = ",".join(str(num) for num in item.embedding)
            # escape quota in document text
            item_text = _escape_str(item.node.text)
            item_value_str = f"""('{item.id}', '{item.doc_id}', '{item_text}', [{item_embedding}], 
            '{json.dumps(item.node.node_info)}', '{json.dumps(item.node.extra_info)}')"""
            _data.append(item_value_str)
        insert_statement = f"""
                INSERT INTO TABLE 
                    {self.config.database}.{self.config.table}({columns})
                VALUES
                {','.join(_data)}
                """
        return insert_statement

    def add(
        self,
        embedding_results: List[NodeEmbeddingResult],
    ) -> List[str]:
        """Add embedding results to index.

        Args
            embedding_results: List[NodeEmbeddingResult]: list of embedding results

        """

        if not self._index_existed:
            self._create_index(len(embedding_results))

        column_names = ["id", "doc_id", "text", "vector", "node_info", "extra_info"]
        for result_batch in iter_batch(embedding_results, self._batch_size):
            insert_statement = self._build_insert_statement(
                values=result_batch, column_names=column_names
            )
            self._client.command(insert_statement)

        return [result.id for result in embedding_results]

    def delete(self, doc_id: str, **delete_kwargs: Any) -> None:
        """Delete a document.

        Args:
            doc_id (str): document id

        """
        self._client.commend(
            f"""DELETE FROM TABLE {self.config.database}.{self.config.table} WHERE doc_id = '{doc_id}'"""
        )

    def query(self, query: VectorStoreQuery) -> VectorStoreQueryResult:
        """Query index for top k most similar nodes."""

        """Perform a similarity search with MyScale by vectors
        Args:
            query (str): query string
            k (int, optional): Top K neighbors to retrieve. Defaults to 4.
            where_str (Optional[str], optional): where condition string. Defaults to None.
            NOTE: Please do not let end-user to fill this out and always be aware of SQL injection.
                  When dealing with metadatas, remeber to use `{metadata-name-you-set}.attribute` 
                  instead of `attribute` alone. The default name for metadat column is `metadata`.
        Returns:
            List[Document]: List of (Document, similarity)
        """

        query_embedding = cast(List[float], query.query_embedding)

        where_str = None
        if query.doc_ids is not None:
            doc_id_filter = ",".join(str(id) for id in query.doc_ids)
            where_str = f"doc_id in [{doc_id_filter}]"

        query_statement = _build_query_statement(
            config=self.config,
            query_embed=query_embedding,
            where_str=where_str,
            limit=query.similarity_top_k,
        )

        try:
            nodes = []
            ids = []
            similarities = []
            for r in self._client.query(query_statement).named_results():
                node = Node(
                    doc_id=r["doc_id"],
                    text=r["text"],
                    extra_info=r["extra_info"],
                    node_info=r["node_info"],
                    relationships={DocumentRelationship.SOURCE: r["doc_id"]},
                )

                nodes.append(node)
                similarities.append(r["dist"])
                ids.append(r["id"])
            return VectorStoreQueryResult(
                nodes=nodes, similarities=similarities, ids=ids
            )
        except Exception as e:
            logger.error(f"\033[91m\033[1m{type(e)}\033[0m \033[95m{str(e)}\033[0m")
            raise e
