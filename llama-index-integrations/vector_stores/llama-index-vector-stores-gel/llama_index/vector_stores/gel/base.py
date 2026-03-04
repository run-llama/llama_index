import logging
from typing import (
    Any,
    List,
    Optional,
    Sequence,
)

from llama_index.core.schema import BaseNode, TextNode

from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    FilterOperator,
    FilterCondition,
    MetadataFilters,
    MetadataFilter,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)

from llama_index.core.bridge.pydantic import PrivateAttr

from jinja2 import Template
import json
import textwrap


_logger = logging.getLogger(__name__)


IMPORT_ERROR_MESSAGE = """
Error: Gel Python package is not installed.
Please install it using 'pip install gel'.
"""

NO_PROJECT_MESSAGE = """
Error: it appears that the Gel project has not been initialized.
If that's the case, please run 'gel project init' to get started.
"""

MISSING_RECORD_TYPE_TEMPLATE = """
Error: Record type {{record_type}} is missing from the Gel schema.

In order to use the LlamaIndex integration, ensure you put the following in dbschema/default.gel:

    using extension pgvector;

    module default {
        scalar type EmbeddingVector extending ext::pgvector::vector<1536>;

        type {{record_type}} {
            required collection: str;
            text: str;
            embedding: EmbeddingVector;
            external_id: str {
                constraint exclusive;
            };
            metadata: json;

            index ext::pgvector::hnsw_cosine(m := 16, ef_construction := 128)
                on (.embedding)
        }
    }

Remember that you also need to run a migration:

    $ gel migration create
    $ gel migrate

"""

try:
    import gel
except ImportError as e:
    _logger.error(IMPORT_ERROR_MESSAGE)
    raise


def format_query(text: str) -> Template:
    return Template(textwrap.dedent(text.strip()))


COSINE_SIMILARITY_QUERY = format_query(
    """
    with collection_records := (select {{record_type}} filter .collection = <str>$collection_name)
    select collection_records {
        external_id,
        text,
        embedding,
        metadata,
        cosine_similarity := 1 - ext::pgvector::cosine_distance(
            .embedding, <ext::pgvector::vector>$query_embedding),
    }
    {{filter_clause}}
    order by .cosine_similarity desc empty last
    limit <optional int64>$limit;
    """
)

SELECT_BY_DOC_ID_QUERY = format_query(
    """
    select {{record_type}} {
        external_id,
        text,
        embedding,
        metadata,
    }
    filter .external_id in array_unpack(<array<str>>$external_ids);
    """
)

INSERT_QUERY = format_query(
    """
    select (
        insert {{record_type}} {
            collection := <str>$collection_name,
            external_id := <optional str>$external_id,
            text := <str>$text,
            embedding := <ext::pgvector::vector>$embedding,
            metadata := <json>$metadata,
        }
    ) { external_id }
    """
)

DELETE_BY_IDS_QUERY = format_query(
    """
    with collection_records := (select {{record_type}} filter .collection = <str>$collection_name)
    delete {{record_type}}
    filter .external_id in array_unpack(<array<str>>$external_ids);
    """
)

DELETE_ALL_QUERY = format_query(
    """
    delete {{record_type}}
    filter .collection = <str>$collection_name;
    """
)


def get_filter_clause(filters: MetadataFilters) -> str:
    """Convert metadata filters to Gel query filter clause."""
    subclauses = []
    for filter in filters.filters:
        if isinstance(filter, MetadataFilters):
            subclause = get_filter_clause(filter)
        elif isinstance(filter, MetadataFilter):
            formatted_value = (
                f'"{filter.value}"' if isinstance(filter.value, str) else filter.value
            )
            if filter.operator == FilterOperator.EQ.value:
                subclause = (
                    f'<str>json_get(.metadata, "{filter.key}") = {formatted_value}'
                )
            elif filter.operator == FilterOperator.GT.value:
                subclause = (
                    f'<str>json_get(.metadata, "{filter.key}") > {formatted_value}'
                )
            elif filter.operator == FilterOperator.LT.value:
                subclause = (
                    f'<str>json_get(.metadata, "{filter.key}") < {formatted_value}'
                )
            elif filter.operator == FilterOperator.NE.value:
                subclause = (
                    f'<str>json_get(.metadata, "{filter.key}") != {formatted_value}'
                )
            elif filter.operator == FilterOperator.GTE.value:
                subclause = (
                    f'<str>json_get(.metadata, "{filter.key}") >= {formatted_value}'
                )
            elif filter.operator == FilterOperator.LTE.value:
                subclause = (
                    f'<str>json_get(.metadata, "{filter.key}") <= {formatted_value}'
                )
            elif filter.operator == FilterOperator.IN.value:
                subclause = f'<str>json_get(.metadata, "{filter.key}") in array_unpack({formatted_value})'
            elif filter.operator == FilterOperator.NIN.value:
                subclause = f'<str>json_get(.metadata, "{filter.key}") not in array_unpack({formatted_value})'
            elif filter.operator == FilterOperator.ANY.value:
                subclause = f'any(<str>json_get(.metadata, "{filter.key}") = array_unpack({formatted_value}))'
            elif filter.operator == FilterOperator.ALL.value:
                subclause = f'all(<str>json_get(.metadata, "{filter.key}") = array_unpack({formatted_value}))'
            elif filter.operator == FilterOperator.TEXT_MATCH.value:
                subclause = (
                    f'<str>json_get(.metadata, "{filter.key}") like {formatted_value}'
                )
            elif filter.operator == FilterOperator.CONTAINS.value:
                subclause = f'contains(<str>json_get(.metadata, "{filter.key}"), {formatted_value})'
            elif filter.operator == FilterOperator.IS_EMPTY.value:
                subclause = f'not exists <str>json_get(.metadata, "{filter.key}")'
            else:
                raise ValueError(f"Unknown operator: {filter.operator}")

        subclauses.append(subclause)

    if filters.condition == FilterCondition.AND:
        filter_clause = " and ".join(subclauses)
        return "(" + filter_clause + ")" if len(subclauses) > 1 else filter_clause
    elif filters.condition == FilterCondition.OR:
        filter_clause = " or ".join(subclauses)
        return "(" + filter_clause + ")" if len(subclauses) > 1 else filter_clause
    else:
        raise ValueError(f"Unknown condition: {filters.condition}")


class GelVectorStore(BasePydanticVectorStore):
    """
    Gel-backed vector store implementation.

    Stores and retrieves vectors using Gel database with pgvector extension.
    """

    stores_text: bool = True
    collection_name: str
    record_type: str

    _sync_client: gel.Client = PrivateAttr()
    _async_client: gel.AsyncIOClient = PrivateAttr()

    def __init__(
        self,
        collection_name: str = "default",
        record_type: str = "Record",
    ):
        """
        Initialize GelVectorStore.

        Args:
            collection_name: Name of the collection to store vectors in
            record_type: The record type name in Gel schema

        """
        super().__init__(
            collection_name=collection_name,
            record_type=record_type,
        )

        self._sync_client = None
        self._async_client = None

    def get_sync_client(self):
        """Get or initialize a synchronous Gel client."""
        if self._async_client is not None:
            raise RuntimeError(
                "GelVectorStore has already been used in async mode. "
                "If you were intentionally trying to use different IO modes at the same time, "
                "please create a new instance instead."
            )
        if self._sync_client is None:
            self._sync_client = gel.create_client()

            try:
                self._sync_client.ensure_connected()
            except gel.errors.ClientConnectionError as e:
                _logger.error(NO_PROJECT_MESSAGE)
                raise

            try:
                self._sync_client.query(f"select {self.record_type};")
            except gel.errors.InvalidReferenceError as e:
                _logger.error(
                    Template(MISSING_RECORD_TYPE_TEMPLATE).render(
                        record_type=self.record_type
                    )
                )
                raise

        return self._sync_client

    async def get_async_client(self):
        """Get or initialize an asynchronous Gel client."""
        if self._sync_client is not None:
            raise RuntimeError(
                "GelVectorStore has already been used in sync mode. "
                "If you were intentionally trying to use different IO modes at the same time, "
                "please create a new instance instead."
            )
        if self._async_client is None:
            self._async_client = gel.create_async_client()

            try:
                await self._async_client.ensure_connected()
            except gel.errors.ClientConnectionError as e:
                _logger.error(NO_PROJECT_MESSAGE)
                raise

            try:
                await self._async_client.query(f"select {self.record_type};")
            except gel.errors.InvalidReferenceError as e:
                _logger.error(
                    Template(MISSING_RECORD_TYPE_TEMPLATE).render(
                        record_type=self.record_type
                    )
                )
                raise

        return self._async_client

    @property
    def client(self) -> Any:
        """Get client."""
        return self.get_sync_client()

    def get_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
    ) -> List[BaseNode]:
        """Get nodes from vector store."""
        assert filters is None, "Filters are not supported in get_nodes"
        if node_ids is None:
            return []

        client = self.get_sync_client()

        results = client.query(
            SELECT_BY_DOC_ID_QUERY.render(record_type=self.record_type),
            external_ids=node_ids,
        )
        return [
            TextNode(
                id_=result.external_id,
                text=result.text,
                metadata=json.loads(result.metadata),
                embedding=result.embedding,
            )
            for result in results
        ]

    async def aget_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
    ) -> List[BaseNode]:
        """Async version of get_nodes."""
        assert filters is None, "Filters are not supported in get_nodes"
        if node_ids is None:
            return []

        client = await self.get_async_client()

        results = await client.query(
            SELECT_BY_DOC_ID_QUERY.render(record_type=self.record_type),
            external_ids=node_ids,
        )
        return [
            TextNode(
                id_=result.external_id,
                text=result.text,
                metadata=json.loads(result.metadata),
                embedding=result.embedding,
            )
            for result in results
        ]

    def add(
        self,
        nodes: Sequence[BaseNode],
        **kwargs: Any,
    ) -> List[str]:
        """Add nodes to vector store."""
        inserted_ids = []

        client = self.get_sync_client()

        for node in nodes:
            result = client.query(
                INSERT_QUERY.render(record_type=self.record_type),
                collection_name=self.collection_name,
                external_id=node.id_,
                text=node.get_content(),
                embedding=node.embedding,
                metadata=json.dumps(node.metadata),
            )
            inserted_ids.append(result[0].external_id)

        return inserted_ids

    async def async_add(self, nodes: Sequence[BaseNode], **kwargs: Any) -> List[str]:
        """Async version of add."""
        inserted_ids = []

        client = await self.get_async_client()

        for node in nodes:
            result = await client.query(
                INSERT_QUERY.render(record_type=self.record_type),
                collection_name=self.collection_name,
                external_id=node.id_,
                text=node.get_content(),
                embedding=node.embedding,
                metadata=json.dumps(node.metadata),
            )
            inserted_ids.append(result[0].external_id)

        return inserted_ids

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """Delete nodes using with ref_doc_id."""
        client = self.get_sync_client()

        result = client.query(
            DELETE_BY_IDS_QUERY.render(record_type=self.record_type),
            collection_name=self.collection_name,
            external_ids=[ref_doc_id],
        )

    async def adelete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """Async version of delete."""
        client = await self.get_async_client()

        result = await client.query(
            DELETE_BY_IDS_QUERY.render(record_type=self.record_type),
            collection_name=self.collection_name,
            external_ids=[ref_doc_id],
        )

    def clear(self) -> None:
        """Clear all nodes from configured vector store."""
        client = self.get_sync_client()

        result = client.query(
            DELETE_ALL_QUERY.render(record_type=self.record_type),
            collection_name=self.collection_name,
        )

    async def aclear(self) -> None:
        """Clear all nodes from configured vector store."""
        client = await self.get_async_client()

        result = await client.query(
            DELETE_ALL_QUERY.render(record_type=self.record_type),
            collection_name=self.collection_name,
        )

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Query vector store."""
        assert query.query_embedding is not None, "query_embedding is required"

        filter_clause = (
            "filter " + get_filter_clause(query.filters) if query.filters else ""
        )

        assert query.mode == VectorStoreQueryMode.DEFAULT

        rendered_query = COSINE_SIMILARITY_QUERY.render(
            record_type=self.record_type, filter_clause=filter_clause
        )

        client = self.get_sync_client()

        results = client.query(
            rendered_query,
            query_embedding=query.query_embedding,
            collection_name=self.collection_name,
            limit=query.similarity_top_k,
        )

        return VectorStoreQueryResult(
            nodes=[
                TextNode(
                    id_=result.external_id,
                    text=result.text,
                    metadata=json.loads(result.metadata),
                    embedding=result.embedding,
                )
                for result in results
            ],
            similarities=[result.cosine_similarity for result in results],
            ids=[result.external_id for result in results],
        )

    async def aquery(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> VectorStoreQueryResult:
        """Async version of query."""
        assert query.query_embedding is not None, "query_embedding is required"

        filter_clause = (
            "filter " + get_filter_clause(query.filters) if query.filters else ""
        )

        assert query.mode == VectorStoreQueryMode.DEFAULT

        rendered_query = COSINE_SIMILARITY_QUERY.render(
            record_type=self.record_type, filter_clause=filter_clause
        )

        client = await self.get_async_client()

        results = await client.query(
            rendered_query,
            query_embedding=query.query_embedding,
            collection_name=self.collection_name,
            limit=query.similarity_top_k,
        )

        return VectorStoreQueryResult(
            nodes=[
                TextNode(
                    id_=result.external_id,
                    text=result.text,
                    metadata=json.loads(result.metadata),
                    embedding=result.embedding,
                )
                for result in results
            ],
            similarities=[result.cosine_similarity for result in results],
            ids=[result.external_id for result in results],
        )

    def persist(self, persist_path: str, fs) -> None:
        _logger.warning("GelVectorStore.persist() is a no-op")
