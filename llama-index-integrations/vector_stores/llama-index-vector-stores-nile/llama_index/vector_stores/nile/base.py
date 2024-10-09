# Standard library imports
import json
import logging
import uuid
from typing import Any, List

# Third-party imports
import psycopg
from psycopg import sql

# Local application/library specific imports
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.constants import DEFAULT_EMBEDDING_DIM
from llama_index.core.schema import BaseNode, MetadataMode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    MetadataFilters,
    FilterOperator,
    MetadataFilter,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)
from llama_index.core.vector_stores.utils import (
    metadata_dict_to_node,
    node_to_metadata_dict,
)

_logger = logging.getLogger(__name__)

class NileVectorStore(BasePydanticVectorStore):
    """Nile Vector Store

    Examples [TBD]:
        `pip install llama-index-vector-stores-nile`

        ```python
        from llama_index.vector_stores.nile import NileVectorStore

        # Create NileVectorStore instance
        vector_store = NileVectorStore.from_params(
            service_url="https://api.nile.xyz",
            table_name="your_table_name_here",
    """
    stores_text: bool = True
    flat_metadata: bool = False

    service_url: str
    table_name: str
    num_dimensions: int
    tenant_aware: bool
    
    _sync_conn: Any = PrivateAttr()
    _async_conn: Any = PrivateAttr()
    
    def _create_clients(self) -> None:
        self._sync_conn = psycopg.connect(self.service_url)
        # NOTE: we can't await in this function since it is called during __init__, may need to move this elsewhere
        self._async_conn = psycopg.connect(self.service_url)
        
    def _create_tables(self) -> None:
        _logger.info(f"Creating tables for {self.table_name} with {self.num_dimensions} dimensions")
        with self._sync_conn.cursor() as cursor:
            if self.tenant_aware:
                query = sql.SQL('''
                                CREATE TABLE IF NOT EXISTS {table_name} 
                                (id UUID DEFAULT (gen_random_uuid()), tenant_id UUID, embedding VECTOR({num_dimensions}), content TEXT, metadata JSONB)
                                ''').format(
                                    table_name=sql.Identifier(self.table_name),
                                    num_dimensions=sql.Literal(self.num_dimensions)
                                )
                cursor.execute(query)
            else:
                query = sql.SQL('''
                                CREATE TABLE IF NOT EXISTS {table_name} 
                                (id UUID DEFAULT (gen_random_uuid()), embedding VECTOR(num_dimensions}), content TEXT, metadata JSONB)
                                ''').format(
                                    table_name=sql.Identifier(self.table_name),
                                    num_dimensions=sql.Literal(self.num_dimensions)
                                )
                cursor.execute(query)
        self._sync_conn.commit()
    
    # NOTE: Maybe allow specifying schema name
    # TODO: Allow specifying index type and parameters
    def __init__(self, service_url: str, table_name: str, tenant_aware: bool = False, num_dimensions: int = DEFAULT_EMBEDDING_DIM) -> None:
        # TODO: Do we need lower case table name? do we want to add prefix?
        super().__init__(service_url=service_url, table_name=table_name, num_dimensions=num_dimensions, tenant_aware=tenant_aware)
        
        self._create_clients()
        self._create_tables()
        
    @classmethod
    def class_name(cls) -> str:
        return "NileVectorStore"
    
    @property
    def client(self) -> Any:
        return self._sync_conn

    async def close(self) -> None:
        self._sync_conn.close()
        await self._async_conn.close()
        
    @classmethod
    def from_params(cls, service_url: str, table_name: str, tenant_aware: bool = False, num_dimensions: int = DEFAULT_EMBEDDING_DIM) -> "NileVectorStore":
        return cls(service_url=service_url, table_name=table_name, tenant_aware=tenant_aware, num_dimensions=num_dimensions)
    
    # We extract tenant_id from the node metadata. 
    # NOTE: Maybe we should also allow passing tenant_id in kwargs.
    def _node_to_row(self, node: BaseNode) -> Any:
        metadata = node_to_metadata_dict(
            node,
            remove_text=True,
            flat_metadata=self.flat_metadata,
        )
        tenant_id = node.metadata.get("tenant_id", None)
        return [tenant_id, metadata, node.get_content(metadata_mode=MetadataMode.NONE), node.embedding]
    
    def _insert_row(self, cursor: Any, row: Any) -> str:
        _logger.info(f"Inserting row into {self.table_name} with tenant_id {row[0]}")
        if self.tenant_aware:
            query = sql.SQL("""
                           INSERT INTO {} (tenant_id, metadata, content, embedding) VALUES (%(tenant_id)s, %(metadata)s, %(content)s, %(embedding)s) returning id
                       """).format(sql.Identifier(self.table_name))
            cursor.execute(query, {'tenant_id': row[0], 'metadata': json.dumps(row[1]), 'content': row[2], 'embedding': row[3]})    
        else:
            query = sql.SQL("""
                           INSERT INTO {} (metadata, content, embedding) VALUES (%(metadata)s, %(content)s, %(embedding)s) returning id
                       """).format(sql.Identifier(self.table_name))
            cursor.execute(query, {'metadata': json.dumps(row[0]), 'content': row[1], 'embedding': row[2]})
        id = cursor.fetchone()[0]
        self._sync_conn.commit()
        return id
    
    def add(self, nodes: List[BaseNode], **add_kwargs: Any) -> List[str]:
        rows_to_insert = [self._node_to_row(node) for node in nodes]
        ids = []
        with self._sync_conn.cursor() as cursor:
            # TODO: Use batch insert
            for row in rows_to_insert:
                # this will throw an error if tenant_id is None, which is what we want
                ids.append(self._insert_row(cursor, row))
            self._sync_conn.commit()
        return ids
        
    async def async_add(self, nodes: List[BaseNode], **add_kwargs: Any) -> List[str]:
        rows_to_insert = [self._node_to_row(node) for node in nodes]
        ids = []
        async with self._async_conn.cursor() as cursor:
            for row in rows_to_insert:
                ids.append(self._insert_row(cursor, row))
            await self._async_conn.commit()
        return ids
    
    def _set_tenant_context(self, cursor: Any, tenant_id: Any) -> None:
        if self.tenant_aware:
            cursor.execute(sql.SQL(""" set nile.tenant_id = {} """).format(sql.Literal(tenant_id)))
        else:
            cursor.execute(sql.SQL(""" reset nile.tenant_id """))

    def _to_postgres_operator(self, operator: FilterOperator) -> str:
        if operator == FilterOperator.EQ:
            return "="
        elif operator == FilterOperator.GT:
            return ">"
        elif operator == FilterOperator.LT:
            return "<"
        elif operator == FilterOperator.NE:
            return "!="
        elif operator == FilterOperator.GTE:
            return ">="
        elif operator == FilterOperator.LTE:
            return "<="
        elif operator == FilterOperator.IN:
            return "IN"
        elif operator == FilterOperator.NIN:
            return "NOT IN"
        elif operator == FilterOperator.CONTAINS:
            return "@>"
        else:
            _logger.warning(f"Unknown operator: {operator}, fallback to '='")
            return "="
        
    def _create_where_clause(self, filters: MetadataFilters) -> None:
        where_clauses = []
        if filters is None:
            return sql.SQL(""" """)
        _logger.debug(f"Filters: {filters}")
        for filter in filters.filters:
            if isinstance(filter, MetadataFilters):
                raise ValueError("Nested MetadataFilters are not supported yet")
            if isinstance(filter, MetadataFilter):
                # The string concat looks terrible, but is in fact safe from SQL injection since we use "=" 
                # to replace any unknown operator. Unfortuantely, we can't use psycopg's sql.Literal or sql.Identifier 
                # for the operator since we need to leave it unquoted.
                if filter.operator in [FilterOperator.IN, FilterOperator.NIN]:
                    where_clauses.append(sql.SQL(" metadata->>{} " + 
                                                 self._to_postgres_operator(filter.operator) + 
                                                 " ({})").format(sql.Literal(filter.key), 
                                                                    self._to_postgres_operator(filter.operator), 
                                                                    sql.Literal(filter.value)))
                elif filter.operator in [FilterOperator.CONTAINS]:
                    where_clauses.append(sql.SQL(""" metadata->{} @> [{}]""").format(sql.Literal(filter.key), sql.Literal(filter.value)))
                else:
                    where_clauses.append(sql.SQL(" metadata->>{} " + 
                                                 self._to_postgres_operator(filter.operator) + 
                                                 " {}").format(sql.Literal(filter.key),
                                                                sql.Literal(filter.value)))
        _logger.debug(f"Where clauses: {where_clauses}")
        if len(where_clauses) == 0:
            return sql.SQL(""" """)
        else:
            return sql.SQL(""" WHERE {}""").format(sql.SQL(filters.condition).join(where_clauses))

    # TODO: Add support for vector index GUC
    # NOTE: Add support for nested filters
    # NOTE: Maybe support alternative distance functions (going with just cosine similarity for now)
    def _execute_query(self, cursor: Any, query_embedding: VectorStoreQuery, tenant_id: Any = None) -> List[Any]:
        _logger.info(f"Querying {self.table_name} with tenant_id {tenant_id}")
        self._set_tenant_context(cursor, tenant_id)
        where_clause = self._create_where_clause(query_embedding.filters)
        query = sql.SQL("""
            SELECT
            id, metadata, content, %(query_embedding)s::vector<=>embedding as distance
            FROM
            {table_name}
            {where_clause}
            ORDER BY distance
            LIMIT {limit}
            """).format(
                table_name=sql.Identifier(self.table_name),
                where_clause=where_clause,
                limit=sql.Literal(query_embedding.similarity_top_k)
            )
        _logger.debug(f"Executing query: {query.as_string(cursor)}")
        cursor.execute(query, {'query_embedding': query_embedding.query_embedding})
        return cursor.fetchall()
    
    
    def _process_query_results(self, results: List[Any]) -> VectorStoreQueryResult:
        nodes = []
        similarities = []
        ids = []
        for row in results:
            node = metadata_dict_to_node(row[1])
            node.set_content(row[2])
            nodes.append(node)
            similarities.append(row[3])
            ids.append(row[0])
        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)

    
    # NOTE: Maybe handle tenant_id specified in filter vs. kwargs
    # NOTE: Add support for additional query modes
    def query(self, query_embedding: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        # get and validate tenant_id
        tenant_id = kwargs.get("tenant_id", None)
        if self.tenant_aware and tenant_id is None:
            raise ValueError("tenant_id must be specified in kwargs if tenant_aware is True")
        # check query mode
        if query_embedding.mode != VectorStoreQueryMode.DEFAULT:
            raise ValueError("Only DEFAULT mode is currently supported")
        # query
        with self._sync_conn.cursor() as cursor:
            self._set_tenant_context(cursor, tenant_id)
            results = self._execute_query(cursor, query_embedding, tenant_id)
        self._sync_conn.commit()
        return self._process_query_results(results)

    async def aquery(self, query_embedding: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        tenant_id = kwargs.get("tenant_id", None)
        if self.tenant_aware and tenant_id is None:
            raise ValueError("tenant_id must be specified in kwargs if tenant_aware is True")
        async with self._async_conn.cursor() as cursor:
            results = self._execute_query(cursor, query_embedding, tenant_id)
        await self._async_conn.commit()
        return self._process_query_results(results)
    

    def create_tenant(self, tenant_name: str) -> uuid.UUID:
        """
        Create a new tenant and return the tenant_id.
        
        Parameters:
            tenant_name (str): The name of the tenant to create.

        Returns:
            tenant_id (uuid.UUID): The id of the newly created tenant.
        """
        with self._sync_conn.cursor() as cursor:
            cursor.execute("""
                           INSERT INTO tenants (name) VALUES (%(tenant_name)s) returning id
                           """,
                           {'tenant_name': tenant_name})
            tenant_id = cursor.fetchone()[0]
            self._sync_conn.commit()
            return tenant_id
    
    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        tenant_id = delete_kwargs.get("tenant_id", None)
        _logger.info(f"Deleting document {ref_doc_id} with tenant_id {tenant_id}")
        if self.tenant_aware and tenant_id is None:
            raise ValueError("tenant_id must be specified in delete_kwargs if tenant_aware is True")
        with self._sync_conn.cursor() as cursor:
            self._set_tenant_context(cursor, tenant_id)
            cursor.execute(
                sql.SQL("DELETE FROM {} WHERE metadata->>'doc_id' = %(ref_doc_id)s").format(sql.Identifier(self.table_name)),
                {'ref_doc_id': ref_doc_id}
            )
        self._sync_conn.commit()
    
    async def adelete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        tenant_id = delete_kwargs.get("tenant_id", None)
        _logger.info(f"Deleting document {ref_doc_id} with tenant_id {tenant_id}")
        if self.tenant_aware and tenant_id is None:
            raise ValueError("tenant_id must be specified in delete_kwargs if tenant_aware is True")
        async with self._async_conn.cursor() as cursor:
            self._set_tenant_context(cursor, tenant_id)
            cursor.execute(
                sql.SQL("DELETE FROM {} WHERE metadata->>'doc_id' = %(ref_doc_id)s").format(sql.Identifier(self.table_name)),
                {'ref_doc_id': ref_doc_id}
            )
        await self._async_conn.commit()
    
    # NOTE: Implement get_nodes
    # NOTE: Implement delete_nodes
    # NOTE: Implement clear


