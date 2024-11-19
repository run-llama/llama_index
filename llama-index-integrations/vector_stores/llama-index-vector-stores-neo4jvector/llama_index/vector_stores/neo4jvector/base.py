from typing import Any, Dict, List, Optional, Tuple
import logging

import neo4j

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.schema import BaseNode, MetadataMode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
    FilterOperator,
    MetadataFilters,
    MetadataFilter,
    FilterCondition,
)
from llama_index.core.vector_stores.utils import (
    metadata_dict_to_node,
    node_to_metadata_dict,
)

_logger = logging.getLogger(__name__)


def check_if_not_null(props: List[str], values: List[Any]) -> None:
    """Check if variable is not null and raise error accordingly."""
    for prop, value in zip(props, values):
        if not value:
            raise ValueError(f"Parameter `{prop}` must not be None or empty string")


def sort_by_index_name(
    lst: List[Dict[str, Any]], index_name: str
) -> List[Dict[str, Any]]:
    """Sort first element to match the index_name if exists."""
    return sorted(lst, key=lambda x: x.get("name") != index_name)


def clean_params(params: List[BaseNode]) -> List[Dict[str, Any]]:
    """Convert BaseNode object to a dictionary to be imported into Neo4j."""
    clean_params = []
    for record in params:
        text = record.get_content(metadata_mode=MetadataMode.NONE)
        embedding = record.get_embedding()
        id = record.node_id
        metadata = node_to_metadata_dict(record, remove_text=True, flat_metadata=False)
        # Remove redundant metadata information
        for k in ["document_id", "doc_id"]:
            del metadata[k]
        clean_params.append(
            {"text": text, "embedding": embedding, "id": id, "metadata": metadata}
        )
    return clean_params


def _get_search_index_query(hybrid: bool) -> str:
    if not hybrid:
        return (
            "CALL db.index.vector.queryNodes($index, $k, $embedding) YIELD node, score "
        )
    return (
        "CALL { "
        "CALL db.index.vector.queryNodes($index, $k, $embedding) "
        "YIELD node, score "
        "WITH collect({node:node, score:score}) AS nodes, max(score) AS max "
        "UNWIND nodes AS n "
        # We use 0 as min
        "RETURN n.node AS node, (n.score / max) AS score UNION "
        "CALL db.index.fulltext.queryNodes($keyword_index, $query, {limit: $k}) "
        "YIELD node, score "
        "WITH collect({node:node, score:score}) AS nodes, max(score) AS max "
        "UNWIND nodes AS n "
        # We use 0 as min
        "RETURN n.node AS node, (n.score / max) AS score "
        "} "
        # dedup
        "WITH node, max(score) AS score ORDER BY score DESC LIMIT $k "
    )


def remove_lucene_chars(text: Optional[str]) -> Optional[str]:
    """Remove Lucene special characters."""
    if not text:
        return None
    special_chars = [
        "+",
        "-",
        "&",
        "|",
        "!",
        "(",
        ")",
        "{",
        "}",
        "[",
        "]",
        "^",
        '"',
        "~",
        "*",
        "?",
        ":",
        "\\",
    ]
    for char in special_chars:
        if char in text:
            text = text.replace(char, " ")
    return text.strip()


def _to_neo4j_operator(operator: FilterOperator) -> str:
    if operator == FilterOperator.EQ:
        return "="
    elif operator == FilterOperator.GT:
        return ">"
    elif operator == FilterOperator.LT:
        return "<"
    elif operator == FilterOperator.NE:
        return "<>"
    elif operator == FilterOperator.GTE:
        return ">="
    elif operator == FilterOperator.LTE:
        return "<="
    elif operator == FilterOperator.IN:
        return "IN"
    elif operator == FilterOperator.NIN:
        return "NOT IN"
    elif operator == FilterOperator.CONTAINS:
        return "CONTAINS"
    else:
        _logger.warning(f"Unknown operator: {operator}, fallback to '='")
        return "="


def collect_params(
    input_data: List[Tuple[str, Dict[str, str]]],
) -> Tuple[List[str], Dict[str, Any]]:
    """Transform the input data into the desired format.

    Args:
    - input_data (list of tuples): Input data to transform.
      Each tuple contains a string and a dictionary.

    Returns:
    - tuple: A tuple containing a list of strings and a dictionary.
    """
    # Initialize variables to hold the output parts
    query_parts = []
    params = {}

    # Loop through each item in the input data
    for query_part, param in input_data:
        # Append the query part to the list
        query_parts.append(query_part)
        # Update the params dictionary with the param dictionary
        params.update(param)

    # Return the transformed data
    return (query_parts, params)


def filter_to_cypher(index: int, filter: MetadataFilter) -> str:
    return (
        f"n.`{filter.key}` {_to_neo4j_operator(filter.operator)} $param_{index}",
        {f"param_{index}": filter.value},
    )


def construct_metadata_filter(filters: MetadataFilters):
    cypher_snippets = []
    for index, filter in enumerate(filters.filters):
        cypher_snippets.append(filter_to_cypher(index, filter))

    collected_snippets = collect_params(cypher_snippets)

    if filters.condition == FilterCondition.OR:
        return (" OR ".join(collected_snippets[0]), collected_snippets[1])
    else:
        return (" AND ".join(collected_snippets[0]), collected_snippets[1])


class Neo4jVectorStore(BasePydanticVectorStore):
    """Neo4j Vector Store.

    Examples:
        `pip install llama-index-vector-stores-neo4jvector`


        ```python
        from llama_index.vector_stores.neo4jvector import Neo4jVectorStore

        username = "neo4j"
        password = "pleaseletmein"
        url = "bolt://localhost:7687"
        embed_dim = 1536

        neo4j_vector = Neo4jVectorStore(username, password, url, embed_dim)
        ```
    """

    stores_text: bool = True
    flat_metadata: bool = True

    distance_strategy: str
    index_name: str
    keyword_index_name: str
    hybrid_search: bool
    node_label: str
    embedding_node_property: str
    text_node_property: str
    retrieval_query: str
    embedding_dimension: int

    _driver: neo4j.GraphDatabase.driver = PrivateAttr()
    _database: str = PrivateAttr()
    _support_metadata_filter: bool = PrivateAttr()
    _is_enterprise: bool = PrivateAttr()

    def __init__(
        self,
        username: str,
        password: str,
        url: str,
        embedding_dimension: int,
        database: str = "neo4j",
        index_name: str = "vector",
        keyword_index_name: str = "keyword",
        node_label: str = "Chunk",
        embedding_node_property: str = "embedding",
        text_node_property: str = "text",
        distance_strategy: str = "cosine",
        hybrid_search: bool = False,
        retrieval_query: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            distance_strategy=distance_strategy,
            index_name=index_name,
            keyword_index_name=keyword_index_name,
            hybrid_search=hybrid_search,
            node_label=node_label,
            embedding_node_property=embedding_node_property,
            text_node_property=text_node_property,
            retrieval_query=retrieval_query,
            embedding_dimension=embedding_dimension,
        )

        if distance_strategy not in ["cosine", "euclidean"]:
            raise ValueError("distance_strategy must be either 'euclidean' or 'cosine'")

        self._driver = neo4j.GraphDatabase.driver(url, auth=(username, password))
        self._database = database

        # Verify connection
        try:
            self._driver.verify_connectivity()
        except neo4j.exceptions.ServiceUnavailable:
            raise ValueError(
                "Could not connect to Neo4j database. "
                "Please ensure that the url is correct"
            )
        except neo4j.exceptions.AuthError:
            raise ValueError(
                "Could not connect to Neo4j database. "
                "Please ensure that the username and password are correct"
            )

        # Verify if the version support vector index
        self._verify_version()

        # Verify that required values are not null
        check_if_not_null(
            [
                "index_name",
                "node_label",
                "embedding_node_property",
                "text_node_property",
            ],
            [index_name, node_label, embedding_node_property, text_node_property],
        )

        index_already_exists = self.retrieve_existing_index()
        if not index_already_exists:
            self.create_new_index()
        if hybrid_search:
            fts_node_label = self.retrieve_existing_fts_index()
            # If the FTS index doesn't exist yet
            if not fts_node_label:
                self.create_new_keyword_index()
            else:  # Validate that FTS and Vector index use the same information
                if not fts_node_label == self.node_label:
                    raise ValueError(
                        "Vector and keyword index don't index the same node label"
                    )

    @property
    def client(self) -> neo4j.GraphDatabase.driver:
        return self._driver

    def _verify_version(self) -> None:
        """
        Check if the connected Neo4j database version supports vector indexing.

        Queries the Neo4j database to retrieve its version and compares it
        against a target version (5.11.0) that is known to support vector
        indexing. Raises a ValueError if the connected Neo4j version is
        not supported.
        """
        db_data = self.database_query("CALL dbms.components()")
        version = db_data[0]["versions"][0]
        if "aura" in version:
            version_tuple = (*tuple(map(int, version.split("-")[0].split("."))), 0)
        else:
            version_tuple = tuple(map(int, version.split(".")))

        target_version = (5, 11, 0)

        if version_tuple < target_version:
            raise ValueError(
                "Version index is only supported in Neo4j version 5.11 or greater"
            )

        # Flag for metadata filtering
        metadata_target_version = (5, 18, 0)
        if version_tuple < metadata_target_version:
            self._support_metadata_filter = False
        else:
            self._support_metadata_filter = True
        # Flag for enterprise
        self._is_enterprise = db_data[0]["edition"] == "enterprise"

    def create_new_index(self) -> None:
        """
        This method constructs a Cypher query and executes it
        to create a new vector index in Neo4j.
        """
        index_query = (
            "CALL db.index.vector.createNodeIndex("
            "$index_name,"
            "$node_label,"
            "$embedding_node_property,"
            "toInteger($embedding_dimension),"
            "$similarity_metric )"
        )

        parameters = {
            "index_name": self.index_name,
            "node_label": self.node_label,
            "embedding_node_property": self.embedding_node_property,
            "embedding_dimension": self.embedding_dimension,
            "similarity_metric": self.distance_strategy,
        }
        self.database_query(index_query, params=parameters)

    def retrieve_existing_index(self) -> bool:
        """
        Check if the vector index exists in the Neo4j database
        and returns its embedding dimension.

        This method queries the Neo4j database for existing indexes
        and attempts to retrieve the dimension of the vector index
        with the specified name. If the index exists, its dimension is returned.
        If the index doesn't exist, `None` is returned.

        Returns:
            int or None: The embedding dimension of the existing index if found.
        """
        index_information = self.database_query(
            "SHOW INDEXES YIELD name, type, labelsOrTypes, properties, options "
            "WHERE type = 'VECTOR' AND (name = $index_name "
            "OR (labelsOrTypes[0] = $node_label AND "
            "properties[0] = $embedding_node_property)) "
            "RETURN name, labelsOrTypes, properties, options ",
            params={
                "index_name": self.index_name,
                "node_label": self.node_label,
                "embedding_node_property": self.embedding_node_property,
            },
        )
        # sort by index_name
        index_information = sort_by_index_name(index_information, self.index_name)
        try:
            self.index_name = index_information[0]["name"]
            self.node_label = index_information[0]["labelsOrTypes"][0]
            self.embedding_node_property = index_information[0]["properties"][0]
            index_config = index_information[0]["options"]["indexConfig"]
            if "vector.dimensions" in index_config:
                self.embedding_dimension = index_config["vector.dimensions"]

            return True
        except IndexError:
            return False

    def retrieve_existing_fts_index(self) -> Optional[str]:
        """Check if the fulltext index exists in the Neo4j database.

        This method queries the Neo4j database for existing fts indexes
        with the specified name.

        Returns:
            (Tuple): keyword index information
        """
        index_information = self.database_query(
            "SHOW INDEXES YIELD name, type, labelsOrTypes, properties, options "
            "WHERE type = 'FULLTEXT' AND (name = $keyword_index_name "
            "OR (labelsOrTypes = [$node_label] AND "
            "properties = $text_node_property)) "
            "RETURN name, labelsOrTypes, properties, options ",
            params={
                "keyword_index_name": self.keyword_index_name,
                "node_label": self.node_label,
                "text_node_property": self.text_node_property,
            },
        )
        # sort by index_name
        index_information = sort_by_index_name(index_information, self.index_name)
        try:
            self.keyword_index_name = index_information[0]["name"]
            self.text_node_property = index_information[0]["properties"][0]
            return index_information[0]["labelsOrTypes"][0]
        except IndexError:
            return None

    def create_new_keyword_index(self, text_node_properties: List[str] = []) -> None:
        """
        This method constructs a Cypher query and executes it
        to create a new full text index in Neo4j.
        """
        node_props = text_node_properties or [self.text_node_property]
        fts_index_query = (
            f"CREATE FULLTEXT INDEX {self.keyword_index_name} "
            f"FOR (n:`{self.node_label}`) ON EACH "
            f"[{', '.join(['n.`' + el + '`' for el in node_props])}]"
        )
        self.database_query(fts_index_query)

    def database_query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        params = params or {}
        try:
            data, _, _ = self._driver.execute_query(
                query, database_=self._database, parameters_=params
            )
            return [r.data() for r in data]
        except neo4j.exceptions.Neo4jError as e:
            if not (
                (
                    (  # isCallInTransactionError
                        e.code == "Neo.DatabaseError.Statement.ExecutionFailed"
                        or e.code
                        == "Neo.DatabaseError.Transaction.TransactionStartFailed"
                    )
                    and "in an implicit transaction" in e.message
                )
                or (  # isPeriodicCommitError
                    e.code == "Neo.ClientError.Statement.SemanticError"
                    and (
                        "in an open transaction is not possible" in e.message
                        or "tried to execute in an explicit transaction" in e.message
                    )
                )
            ):
                raise
        # Fallback to allow implicit transactions
        with self._driver.session(database=self._database) as session:
            data = session.run(neo4j.Query(text=query), params)
            return [r.data() for r in data]

    def add(self, nodes: List[BaseNode], **add_kwargs: Any) -> List[str]:
        ids = [r.node_id for r in nodes]
        import_query = (
            "UNWIND $data AS row "
            "CALL { WITH row "
            f"MERGE (c:`{self.node_label}` {{id: row.id}}) "
            "WITH c, row "
            f"CALL db.create.setVectorProperty(c, "
            f"'{self.embedding_node_property}', row.embedding) "
            "YIELD node "
            f"SET c.`{self.text_node_property}` = row.text "
            "SET c += row.metadata } IN TRANSACTIONS OF 1000 ROWS"
        )

        self.database_query(
            import_query,
            params={"data": clean_params(nodes)},
        )

        return ids

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        if query.filters:
            # Verify that 5.18 or later is used
            if not self._support_metadata_filter:
                raise ValueError(
                    "Metadata filtering is only supported in "
                    "Neo4j version 5.18 or greater"
                )
            # Metadata filtering and hybrid doesn't work
            if self.hybrid_search:
                raise ValueError(
                    "Metadata filtering can't be use in combination with "
                    "a hybrid search approach"
                )
            parallel_query = (
                "CYPHER runtime = parallel parallelRuntimeSupport=all "
                if self._is_enterprise
                else ""
            )
            base_index_query = parallel_query + (
                f"MATCH (n:`{self.node_label}`) WHERE "
                f"n.`{self.embedding_node_property}` IS NOT NULL AND "
            )
            if self.embedding_dimension:
                base_index_query += (
                    f"size(n.`{self.embedding_node_property}`) = "
                    f"toInteger({self.embedding_dimension}) AND "
                )
            base_cosine_query = (
                " WITH n as node, vector.similarity.cosine("
                f"n.`{self.embedding_node_property}`, "
                "$embedding) AS score ORDER BY score DESC LIMIT toInteger($k) "
            )
            filter_snippets, filter_params = construct_metadata_filter(query.filters)
            index_query = base_index_query + filter_snippets + base_cosine_query
        else:
            index_query = _get_search_index_query(self.hybrid_search)
            filter_params = {}

        default_retrieval = (
            f"RETURN node.`{self.text_node_property}` AS text, score, "
            "node.id AS id, "
            f"node {{.*, `{self.text_node_property}`: Null, "
            f"`{self.embedding_node_property}`: Null, id: Null }} AS metadata"
        )

        retrieval_query = self.retrieval_query or default_retrieval
        read_query = index_query + retrieval_query

        parameters = {
            "index": self.index_name,
            "k": query.similarity_top_k,
            "embedding": query.query_embedding,
            "keyword_index": self.keyword_index_name,
            "query": remove_lucene_chars(query.query_str),
            **filter_params,
        }

        results = self.database_query(read_query, params=parameters)

        nodes = []
        similarities = []
        ids = []
        for record in results:
            node = metadata_dict_to_node(record["metadata"])
            node.set_content(str(record["text"]))
            nodes.append(node)
            similarities.append(record["score"])
            ids.append(record["id"])

        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        self.database_query(
            f"MATCH (n:`{self.node_label}`) WHERE n.ref_doc_id = $id DETACH DELETE n",
            params={"id": ref_doc_id},
        )
