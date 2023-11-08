from typing import Any, Dict, List, Optional

from llama_index.schema import BaseNode, MetadataMode
from llama_index.vector_stores.types import (
    VectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.vector_stores.utils import metadata_dict_to_node, node_to_metadata_dict


def check_if_not_null(props: List[str], values: List[Any]) -> None:
    """Check if variable is not null and raise error accordingly."""
    for prop, value in zip(props, values):
        if not value:
            raise ValueError(f"Parameter `{prop}` must not be None or empty string")


def sort_by_index_name(
    lst: List[Dict[str, Any]], index_name: str
) -> List[Dict[str, Any]]:
    """Sort first element to match the index_name if exists."""
    return sorted(lst, key=lambda x: x.get("index_name") != index_name)


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


class Neo4jVectorStore(VectorStore):
    stores_text: bool = True
    flat_metadata = True

    def __init__(
        self,
        username: str,
        password: str,
        url: str,
        embedding_dimension: int,
        database: str = "neo4j",
        index_name: str = "vector",
        node_label: str = "Chunk",
        embedding_node_property: str = "embedding",
        text_node_property: str = "text",
        distance_strategy: str = "cosine",
        retrieval_query: str = "",
        **kwargs: Any,
    ) -> None:
        try:
            import neo4j
        except ImportError:
            raise ImportError(
                "Could not import neo4j python package. "
                "Please install it with `pip install neo4j`."
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

        self.distance_strategy = distance_strategy
        self.index_name = index_name
        self.node_label = node_label
        self.embedding_node_property = embedding_node_property
        self.text_node_property = text_node_property
        self.retrieval_query = retrieval_query
        self.embedding_dimension = embedding_dimension

        index_already_exists = self.retrieve_existing_index()
        if not index_already_exists:
            self.create_new_index()

    def _verify_version(self) -> None:
        """
        Check if the connected Neo4j database version supports vector indexing.

        Queries the Neo4j database to retrieve its version and compares it
        against a target version (5.11.0) that is known to support vector
        indexing. Raises a ValueError if the connected Neo4j version is
        not supported.
        """
        version = self.database_query("CALL dbms.components()")[0]["versions"][0]
        if "aura" in version:
            version_tuple = (*tuple(map(int, version.split("-")[0].split("."))), 0)
        else:
            version_tuple = tuple(map(int, version.split(".")))

        target_version = (5, 11, 0)

        if version_tuple < target_version:
            raise ValueError(
                "Version index is only supported in Neo4j version 5.11 or greater"
            )

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
            self.embedding_dimension = index_information[0]["options"]["indexConfig"][
                "vector.dimensions"
            ]

            return True
        except IndexError:
            return False

    def database_query(
        self, query: str, params: Optional[dict] = None
    ) -> List[Dict[str, Any]]:
        """
        This method sends a Cypher query to the connected Neo4j database
        and returns the results as a list of dictionaries.

        Args:
            query (str): The Cypher query to execute.
            params (dict, optional): Dictionary of query parameters. Defaults to {}.

        Returns:
            List[Dict[str, Any]]: List of dictionaries containing the query results.
        """
        from neo4j.exceptions import CypherSyntaxError

        params = params or {}
        with self._driver.session(database=self._database) as session:
            try:
                data = session.run(query, params)
                return [r.data() for r in data]
            except CypherSyntaxError as e:
                raise ValueError(f"Cypher Statement is not valid\n{e}")

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
        default_retrieval = (
            f"RETURN node.`{self.text_node_property}` AS text, score, "
            "node.id AS id, "
            f"node {{.*, `{self.text_node_property}`: Null, "
            f"`{self.embedding_node_property}`: Null, id: Null }} AS metadata"
        )

        retrieval_query = self.retrieval_query or default_retrieval

        read_query = (
            "CALL db.index.vector.queryNodes($index, $k, $embedding) "
            "YIELD node, score "
        ) + retrieval_query

        parameters = {
            "index": self.index_name,
            "k": query.similarity_top_k,
            "embedding": query.query_embedding,
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
