from typing import Any, List, Dict, Optional, Tuple
import neo4j
from llama_index.core.vector_stores.types import VectorStoreQuery


from llama_index.core.graph_stores.types import (
    PropertyGraphStore,
    Triplet,
    LabelledNode,
    Relation,
    EntityNode,
    ChunkNode,
)


def remove_empty_values(input_dict):
    """
    Remove entries with empty values from the dictionary.

    Parameters:
    input_dict (dict): The dictionary from which empty values need to be removed.

    Returns:
    dict: A new dictionary with all empty values removed.
    """
    # Create a new dictionary excluding empty values
    return {key: value for key, value in input_dict.items() if value}


class Neo4jPGStore(PropertyGraphStore):
    """
    Neo4j Property Graph Store.

    This class implements a Neo4j property graph store.

    Args:
        username (str): The username for the Neo4j database.
        password (str): The password for the Neo4j database.
        url (str): The URL for the Neo4j database.
        database (Optional[str]): The name of the database to connect to. Defaults to "neo4j".

    Examples:
        `pip install llama-index-graph-stores-neo4j`

        ```python
        from llama_index.core.indices.property_graph import PropertyGraphIndex
        from llama_index.graph_stores.neo4j import Neo4jLPGStore

        # Create a Neo4jLPGStore instance
        graph_store = Neo4jLPGStore(
            username="neo4j",
            password="neo4j",
            url="bolt://localhost:7687",
            database="neo4j"
        )

        # create the index
        index = PropertyGraphIndex.from_documents(
            documents,
            property_graph_store=graph_store,
        )
        ```
    """

    supports_structured_queries: bool = True
    supports_vector_queries: bool = True

    def __init__(
        self,
        username: str,
        password: str,
        url: str,
        database: Optional[str] = "neo4j",
        **neo4j_kwargs: Any,
    ) -> None:
        self._driver = neo4j.GraphDatabase.driver(
            url, auth=(username, password), **neo4j_kwargs
        )
        self._async_driver = neo4j.AsyncGraphDatabase.driver(
            url,
            auth=(username, password),
            **neo4j_kwargs,
        )
        self._database = database

    @property
    def client(self):
        return self._driver

    def upsert_nodes(self, nodes: List[LabelledNode]) -> None:
        # Lists to hold separated types
        entity_dicts: List[dict] = []
        chunk_dicts: List[dict] = []

        # Sort by type
        for item in nodes:
            if isinstance(item, EntityNode):
                entity_dicts.append({**item.dict(), "id": item.id})
            elif isinstance(item, ChunkNode):
                chunk_dicts.append({**item.dict(), "id": item.id})
            else:
                # Log that we do not support these types of nodes
                # Or raise an error?
                pass

        if chunk_dicts:
            self.structured_query(
                """
                UNWIND $data AS row
                MERGE (c:Chunk {id: row.id})
                SET c.text = row.text
                WITH c, row
                SET c += row.properties
                WITH c, row.embedding AS embedding
                WHERE embedding IS NOT NULL
                CALL db.create.setNodeVectorProperty(c, 'embedding', embedding)
                RETURN count(*)
                """,
                param_map={"data": chunk_dicts},
            )

        if entity_dicts:
            self.structured_query(
                """
                UNWIND $data AS row
                MERGE (e:`__Entity__` {id: row.id})
                SET e += apoc.map.clean(row.properties, [], [])
                SET e.name = row.name
                WITH e, row
                CALL apoc.create.addLabels(e, [row.label])
                YIELD node
                WITH e, row
                CALL {
                    WITH e, row
                    WITH e, row
                    WHERE row.embedding IS NOT NULL
                    CALL db.create.setNodeVectorProperty(e, 'embedding', row.embedding)
                    RETURN count(*) AS count
                }
                WITH e, row WHERE row.properties.triplet_source_id IS NOT NULL
                MERGE (c:Chunk {id: row.properties.triplet_source_id})
                MERGE (e)<-[:MENTIONS]-(c)
                """,
                param_map={"data": entity_dicts},
            )

    def upsert_relations(self, relations: List[Relation]) -> None:
        """Add relations."""
        params = [r.dict() for r in relations]

        self.structured_query(
            """
            UNWIND $data AS row
            MERGE (source:`__Entity__` {id: row.source_id})
            MERGE (target:`__Entity__` {id: row.target_id})
            WITH source, target, row
            CALL apoc.merge.relationship(source, row.label, {}, row.properties, target) YIELD rel
            RETURN count(*)
            """,
            param_map={"data": params},
        )

    def get(
        self,
        properties: Optional[dict] = None,
        ids: Optional[List[str]] = None,
    ) -> List[LabelledNode]:
        """Get nodes."""
        cypher_statement = "MATCH (e) "

        params = {}
        if properties or ids:
            cypher_statement += "WHERE "

        if ids:
            cypher_statement += "e.id in $ids "
            params["ids"] = ids

        if properties:
            prop_list = []
            for i, prop in enumerate(properties):
                prop_list.append(f"e.`{prop}` = $property_{i}")
                params[f"property_{i}"] = properties[prop]
            cypher_statement += " AND ".join(prop_list)

        return_statement = """
        WITH e
        RETURN e.id AS name,
               [l in labels(e) WHERE l <> '__Entity__' | l][0] AS type,
               e{.* , embedding: Null, id: Null} AS properties
        """
        cypher_statement += return_statement

        response = self.structured_query(cypher_statement, param_map=params)

        nodes = []
        for record in response:
            if "text" in record["properties"]:
                text = record["properties"].pop("text")
                nodes.append(
                    ChunkNode(
                        id_=record["name"],
                        text=text,
                        properties=remove_empty_values(record["properties"]),
                    )
                )
            else:
                nodes.append(
                    EntityNode(
                        name=record["name"],
                        type=record["type"],
                        properties=remove_empty_values(record["properties"]),
                    )
                )

        return nodes

    def get_triplets(
        self,
        entity_names: Optional[List[str]] = None,
        relation_names: Optional[List[str]] = None,
        properties: Optional[dict] = None,
        ids: Optional[List[str]] = None,
    ) -> List[Triplet]:
        # TODO: handle ids of chunk nodes
        cypher_statement = "MATCH (e:`__Entity__`) "

        params = {}
        if entity_names or properties or ids:
            cypher_statement += "WHERE "

        if entity_names:
            cypher_statement += "e.name in $entity_names "
            params["entity_names"] = entity_names

        if ids:
            cypher_statement += "e.id in $ids "
            params["ids"] = ids

        if properties:
            prop_list = []
            for i, prop in enumerate(properties):
                prop_list.append(f"e.`{prop}` = $property_{i}")
                params[f"property_{i}"] = properties[prop]
            cypher_statement += " AND ".join(prop_list)

        return_statement = f"""
        WITH e
        CALL {{
            WITH e
            MATCH (e)-[r{':`' + '`|`'.join(relation_names) + '`' if relation_names else ''}]->(t)
            RETURN e.name AS source_id, [l in labels(e) WHERE l <> '__Entity__' | l][0] AS source_type,
                   e{{.* , embedding: Null, name: Null}} AS source_properties,
                   type(r) AS type,
                   t.name AS target_id, [l in labels(t) WHERE l <> '__Entity__' | l][0] AS target_type,
                   t{{.* , embedding: Null, name: Null}} AS target_properties
            UNION ALL
            WITH e
            MATCH (e)<-[r{':`' + '`|`'.join(relation_names) + '`' if relation_names else ''}]-(t)
            RETURN t.name AS source_id, [l in labels(t) WHERE l <> '__Entity__' | l][0] AS source_type,
                   e{{.* , embedding: Null, name: Null}} AS source_properties,
                   type(r) AS type,
                   e.name AS target_id, [l in labels(e) WHERE l <> '__Entity__' | l][0] AS target_type,
                   t{{.* , embedding: Null, name: Null}} AS target_properties
        }}
        RETURN source_id, source_type, type, target_id, target_type, source_properties, target_properties"""
        cypher_statement += return_statement

        data = self.structured_query(cypher_statement, param_map=params)

        triples = []
        for record in data:
            source = EntityNode(
                name=record["source_id"],
                label=record["source_type"],
                properties=remove_empty_values(record["source_properties"]),
            )
            target = EntityNode(
                name=record["target_id"],
                label=record["target_type"],
                properties=remove_empty_values(record["target_properties"]),
            )
            rel = Relation(
                source_id=record["source_id"],
                target_id=record["target_id"],
                label=record["type"],
            )
            triples.append([source, rel, target])
        return triples

    def get_rel_map(
        self, graph_nodes: List[LabelledNode], depth: int = 2, limit: int = 30
    ) -> List[Triplet]:
        """Get depth-aware rel map."""
        triples = []

        ids = [node.id for node in graph_nodes]
        # Needs some optimization / atm only outgoing rels
        response = self.structured_query(
            f"""
            MATCH (e:`__Entity__`)
            WHERE e.id in $ids
            MATCH p=(e)-[:!MENTIONS]->{{1,{depth}}}()
            UNWIND relationships(p) AS rel
            WITH distinct rel
            WITH startNode(rel) AS source,
                type(rel) AS type,
                endNode(rel) AS endNode
            RETURN source.id AS source_id, [l in labels(source) WHERE l <> '__Entity__' | l][0] AS source_type,
                    source{{.* , embedding: Null, id: Null}} AS source_properties,
                    type,
                    endNode.id AS target_id, [l in labels(endNode) WHERE l <> '__Entity__' | l][0] AS target_type,
                    endNode{{.* , embedding: Null, id: Null}} AS target_properties
            LIMIT toInteger($limit)
            """,
            param_map={"ids": ids, "limit": limit},
        )

        for record in response:
            source = EntityNode(
                name=record["source_id"],
                label=record["source_type"],
                properties=remove_empty_values(record["source_properties"]),
            )
            target = EntityNode(
                name=record["target_id"],
                label=record["target_type"],
                properties=remove_empty_values(record["target_properties"]),
            )
            rel = Relation(
                source_id=record["source_id"],
                target_id=record["target_id"],
                label=record["type"],
            )
            triples.append([source, rel, target])

        return triples

    def structured_query(
        self, query: str, param_map: Optional[Dict[str, Any]] = None
    ) -> Any:
        param_map = param_map or {}

        with self._driver.session(database=self._database) as session:
            result = session.run(query, param_map)
            return [d.data() for d in result]

    def vector_query(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> Tuple[List[LabelledNode], List[float]]:
        """Query the graph store with a vector store query."""
        data = self.structured_query(
            """MATCH (e:`__Entity__`)
            WHERE e.embedding IS NOT NULL AND size(e.embedding) = $dimension
            WITH e, vector.similarity.cosine(e.embedding, $embedding) AS score
            ORDER BY score DESC LIMIT toInteger($limit)
            RETURN e.id AS name,
               [l in labels(e) WHERE l <> '__Entity__' | l][0] AS type,
               e{.* , embedding: Null, name: Null, id: Null} AS properties,
               score""",
            param_map={
                "embedding": query.query_embedding,
                "dimension": len(query.query_embedding),
                "limit": query.similarity_top_k,
            },
        )

        nodes = []
        scores = []
        for record in data:
            node = EntityNode(
                name=record["name"],
                label=record["type"],
                properties=remove_empty_values(record["properties"]),
            )
            nodes.append(node)
            scores.append(record["score"])

        return (nodes, scores)

    def delete(
        self,
        entity_names: Optional[List[str]] = None,
        relation_names: Optional[List[str]] = None,
        properties: Optional[dict] = None,
        ids: Optional[List[str]] = None,
    ) -> None:
        """Delete matching data."""
        if entity_names:
            self.structured_query(
                "MATCH (n:`__Entity__`) WHERE n.name IN $entity_names DETACH DELETE n",
                param_map={"entity_names": entity_names},
            )

        if ids:
            self.structured_query(
                "MATCH (n:`__Entity__`) WHERE n.id IN $ids DETACH DELETE n",
                param_map={"ids": ids},
            )

        if relation_names:
            for rel in relation_names:
                self.structured_query(f"MATCH ()-[r:`{rel}`]->() DELETE r")

        if properties:
            cypher = "MATCH (e:`__Entity__`) WHERE "
            prop_list = []
            params = {}
            for i, prop in enumerate(properties):
                prop_list.append(f"e.`{prop}` = $property_{i}")
                params[f"property_{i}"] = properties[prop]
            cypher += " AND ".join(prop_list)
            self.structured_query(cypher + " DETACH DELETE e", param_map=params)

    def get_schema(self) -> Any:
        pass
