from typing import Any, List, Dict, Optional, Tuple

from llama_index.core.graph_stores.prompts import DEFAULT_CYPHER_TEMPALTE
from llama_index.core.graph_stores.types import (
    PropertyGraphStore,
    Triplet,
    LabelledNode,
    Relation,
    EntityNode,
    ChunkNode,
)
from llama_index.core.graph_stores.utils import (
    value_sanitize,
)
from llama_index.core.prompts import PromptTemplate
from llama_index.core.vector_stores.types import VectorStoreQuery

import redis
from falkordb import FalkorDB


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


BASE_ENTITY_LABEL = "__Entity__"
EXCLUDED_LABELS = []
EXCLUDED_RELS = []
EXHAUSTIVE_SEARCH_LIMIT = 10000
# Threshold for returning all available prop values in graph schema
DISTINCT_VALUE_LIMIT = 10

node_properties_query = """
MATCH (n)
WITH keys(n) as keys, labels(n) AS labels
WITH CASE WHEN keys = [] THEN [NULL] ELSE keys END AS keys, labels
UNWIND labels AS label
WITH label, keys
WHERE NOT label IN $EXCLUDED_LABELS
UNWIND keys AS key
WITH label, collect(DISTINCT key) AS keys
RETURN {label:label, keys:keys} AS output
"""

rel_properties_query = """
MATCH ()-[r]->()
WITH keys(r) as keys, type(r) AS types
WITH CASE WHEN keys = [] THEN [NULL] ELSE keys END AS keys, types
UNWIND types AS type
WITH type, keys
WHERE NOT type IN $EXCLUDED_LABELS
UNWIND keys AS key WITH type,
collect(DISTINCT key) AS keys
RETURN {type:type, keys:keys} AS output
"""

rel_query = """
MATCH (n)-[r]->(m)
UNWIND labels(n) as src_label
UNWIND labels(m) as dst_label
UNWIND type(r) as rel_type
RETURN DISTINCT {start: src_label, type: rel_type, end: dst_label} AS output
"""


class FalkorDBPropertyGraphStore(PropertyGraphStore):
    r"""
    FalkorDB Property Graph Store.

    This class implements a FalkorDB property graph store.

    If you are using local FalkorDB instead of FalkorDB Cloud, here's a helpful
    command for launching the docker container:

    ```bash
    docker run \
        -p 3000:3000 -p 6379:6379 \
        -v $PWD/data:/data \
        falkordb/falkordb:latest
    ```

    Args:
        url (str): The URL for the FalkorDB database.
        database (Optional[str]): The name of the database to connect to. Defaults to "falkor".

    Examples:
        `pip install llama-index-graph-stores-falkordb`

        ```python
        from llama_index.core.indices.property_graph import PropertyGraphIndex
        from llama_index.graph_stores.falkordb import FalkorDBPropertyGraphStore

        # Create a FalkorDBPropertyGraphStore instance
        graph_store = FalkorDBPropertyGraphStore(
            url="falkordb://localhost:6379",
            database="falkor"
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
    text_to_cypher_template: PromptTemplate = DEFAULT_CYPHER_TEMPALTE

    def __init__(
        self,
        url: str,
        database: str = "falkor",
        refresh_schema: bool = True,
        sanitize_query_output: bool = True,
        **falkordb_kwargs: Any,
    ) -> None:
        self.sanitize_query_output = sanitize_query_output
        self._driver = FalkorDB.from_url(url).select_graph(database)
        self._database = database
        self.structured_schema = {}
        if refresh_schema:
            self.refresh_schema()

    @property
    def client(self):
        return self._driver

    def refresh_schema(self) -> None:
        """Refresh the schema."""
        node_query_results = self.structured_query(
            node_properties_query,
            param_map={"EXCLUDED_LABELS": [*EXCLUDED_LABELS, BASE_ENTITY_LABEL]},
        )
        node_properties = (
            [el["output"] for el in node_query_results] if node_query_results else []
        )

        rels_query_result = self.structured_query(
            rel_properties_query, param_map={"EXCLUDED_LABELS": EXCLUDED_RELS}
        )
        rel_properties = (
            [el["output"] for el in rels_query_result] if rels_query_result else []
        )

        rel_objs_query_result = self.structured_query(
            rel_query,
            param_map={"EXCLUDED_LABELS": [*EXCLUDED_LABELS, BASE_ENTITY_LABEL]},
        )
        relationships = (
            [el["output"] for el in rel_objs_query_result]
            if rel_objs_query_result
            else []
        )

        # Get constraints & indexes
        try:
            constraint = self.structured_query("CALL db.constraints()")
            index = self.structured_query(
                "CALL db.indexes() YIELD label, properties, entitytype " "RETURN *"
            )
        except (
            redis.exceptions.ResponseError
        ):  # Read-only user might not have access to schema information
            constraint = []
            index = []

        self.structured_schema = {
            "node_props": {el["label"]: el["keys"] for el in node_properties},
            "rel_props": {el["type"]: el["keys"] for el in rel_properties},
            "relationships": relationships,
            "metadata": {"constraint": constraint, "index": index},
        }

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
                SET c.embedding = vecf32(embedding)
                RETURN count(*)
                """,
                param_map={"data": chunk_dicts},
            )

        if entity_dicts:
            for entity_dict in entity_dicts:
                self.structured_query(
                    f"""
                    MERGE (e:`__Entity__` {{id: $data.id}})
                    SET e += $data.properties
                    SET e.name = $data.name
                    WITH e
                    SET e:{entity_dict["label"]}
                    WITH e
                    CALL {{
                        WITH e
                        WITH e
                        WHERE $data.embedding IS NOT NULL
                        SET e.embedding = vecf32($data.embedding)
                        RETURN count(*) AS count
                    }}
                    WITH e WHERE $data.properties.triplet_source_id IS NOT NULL
                    MERGE (c:Chunk {{id: $data.properties.triplet_source_id}})
                    MERGE (e)<-[:MENTIONS]-(c)
                    """,
                    param_map={"data": entity_dict},
                )

    def upsert_relations(self, relations: List[Relation]) -> None:
        """Add relations."""
        params = [r.dict() for r in relations]

        for param in params:
            self.structured_query(
                f"""
                MERGE (source {{id: $data.source_id}})
                ON CREATE SET source:Chunk
                MERGE (target {{id: $data.target_id}})
                ON CREATE SET target:Chunk
                WITH source, target
                CREATE (source)-[r:`{param["label"]}`]->(target)
                SET r += $data.properties
                RETURN count(*)
                """,
                param_map={"data": param},
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
        response = response if response else []

        nodes = []
        for record in response:
            # text indicates a chunk node
            # none on the type indicates an implicit node, likely a chunk node
            if "text" in record["properties"] or record["type"] is None:
                text = record["properties"].pop("text", "")
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
                        label=record["type"],
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
            MATCH (e)-[r{':`' + '`|`'.join(relation_names) + '`' if relation_names else ''}]->(t:__Entity__)
            RETURN e.name AS source_id, [l in labels(e) WHERE l <> '__Entity__' | l][0] AS source_type,
                   e{{.* , embedding: Null, name: Null}} AS source_properties,
                   type(r) AS type,
                   t.name AS target_id, [l in labels(t) WHERE l <> '__Entity__' | l][0] AS target_type,
                   t{{.* , embedding: Null, name: Null}} AS target_properties
            UNION ALL
            WITH e
            MATCH (e)<-[r{':`' + '`|`'.join(relation_names) + '`' if relation_names else ''}]-(t:__Entity__)
            RETURN t.name AS source_id, [l in labels(t) WHERE l <> '__Entity__' | l][0] AS source_type,
                   e{{.* , embedding: Null, name: Null}} AS source_properties,
                   type(r) AS type,
                   e.name AS target_id, [l in labels(e) WHERE l <> '__Entity__' | l][0] AS target_type,
                   t{{.* , embedding: Null, name: Null}} AS target_properties
        }}
        RETURN source_id, source_type, type, target_id, target_type, source_properties, target_properties"""
        cypher_statement += return_statement

        data = self.structured_query(cypher_statement, param_map=params)
        data = data if data else []

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
        self,
        graph_nodes: List[LabelledNode],
        depth: int = 2,
        limit: int = 30,
        ignore_rels: Optional[List[str]] = None,
    ) -> List[Triplet]:
        """Get depth-aware rel map."""
        triples = []

        ids = [node.id for node in graph_nodes]
        # Needs some optimization
        response = self.structured_query(
            f"""
            WITH $ids AS id_list
            UNWIND range(0, size(id_list) - 1) AS idx
            MATCH (e:`__Entity__`)
            WHERE e.id = id_list[idx]
            MATCH p=(e)-[r*1..{depth}]-(other)
            WHERE ALL(rel in relationships(p) WHERE type(rel) <> 'MENTIONS')
            UNWIND relationships(p) AS rel
            WITH distinct rel, idx
            WITH startNode(rel) AS source,
                type(rel) AS type,
                endNode(rel) AS endNode,
                idx
            LIMIT $limit
            RETURN source.id AS source_id, [l in labels(source) WHERE l <> '__Entity__' | l][0] AS source_type,
                source{{.* , embedding: Null, id: Null}} AS source_properties,
                type,
                endNode.id AS target_id, [l in labels(endNode) WHERE l <> '__Entity__' | l][0] AS target_type,
                endNode{{.* , embedding: Null, id: Null}} AS target_properties,
                idx
            ORDER BY idx
            LIMIT $limit
            """,
            param_map={"ids": ids, "limit": limit},
        )
        response = response if response else []

        ignore_rels = ignore_rels or []
        for record in response:
            if record["type"] in ignore_rels:
                continue

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

        result = self._driver.query(query, param_map)
        full_result = [
            {h[1]: d[i] for i, h in enumerate(result.header)} for d in result.result_set
        ]

        if self.sanitize_query_output:
            return [value_sanitize(el) for el in full_result]
        return full_result

    def vector_query(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> Tuple[List[LabelledNode], List[float]]:
        """Query the graph store with a vector store query."""
        conditions = None
        if query.filters:
            conditions = [
                f"e.{filter.key} {filter.operator.value} {filter.value}"
                for filter in query.filters.filters
            ]
        filters = (
            f" {query.filters.condition.value} ".join(conditions).replace("==", "=")
            if conditions is not None
            else "1 = 1"
        )

        data = self.structured_query(
            f"""MATCH (e:`__Entity__`)
            WHERE e.embedding IS NOT NULL AND ({filters})
            WITH e, vec.euclideanDistance(e.embedding, vecf32($embedding)) AS score
            ORDER BY score LIMIT $limit
            RETURN e.id AS name,
               [l in labels(e) WHERE l <> '__Entity__' | l][0] AS type,
               e{{.* , embedding: Null, name: Null, id: Null}} AS properties,
               score""",
            param_map={
                "embedding": query.query_embedding,
                "dimension": len(query.query_embedding),
                "limit": query.similarity_top_k,
            },
        )
        data = data if data else []

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
                "MATCH (n) WHERE n.name IN $entity_names DETACH DELETE n",
                param_map={"entity_names": entity_names},
            )

        if ids:
            self.structured_query(
                "MATCH (n) WHERE n.id IN $ids DETACH DELETE n",
                param_map={"ids": ids},
            )

        if relation_names:
            for rel in relation_names:
                self.structured_query(f"MATCH ()-[r:`{rel}`]->() DELETE r")

        if properties:
            cypher = "MATCH (e) WHERE "
            prop_list = []
            params = {}
            for i, prop in enumerate(properties):
                prop_list.append(f"e.`{prop}` = $property_{i}")
                params[f"property_{i}"] = properties[prop]
            cypher += " AND ".join(prop_list)
            self.structured_query(cypher + " DETACH DELETE e", param_map=params)

    def get_schema(self, refresh: bool = False) -> Any:
        if refresh:
            self.refresh_schema()

        return self.structured_schema

    def get_schema_str(self, refresh: bool = False) -> str:
        schema = self.get_schema(refresh=refresh)

        formatted_node_props = []
        formatted_rel_props = []

        # Format node properties
        for label, props in schema["node_props"].items():
            props_str = ", ".join(
                [f"{prop['property']}: {prop['type']}" for prop in props]
            )
            formatted_node_props.append(f"{label} {{{props_str}}}")

        # Format relationship properties using structured_schema
        for type, props in schema["rel_props"].items():
            props_str = ", ".join(
                [f"{prop['property']}: {prop['type']}" for prop in props]
            )
            formatted_rel_props.append(f"{type} {{{props_str}}}")

        # Format relationships
        formatted_rels = [
            f"(:{el['start']})-[:{el['type']}]->(:{el['end']})"
            for el in schema["relationships"]
        ]

        return "\n".join(
            [
                "Node properties:",
                "\n".join(formatted_node_props),
                "Relationship properties:",
                "\n".join(formatted_rel_props),
                "The relationships:",
                "\n".join(formatted_rels),
            ]
        )


FalkorDBPGStore = FalkorDBPropertyGraphStore
