import logging
from collections import defaultdict
from types import TracebackType
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Type

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
from falkordb import Edge, FalkorDB, Node, Path

logger = logging.getLogger(__name__)


def remove_empty_values(input_dict):
    """
    Remove entries with empty values from the dictionary.

    Parameters
    ----------
    input_dict (dict): The dictionary from which empty values need to be removed.

    Returns
    -------
    dict: A new dictionary with all empty values removed.

    """
    # Create a new dictionary excluding empty values
    return {key: value for key, value in input_dict.items() if value}


BASE_ENTITY_LABEL = "__Entity__"
CHUNK_LABEL = "Chunk"
# Every node written by this store carries one of these labels. FalkorDB indexes
# are label-scoped, so lookups have to name a label to be able to use them.
STORE_NODE_LABELS = (BASE_ENTITY_LABEL, CHUNK_LABEL)
EMBEDDING_KEY = "embedding"
EXCLUDED_LABELS = []
EXCLUDED_RELS = []
EXHAUSTIVE_SEARCH_LIMIT = 10000
# Threshold for returning all available prop values in graph schema
DISTINCT_VALUE_LIMIT = 10
# Number of rows sent to the server in a single write statement
CHUNK_SIZE = 1000
# Number of entities/relations inspected when deriving the graph schema
SCHEMA_SAMPLE_SIZE = 1000
VECTOR_SIMILARITY_FUNCTION = "cosine"


def escape_identifier(identifier: str) -> str:
    """
    Quote a label, relationship type or property key for safe interpolation.

    FalkorDB does not support escaping a backtick inside a quoted identifier
    (doubling it is a syntax error), so backticks are stripped rather than
    escaped. Quoting is required because labels and relationship types are
    frequently produced by an LLM and may contain spaces or reserved words.
    """
    return f"`{str(identifier).replace('`', '')}`"


def convert_operator(operator: str) -> str:
    """Map a llama-index filter operator onto its Cypher equivalent."""
    mapping = {
        "==": "=",
        "!=": "<>",
        "nin": "IN",
        "in": "IN",
        "contains": "CONTAINS",
        "text_match": "CONTAINS",
    }
    return mapping.get(operator, operator)


def to_plain_value(value: Any) -> Any:
    """
    Convert FalkorDB result objects into plain Python containers.

    The FalkorDB client returns ``Node``/``Edge``/``Path`` instances, which are
    opaque to ``value_sanitize`` (so embeddings would never be stripped) and
    render as ``<falkordb.node.Node object at 0x...>`` when a retriever feeds
    the result of a text-to-Cypher query back to the LLM.
    """
    if isinstance(value, Node):
        return dict(value.properties)
    if isinstance(value, Edge):
        return dict(value.properties)
    if isinstance(value, Path):
        return {
            "nodes": [to_plain_value(node) for node in value.nodes()],
            "relationships": [to_plain_value(edge) for edge in value.edges()],
        }
    if isinstance(value, dict):
        return {key: to_plain_value(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain_value(val) for val in value]
    return value


def sample_value_type(value: Any) -> str:
    """Derive a Cypher-ish type name from a sampled property value."""
    if isinstance(value, bool):
        return "BOOLEAN"
    if isinstance(value, int):
        return "INTEGER"
    if isinstance(value, float):
        return "FLOAT"
    if isinstance(value, str):
        return "STRING"
    if isinstance(value, (list, tuple)):
        return "LIST"
    if isinstance(value, dict):
        return "MAP"
    return "UNKNOWN"


def _batched(rows: List[dict], size: Optional[int] = None) -> Iterator[List[dict]]:
    """Yield ``rows`` in slices of at most ``size`` elements."""
    size = size or CHUNK_SIZE
    for index in range(0, len(rows), size):
        yield rows[index : index + size]


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
        database (Optional[str]): The name of the graph to connect to. Defaults to "falkor".
        refresh_schema (bool): Whether to read the graph schema on startup. Defaults to True.
        sanitize_query_output (bool): Whether to strip oversized values (such as
            embeddings) from query results. Defaults to True.
        create_indexes (bool): Whether to create the indexes used for fast lookups
            and vector search. Defaults to True.
        timeout (Optional[int]): Query timeout in milliseconds. Defaults to None.
        **falkordb_kwargs (Any): Additional keyword arguments forwarded to the
            FalkorDB client (e.g. ``username``, ``password``, ``ssl``).

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
        create_indexes: bool = True,
        timeout: Optional[int] = None,
        **falkordb_kwargs: Any,
    ) -> None:
        self.sanitize_query_output = sanitize_query_output
        self._create_indexes = create_indexes
        self._timeout = timeout
        self._driver = FalkorDB.from_url(url, **falkordb_kwargs)
        self._graph = self._driver.select_graph(database)
        self._database = database
        self.structured_schema = {}
        # label -> dimension of the vector index defined on `embedding`
        self._vector_index_dimensions: Dict[str, int] = {}

        if self._create_indexes:
            self._create_range_indexes()
        self._refresh_vector_index_info()

        if refresh_schema:
            self.refresh_schema()

    @property
    def client(self):
        return self._graph

    ### ----- Index management ----- ###

    def _run_index_statement(self, statement: str) -> bool:
        """Run an index/constraint statement, tolerating "already exists" errors."""
        try:
            self.structured_query(statement)
        except redis.exceptions.ResponseError as e:
            message = str(e).lower()
            if "already indexed" in message or "already exists" in message:
                return True
            logger.warning("Could not create index with `%s`: %s", statement, e)
            return False
        return True

    def _create_range_indexes(self) -> None:
        """Create the range indexes backing every `id` lookup and MERGE."""
        for label in (BASE_ENTITY_LABEL, CHUNK_LABEL):
            self._run_index_statement(
                f"CREATE INDEX FOR (n:{escape_identifier(label)}) ON (n.id)"
            )

    def _refresh_vector_index_info(self) -> None:
        """Record the dimension of every existing vector index on `embedding`."""
        self._vector_index_dimensions = {}
        try:
            indexes = self.structured_query("CALL db.indexes()")
        except redis.exceptions.ResponseError as e:
            logger.debug("Could not list indexes: %s", e)
            return

        for index in indexes or []:
            types = index.get("types") or {}
            if "VECTOR" not in (types.get(EMBEDDING_KEY) or []):
                continue
            options = (index.get("options") or {}).get(EMBEDDING_KEY) or {}
            dimension = options.get("dimension")
            if index.get("label") is not None and dimension is not None:
                self._vector_index_dimensions[index["label"]] = int(dimension)

    def _ensure_vector_index(self, label: str, dimension: int) -> None:
        """
        Create the vector index for ``label`` if it does not exist yet.

        FalkorDB requires the embedding dimension up front, so the index can
        only be created once the first embedding is known.
        """
        if not self._create_indexes:
            return

        existing = self._vector_index_dimensions.get(label)
        if existing == dimension:
            return
        if existing is not None:
            logger.warning(
                "A vector index on `%s`.embedding already exists with dimension %s, "
                "but an embedding of dimension %s was provided. Vector search will "
                "fall back to a full scan.",
                label,
                existing,
                dimension,
            )
            return

        created = self._run_index_statement(
            f"CREATE VECTOR INDEX FOR (n:{escape_identifier(label)}) ON (n.{EMBEDDING_KEY}) "
            f"OPTIONS {{dimension: {int(dimension)}, "
            f"similarityFunction: '{VECTOR_SIMILARITY_FUNCTION}'}}"
        )
        if created:
            self._refresh_vector_index_info()

    ### ----- Schema ----- ###

    def _sample_node_properties(self) -> List[dict]:
        """
        Derive the property types of every node label in the graph.

        Labels are enumerated exhaustively and only the property *values* are
        sampled, so a label that happens to sit outside the first
        ``SCHEMA_SAMPLE_SIZE`` scanned nodes still appears in the schema.
        """
        excluded = {*EXCLUDED_LABELS, BASE_ENTITY_LABEL}
        labels = [
            row["label"]
            for row in self.structured_query("CALL db.labels()") or []
            if row["label"] not in excluded
        ]

        outputs = []
        for label in labels:
            rows = self.structured_query(
                f"""
                MATCH (n:{escape_identifier(label)})
                WITH n LIMIT $SAMPLE_SIZE
                UNWIND [k IN keys(n) WHERE k <> '{EMBEDDING_KEY}'] AS key
                WITH key, collect(n[key])[0] AS sample
                RETURN collect({{property: key, sample: sample}}) AS keys
                """,
                param_map={"SAMPLE_SIZE": SCHEMA_SAMPLE_SIZE},
            )
            outputs.append({"label": label, "keys": rows[0]["keys"] if rows else []})
        return outputs

    def _sample_rel_properties(self) -> List[dict]:
        """Derive the property types of every relationship type in the graph."""
        rel_types = [
            row["relationshipType"]
            for row in self.structured_query("CALL db.relationshipTypes()") or []
            if row["relationshipType"] not in EXCLUDED_RELS
        ]

        outputs = []
        for rel_type in rel_types:
            rows = self.structured_query(
                f"""
                MATCH ()-[r:{escape_identifier(rel_type)}]->()
                WITH r LIMIT $SAMPLE_SIZE
                UNWIND keys(r) AS key
                WITH key, collect(r[key])[0] AS sample
                RETURN collect({{property: key, sample: sample}}) AS keys
                """,
                param_map={"SAMPLE_SIZE": SCHEMA_SAMPLE_SIZE},
            )
            outputs.append({"type": rel_type, "keys": rows[0]["keys"] if rows else []})
        return outputs

    def _sample_rel_triples(self) -> List[dict]:
        """
        Derive the (start label, type, end label) triples present in the graph.

        Relationship types are enumerated exhaustively and their endpoint labels
        are sampled per type. The obvious `MATCH (n)-[r]->(m)` instead traverses
        every relationship in the graph, and `PropertyGraphIndex` refreshes the
        schema after *every* ingestion batch, so that cost is paid over and over
        and grows with the graph.
        """
        triples: List[dict] = []
        seen: Set[Tuple[str, str, str]] = set()

        for row in self.structured_query("CALL db.relationshipTypes()") or []:
            rel_type = row["relationshipType"]
            if rel_type in EXCLUDED_RELS:
                continue

            rows = self.structured_query(
                f"""
                MATCH (n)-[r:{escape_identifier(rel_type)}]->(m)
                WITH n, m LIMIT $SAMPLE_SIZE
                UNWIND labels(n) AS start_label
                UNWIND labels(m) AS end_label
                RETURN DISTINCT start_label, end_label
                """,
                param_map={"SAMPLE_SIZE": SCHEMA_SAMPLE_SIZE},
            )
            for record in rows or []:
                key = (record["start_label"], rel_type, record["end_label"])
                if key in seen:
                    continue
                seen.add(key)
                triples.append(
                    {
                        "start": record["start_label"],
                        "type": rel_type,
                        "end": record["end_label"],
                    }
                )
        return triples

    def refresh_schema(self) -> None:
        """Refresh the schema."""
        node_properties = self._sample_node_properties()
        rel_properties = self._sample_rel_properties()
        relationships = self._sample_rel_triples()

        # Get constraints & indexes
        try:
            constraint = self.structured_query("CALL db.constraints()")
            index = self.structured_query(
                "CALL db.indexes() YIELD label, properties, entitytype RETURN *"
            )
        except (
            redis.exceptions.ResponseError
        ):  # Read-only user might not have access to schema information
            constraint = []
            index = []

        self.structured_schema = {
            "node_props": {
                el["label"]: self._format_properties(el["keys"])
                for el in node_properties
            },
            "rel_props": {
                el["type"]: self._format_properties(el["keys"]) for el in rel_properties
            },
            "relationships": relationships,
            "metadata": {"constraint": constraint, "index": index},
        }

    @staticmethod
    def _format_properties(keys: Optional[List[Any]]) -> List[Dict[str, str]]:
        """Turn sampled property values into ``{property, type}`` entries."""
        formatted = []
        for key in keys or []:
            if isinstance(key, dict):
                formatted.append(
                    {
                        "property": key.get("property"),
                        "type": sample_value_type(key.get("sample")),
                    }
                )
            else:
                # Defensive: older schemas stored bare property names
                formatted.append({"property": key, "type": "UNKNOWN"})
        return formatted

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

    ### ----- Writes ----- ###

    def upsert_nodes(self, nodes: Sequence[LabelledNode]) -> None:
        # Lists to hold separated types
        entity_dicts: List[dict] = []
        chunk_dicts: List[dict] = []

        # Sort by type
        for item in nodes:
            if isinstance(item, EntityNode):
                entity_dicts.append({**item.model_dump(), "id": item.id})
            elif isinstance(item, ChunkNode):
                chunk_dicts.append({**item.model_dump(), "id": item.id})
            else:
                logger.warning(
                    "Unsupported node type `%s`, skipping.", type(item).__name__
                )

        if chunk_dicts:
            for batch in _batched(chunk_dicts):
                self.structured_query(
                    f"""
                    UNWIND $data AS row
                    MERGE (c:{escape_identifier(CHUNK_LABEL)} {{id: row.id}})
                    SET c.text = row.text
                    SET c += row.properties
                    SET c.{EMBEDDING_KEY} = CASE WHEN row.embedding IS NULL
                        THEN c.{EMBEDDING_KEY} ELSE vecf32(row.embedding) END
                    RETURN count(*)
                    """,
                    param_map={"data": batch},
                )

        if not entity_dicts:
            return

        for dimension in {
            len(entity["embedding"])
            for entity in entity_dicts
            if entity.get("embedding")
        }:
            self._ensure_vector_index(BASE_ENTITY_LABEL, dimension)

        # FalkorDB has no APOC, so labels cannot be parameterized. Group the
        # entities by label to keep a single statement per label instead of one
        # statement per node.
        entities_by_label: Dict[str, List[dict]] = defaultdict(list)
        for entity in entity_dicts:
            entities_by_label[entity["label"]].append(entity)

        for label, rows in entities_by_label.items():
            for batch in _batched(rows):
                self.structured_query(
                    f"""
                    UNWIND $data AS row
                    MERGE (e:{escape_identifier(BASE_ENTITY_LABEL)} {{id: row.id}})
                    SET e += row.properties
                    SET e.name = row.name
                    SET e:{escape_identifier(label)}
                    SET e.{EMBEDDING_KEY} = CASE WHEN row.embedding IS NULL
                        THEN e.{EMBEDDING_KEY} ELSE vecf32(row.embedding) END
                    RETURN count(*)
                    """,
                    param_map={"data": batch},
                )

        # Create MENTIONS relationships for entities with a triplet_source_id
        mentions = [
            {
                "entity_id": entity["id"],
                "chunk_id": entity["properties"]["triplet_source_id"],
            }
            for entity in entity_dicts
            if (entity.get("properties") or {}).get("triplet_source_id")
        ]
        for batch in _batched(mentions):
            self.structured_query(
                f"""
                UNWIND $data AS row
                MATCH (e:{escape_identifier(BASE_ENTITY_LABEL)} {{id: row.entity_id}})
                MERGE (c:{escape_identifier(CHUNK_LABEL)} {{id: row.chunk_id}})
                MERGE (e)<-[:MENTIONS]-(c)
                RETURN count(*)
                """,
                param_map={"data": batch},
            )

    def _existing_chunk_ids(self, ids: List[str]) -> Set[str]:
        """Return the subset of ``ids`` that already exist as chunk nodes."""
        found: Set[str] = set()
        for batch in _batched([{"id": node_id} for node_id in ids]):
            rows = self.structured_query(
                f"""
                UNWIND $data AS row
                MATCH (c:{escape_identifier(CHUNK_LABEL)} {{id: row.id}})
                RETURN c.id AS id
                """,
                param_map={"data": batch},
            )
            found.update(row["id"] for row in rows or [])
        return found

    def upsert_relations(self, relations: List[Relation]) -> None:
        """Add relations."""
        if not relations:
            return

        # An endpoint that does not exist yet is created with BASE_ENTITY_LABEL
        # so that a later `upsert_nodes` MERGE matches this very node instead of
        # inserting a second one with the same id. Endpoints that already exist
        # as chunks keep their label for the same reason.
        endpoint_ids = sorted(
            {relation.source_id for relation in relations}
            | {relation.target_id for relation in relations}
        )
        chunk_ids = self._existing_chunk_ids(endpoint_ids)

        def label_for(node_id: str) -> str:
            return CHUNK_LABEL if node_id in chunk_ids else BASE_ENTITY_LABEL

        # Neither relationship types nor labels can be parameterized, so group
        # the rows by every identifier the statement has to interpolate. Merging
        # on a labelled pattern is also what lets FalkorDB use the `id` indexes
        # instead of scanning every node in the graph for each row.
        grouped: Dict[Tuple[str, str, str], List[dict]] = defaultdict(list)
        for relation in relations:
            key = (
                relation.label,
                label_for(relation.source_id),
                label_for(relation.target_id),
            )
            grouped[key].append(relation.model_dump())

        for (label, source_label, target_label), rows in grouped.items():
            for batch in _batched(rows):
                self.structured_query(
                    f"""
                    UNWIND $data AS row
                    MERGE (source:{escape_identifier(source_label)} {{id: row.source_id}})
                    MERGE (target:{escape_identifier(target_label)} {{id: row.target_id}})
                    MERGE (source)-[r:{escape_identifier(label)}]->(target)
                    SET r += row.properties
                    RETURN count(*)
                    """,
                    param_map={"data": batch},
                )

    ### ----- Reads ----- ###

    def _match_store_nodes(
        self, conditions: List[str], params: Dict[str, Any]
    ) -> List[dict]:
        """
        Run a node lookup once per store-managed label and merge the results.

        Every node this store writes carries either ``BASE_ENTITY_LABEL`` or
        ``CHUNK_LABEL``. Scoping the match to those labels is what lets FalkorDB
        use the `id` range indexes: its indexes are label-scoped, so an
        unlabelled `MATCH (e)` degrades to an `All Node Scan` of the whole
        graph. That matters because `PropertyGraphIndex` calls `get()` twice per
        ingestion batch to deduplicate, making ingestion quadratic.
        """
        where = f"WHERE {' AND '.join(conditions)} " if conditions else ""
        return_statement = f"""
        WITH e
        RETURN e.id AS name,
               [l in labels(e) WHERE l <> '{BASE_ENTITY_LABEL}' | l][0] AS type,
               e{{.* , embedding: Null, id: Null}} AS properties
        """

        records: List[dict] = []
        seen: Set[str] = set()
        for label in STORE_NODE_LABELS:
            rows = self.structured_query(
                f"MATCH (e:{escape_identifier(label)}) {where}{return_statement}",
                param_map=params,
            )
            for record in rows or []:
                # A node carrying both labels would otherwise be returned twice.
                if record["name"] in seen:
                    continue
                seen.add(record["name"])
                records.append(record)
        return records

    def get(
        self,
        properties: Optional[dict] = None,
        ids: Optional[List[str]] = None,
    ) -> List[LabelledNode]:
        """Get nodes."""
        params: Dict[str, Any] = {}
        conditions: List[str] = []

        if ids:
            conditions.append("e.id IN $ids")
            params["ids"] = ids

        if properties:
            for i, prop in enumerate(properties):
                conditions.append(f"e.{escape_identifier(prop)} = $property_{i}")
                params[f"property_{i}"] = properties[prop]

        response = self._match_store_nodes(conditions, params)

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
        cypher_statement = f"MATCH (e:{escape_identifier(BASE_ENTITY_LABEL)}) "

        params: Dict[str, Any] = {}
        conditions: List[str] = []

        if entity_names:
            conditions.append("e.name IN $entity_names")
            params["entity_names"] = entity_names

        if ids:
            conditions.append("e.id IN $ids")
            params["ids"] = ids

        if properties:
            for i, prop in enumerate(properties):
                conditions.append(f"e.{escape_identifier(prop)} = $property_{i}")
                params[f"property_{i}"] = properties[prop]

        if conditions:
            cypher_statement += "WHERE " + " AND ".join(conditions) + " "

        relation_filter = (
            ":" + "|".join(escape_identifier(rel) for rel in relation_names)
            if relation_names
            else ""
        )

        base = escape_identifier(BASE_ENTITY_LABEL)
        return_statement = f"""
        WITH e
        CALL {{
            WITH e
            MATCH (e)-[r{relation_filter}]->(t:{base})
            RETURN e.name AS source_id, [l in labels(e) WHERE l <> '{BASE_ENTITY_LABEL}' | l][0] AS source_type,
                   e{{.* , embedding: Null, name: Null}} AS source_properties,
                   type(r) AS type, r{{.*}} AS rel_properties,
                   t.name AS target_id, [l in labels(t) WHERE l <> '{BASE_ENTITY_LABEL}' | l][0] AS target_type,
                   t{{.* , embedding: Null, name: Null}} AS target_properties
            UNION ALL
            WITH e
            MATCH (e)<-[r{relation_filter}]-(t:{base})
            RETURN t.name AS source_id, [l in labels(t) WHERE l <> '{BASE_ENTITY_LABEL}' | l][0] AS source_type,
                   t{{.* , embedding: Null, name: Null}} AS source_properties,
                   type(r) AS type, r{{.*}} AS rel_properties,
                   e.name AS target_id, [l in labels(e) WHERE l <> '{BASE_ENTITY_LABEL}' | l][0] AS target_type,
                   e{{.* , embedding: Null, name: Null}} AS target_properties
        }}
        RETURN source_id, source_type, type, rel_properties, target_id, target_type, source_properties, target_properties"""
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
                properties=remove_empty_values(record["rel_properties"] or {}),
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
        if not ids:
            return triples

        # Filter the ignored relationships server side so that they do not eat
        # into the `limit` budget.
        ignored = list({*(ignore_rels or [])})

        response = self.structured_query(
            f"""
            WITH $ids AS id_list
            UNWIND range(0, size(id_list) - 1) AS idx
            MATCH (e:{escape_identifier(BASE_ENTITY_LABEL)})
            WHERE e.id = id_list[idx]
            MATCH p=(e)-[r*1..{int(depth)}]-(other)
            WHERE ALL(rel in relationships(p) WHERE type(rel) <> 'MENTIONS')
            UNWIND relationships(p) AS rel
            WITH distinct rel, idx
            WHERE NOT type(rel) IN $ignore_rels
            WITH startNode(rel) AS source,
                type(rel) AS type,
                rel{{.*}} AS rel_properties,
                endNode(rel) AS endNode,
                idx
            LIMIT $limit
            RETURN source.id AS source_id, [l in labels(source) WHERE l <> '__Entity__' | l][0] AS source_type,
                source{{.* , embedding: Null, id: Null}} AS source_properties,
                type,
                rel_properties,
                endNode.id AS target_id, [l in labels(endNode) WHERE l <> '__Entity__' | l][0] AS target_type,
                endNode{{.* , embedding: Null, id: Null}} AS target_properties,
                idx
            ORDER BY idx
            LIMIT $limit
            """,
            param_map={"ids": ids, "limit": int(limit), "ignore_rels": ignored},
        )
        response = response if response else []

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
                properties=remove_empty_values(record["rel_properties"] or {}),
            )
            triples.append([source, rel, target])

        return triples

    def structured_query(
        self, query: str, param_map: Optional[Dict[str, Any]] = None
    ) -> Any:
        param_map = param_map or {}

        result = self._graph.query(query, param_map, timeout=self._timeout)
        full_result = [
            {
                header[1]: to_plain_value(value)
                for header, value in zip(result.header, row)
            }
            for row in result.result_set
        ]

        if self.sanitize_query_output:
            return [value_sanitize(el) for el in full_result]
        return full_result

    def vector_query(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> Tuple[List[LabelledNode], List[float]]:
        """Query the graph store with a vector store query."""
        conditions = []
        filter_params: Dict[str, Any] = {}
        if query.filters:
            for index, filter_ in enumerate(query.filters.filters):
                conditions.append(
                    f"{'NOT ' if filter_.operator.value == 'nin' else ''}"
                    f"e.{escape_identifier(filter_.key)} "
                    f"{convert_operator(filter_.operator.value)} $param_{index}"
                )
                filter_params[f"param_{index}"] = filter_.value
        filters = (
            f" {query.filters.condition.value} ".join(conditions)
            if conditions
            else "1 = 1"
        )

        dimension = len(query.query_embedding or [])
        use_vector_index = (
            not conditions
            and self._vector_index_dimensions.get(BASE_ENTITY_LABEL) == dimension
        )

        if use_vector_index:
            data = self.structured_query(
                f"""CALL db.idx.vector.queryNodes(
                        '{BASE_ENTITY_LABEL}', '{EMBEDDING_KEY}',
                        $limit, vecf32($embedding))
                YIELD node AS e, score
                RETURN e.id AS name,
                    [l in labels(e) WHERE l <> '__Entity__' | l][0] AS type,
                    e{{.* , embedding: Null, name: Null, id: Null}} AS properties,
                    1 - score AS score
                ORDER BY score DESC""",
                param_map={
                    "embedding": query.query_embedding,
                    "limit": int(query.similarity_top_k),
                },
            )
        else:
            try:
                data = self.structured_query(
                    f"""MATCH (e:{escape_identifier(BASE_ENTITY_LABEL)})
                    WHERE e.{EMBEDDING_KEY} IS NOT NULL AND ({filters})
                    WITH e, 1 - vec.cosineDistance(e.{EMBEDDING_KEY}, vecf32($embedding)) AS score
                    ORDER BY score DESC LIMIT $limit
                    RETURN e.id AS name,
                    [l in labels(e) WHERE l <> '__Entity__' | l][0] AS type,
                    e{{.* , embedding: Null, name: Null, id: Null}} AS properties,
                    score""",
                    param_map={
                        "embedding": query.query_embedding,
                        "dimension": dimension,
                        "limit": int(query.similarity_top_k),
                        **filter_params,
                    },
                )
            except redis.exceptions.ResponseError as e:
                if "dimension mismatch" in str(e).lower():
                    raise ValueError(
                        "The graph contains embeddings whose dimension differs from "
                        f"the query embedding ({dimension}). Make sure a single "
                        "embedding model is used for the whole graph."
                    ) from e
                raise

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

        # As in `get()`, the match has to name a label for FalkorDB to be able
        # to use its (label-scoped) indexes instead of scanning the graph.
        def delete_by(condition: str, param_map: Dict[str, Any]) -> None:
            for label in STORE_NODE_LABELS:
                self.structured_query(
                    f"MATCH (e:{escape_identifier(label)}) WHERE {condition} "
                    "DETACH DELETE e",
                    param_map=param_map,
                )

        if entity_names:
            delete_by("e.name IN $entity_names", {"entity_names": entity_names})

        if ids:
            delete_by("e.id IN $ids", {"ids": ids})

        if relation_names:
            for rel in relation_names:
                self.structured_query(
                    f"MATCH ()-[r:{escape_identifier(rel)}]->() DELETE r"
                )

        if properties:
            prop_list = []
            params: Dict[str, Any] = {}
            for i, prop in enumerate(properties):
                prop_list.append(f"e.{escape_identifier(prop)} = $property_{i}")
                params[f"property_{i}"] = properties[prop]
            delete_by(" AND ".join(prop_list), params)

    ### ----- Connection management ----- ###

    def switch_graph(self, graph_name: str) -> None:
        """
        Switch to the given graph name (`graph_name`).

        This method allows users to change the active graph within the same
        database connection.

        Args:
            graph_name (str): The name of the graph to switch to.

        """
        self._graph = self._driver.select_graph(graph_name)
        self._database = graph_name

        if self._create_indexes:
            self._create_range_indexes()
        self._refresh_vector_index_info()

        try:
            self.refresh_schema()
        except Exception as e:
            raise ValueError(f"Could not refresh schema. Error: {e}")

    def close(self) -> None:
        """Explicitly close the FalkorDB connection."""
        if hasattr(self, "_driver"):
            try:
                self._driver.connection.close()
            finally:
                delattr(self, "_driver")

    def __enter__(self) -> "FalkorDBPropertyGraphStore":
        """Enter the runtime context, enabling `with FalkorDBPropertyGraphStore(...)`."""
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """Close the connection when leaving the runtime context."""
        self.close()

    def __del__(self) -> None:
        """Best-effort cleanup; prefer `close()` or the context manager."""
        try:
            self.close()
        except Exception:
            # Suppress any exceptions during garbage collection
            pass


FalkorDBPGStore = FalkorDBPropertyGraphStore
