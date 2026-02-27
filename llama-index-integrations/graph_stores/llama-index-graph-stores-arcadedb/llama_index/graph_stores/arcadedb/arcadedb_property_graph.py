"""ArcadeDB PropertyGraphStore for LlamaIndex.

Drop-in ``PropertyGraphStore`` implementation backed by ArcadeDB via the Bolt
protocol.  Uses the standard ``neo4j`` Python driver — no APOC required.

ArcadeDB enforces a single type per vertex, so semantic labels (``PERSON``,
``CITY``, …) are stored as the ``label`` property on the ``Entity`` vertex
type.  ``Chunk`` is a separate vertex type for text chunks.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from llama_index.core.graph_stores.prompts import DEFAULT_CYPHER_TEMPALTE
from llama_index.core.graph_stores.types import (
    ChunkNode,
    EntityNode,
    LabelledNode,
    PropertyGraphStore,
    Relation,
    Triplet,
)
from llama_index.core.graph_stores.utils import value_sanitize
from llama_index.core.prompts import PromptTemplate
from llama_index.core.vector_stores.types import VectorStoreQuery
from neo4j import GraphDatabase

logger = logging.getLogger(__name__)

_DEFAULT_URI = "bolt://localhost:7687"

# ArcadeDB internal properties to strip from results.
_ARCADEDB_INTERNAL_PROPS = frozenset({"@rid", "@type", "@cat", "@in", "@out"})

# Python type → schema type string (matches neo4j_graphrag format).
_TYPE_MAP: dict[type, str] = {
    bool: "BOOLEAN",  # Must precede int (bool is subclass of int).
    int: "INTEGER",
    float: "FLOAT",
    str: "STRING",
    list: "LIST",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _infer_type(value: Any) -> str:
    """Infer a schema type string from a Python value."""
    for py_type, type_str in _TYPE_MAP.items():
        if isinstance(value, py_type):
            return type_str
    return "STRING"


def _clean_properties(props: dict) -> dict:
    """Remove ArcadeDB internal properties and empty values."""
    return {
        k: v
        for k, v in props.items()
        if k not in _ARCADEDB_INTERNAL_PROPS and v is not None and v != ""
    }


def _strip_embedding(props: dict) -> dict:
    """Remove embedding from properties (for display)."""
    return {k: v for k, v in props.items() if k != "embedding"}


def _strip_meta(props: dict) -> dict:
    """Remove node metadata keys (id, name, label, text, embedding)."""
    return {
        k: v
        for k, v in props.items()
        if k not in ("id", "name", "label", "text", "embedding")
    }


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class ArcadeDBPropertyGraphStore(PropertyGraphStore):
    """LlamaIndex ``PropertyGraphStore`` backed by ArcadeDB over Bolt.

    Example::

        from llama_index.graph_stores.arcadedb import ArcadeDBPropertyGraphStore

        graph_store = ArcadeDBPropertyGraphStore(
            url="bolt://localhost:7687",
            username="root",
            password="playwithdata",
            database="mydb",
        )

    Args:
        url: Bolt endpoint URL.  Falls back to ``ARCADEDB_BOLT_URL`` env var.
        username: Database username.  Falls back to ``ARCADEDB_USERNAME``.
        password: Database password.  Falls back to ``ARCADEDB_PASSWORD``.
        database: Database name.  Falls back to ``ARCADEDB_DATABASE``.
        sanitize_query_output: Strip embeddings from query output.
        refresh_schema: Refresh schema on initialisation.
        driver_config: Extra keyword arguments for the Neo4j driver.
    """

    supports_structured_queries: bool = True
    supports_vector_queries: bool = True
    text_to_cypher_template: PromptTemplate = DEFAULT_CYPHER_TEMPALTE

    def __init__(
        self,
        url: str | None = None,
        username: str | None = None,
        password: str | None = None,
        database: str | None = None,
        *,
        sanitize_query_output: bool = True,
        refresh_schema: bool = True,
        driver_config: dict[str, Any] | None = None,
    ) -> None:
        self._url = url or os.environ.get("ARCADEDB_BOLT_URL", _DEFAULT_URI)
        self._username = username or os.environ.get("ARCADEDB_USERNAME", "root")
        self._password = password or os.environ.get(
            "ARCADEDB_PASSWORD", "playwithdata"
        )
        self._database = database or os.environ.get("ARCADEDB_DATABASE", "")
        self.sanitize_query_output = sanitize_query_output
        self._refresh_schema_on_init = refresh_schema

        self._driver_config = driver_config or {}
        self._driver: Any = None
        self._initialized = False
        self.structured_schema: dict[str, Any] = {}

    def _lazy_init(self) -> None:
        """Lazily initialise the driver, verify connectivity, and create types.

        Called automatically before the first database operation so that the
        constructor stays free of network I/O and side-effects.
        """
        if self._initialized:
            return

        self._driver = GraphDatabase.driver(
            self._url,
            auth=(self._username, self._password),
            **self._driver_config,
        )

        try:
            self._driver.verify_connectivity()
        except Exception as e:
            msg = f"Could not connect to ArcadeDB at {self._url}: {e}"
            raise ConnectionError(msg) from e

        self._ensure_types()
        self._initialized = True

        if self._refresh_schema_on_init:
            self.refresh_schema()

    @property
    def client(self) -> Any:
        """Return the underlying Neo4j driver instance."""
        self._lazy_init()
        return self._driver

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    def _execute(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a Cypher (or ArcadeDB SQL) query and return dicts."""
        self._lazy_init()
        params = params or {}
        try:
            if self._database:
                records, _, _ = self._driver.execute_query(
                    query, params, database_=self._database
                )
            else:
                records, _, _ = self._driver.execute_query(query, params)
        except Exception:
            logger.exception("Query failed: %s | params: %s", query, params)
            raise
        return [dict(r) for r in records]

    def _ensure_types(self) -> None:
        """Create ArcadeDB vertex types and unique indexes if needed."""
        for stmt in (
            "CREATE VERTEX TYPE Entity IF NOT EXISTS",
            "CREATE VERTEX TYPE Chunk IF NOT EXISTS",
        ):
            try:
                self._execute(stmt)
            except Exception:
                logger.debug("Type creation skipped: %s", stmt)

        for stmt in (
            "CREATE INDEX ON Entity (id) UNIQUE",
            "CREATE INDEX ON Chunk (id) UNIQUE",
        ):
            try:
                self._execute(stmt)
            except Exception:
                logger.debug("Index creation skipped: %s", stmt)

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def refresh_schema(self) -> None:
        """Refresh the cached schema by introspecting the graph."""
        # Node labels
        try:
            labels = [
                r["label"]
                for r in self._execute(
                    "MATCH (n) RETURN DISTINCT labels(n)[0] AS label"
                )
                if r.get("label")
            ]
        except Exception:
            labels = []

        # Node properties per label
        node_props: dict[str, list[dict[str, str]]] = {}
        for label in labels:
            try:
                records = self._execute(
                    f"MATCH (n:`{label}`) RETURN n LIMIT 25"  # noqa: S608
                )
                props: dict[str, str] = {}
                for record in records:
                    for key, value in dict(record["n"]).items():
                        if (
                            key not in _ARCADEDB_INTERNAL_PROPS
                            and key not in props
                            and value is not None
                        ):
                            props[key] = _infer_type(value)
                node_props[label] = [
                    {"property": k, "type": v} for k, v in props.items()
                ]
            except Exception:
                logger.debug("Schema: failed to get props for %s", label)

        # Relationship types
        try:
            rel_types = [
                r["type"]
                for r in self._execute(
                    "MATCH ()-[r]->() RETURN DISTINCT type(r) AS type"
                )
                if r.get("type")
            ]
        except Exception:
            rel_types = []

        # Relationship properties per type
        rel_props: dict[str, list[dict[str, str]]] = {}
        for rel_type in rel_types:
            try:
                records = self._execute(
                    f"MATCH ()-[r:`{rel_type}`]->() RETURN r LIMIT 25"  # noqa: S608
                )
                props = {}
                for record in records:
                    for key, value in dict(record["r"]).items():
                        if (
                            key not in _ARCADEDB_INTERNAL_PROPS
                            and key not in props
                            and value is not None
                        ):
                            props[key] = _infer_type(value)
                rel_props[rel_type] = [
                    {"property": k, "type": v} for k, v in props.items()
                ]
            except Exception:
                logger.debug("Schema: failed to get props for rel %s", rel_type)

        # Relationship patterns
        try:
            relationships = [
                {"start": r["start"], "type": r["type"], "end": r["end"]}
                for r in self._execute(
                    "MATCH (a)-[r]->(b) "
                    "RETURN DISTINCT labels(a)[0] AS start, "
                    "type(r) AS type, labels(b)[0] AS end"
                )
                if r.get("start") and r.get("end")
            ]
        except Exception:
            relationships = []

        self.structured_schema = {
            "node_props": node_props,
            "rel_props": rel_props,
            "relationships": relationships,
            "metadata": {"constraint": [], "index": []},
        }

    def get_schema(self, refresh: bool = False) -> Any:
        """Return the structured schema dictionary."""
        if refresh:
            self.refresh_schema()
        return self.structured_schema

    def get_schema_str(self, refresh: bool = False) -> str:
        """Return a human-readable schema string for LLM context."""
        schema = self.get_schema(refresh=refresh)
        lines: list[str] = ["Node properties:"]
        for label, props in schema.get("node_props", {}).items():
            prop_strs = [f"{p['property']}: {p['type']}" for p in props]
            lines.append(f"  {label} {{{', '.join(prop_strs)}}}")

        lines.append("Relationship properties:")
        for rel_type, props in schema.get("rel_props", {}).items():
            prop_strs = [f"{p['property']}: {p['type']}" for p in props]
            lines.append(f"  {rel_type} {{{', '.join(prop_strs)}}}")

        lines.append("Relationships:")
        for rel in schema.get("relationships", []):
            lines.append(
                f"  (:{rel['start']})-[:{rel['type']}]->(:{rel['end']})"
            )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Upsert
    # ------------------------------------------------------------------

    def upsert_nodes(self, nodes: Sequence[LabelledNode]) -> None:
        """Upsert nodes into ArcadeDB."""
        entity_dicts: list[dict] = []
        chunk_dicts: list[dict] = []

        for item in nodes:
            if isinstance(item, EntityNode):
                entity_dicts.append({**item.model_dump(), "id": item.id})
            elif isinstance(item, ChunkNode):
                chunk_dicts.append({**item.model_dump(), "id": item.id})

        # --- Chunks ---
        for chunk in chunk_dicts:
            props = _clean_properties(chunk.get("properties", {}))
            self._execute(
                "MERGE (c:Chunk {id: $id}) "
                "SET c.text = $text "
                "SET c += $properties",
                {
                    "id": chunk["id"],
                    "text": chunk.get("text", ""),
                    "properties": props,
                },
            )
            embedding = chunk.get("embedding")
            if embedding is not None:
                self._execute(
                    "MATCH (c:Chunk {id: $id}) SET c.embedding = $embedding",
                    {"id": chunk["id"], "embedding": embedding},
                )

        # --- Entities ---
        for entity in entity_dicts:
            props = _clean_properties(entity.get("properties", {}))
            self._execute(
                "MERGE (e:Entity {id: $id}) "
                "SET e += $properties "
                "SET e.name = $name "
                "SET e.label = $label",
                {
                    "id": entity["id"],
                    "name": entity.get("name", entity["id"]),
                    "label": entity.get("label", "entity"),
                    "properties": props,
                },
            )
            embedding = entity.get("embedding")
            if embedding is not None:
                self._execute(
                    "MATCH (e:Entity {id: $id}) SET e.embedding = $embedding",
                    {"id": entity["id"], "embedding": embedding},
                )

            # Link to source chunk if triplet_source_id is present
            triplet_source_id = entity.get("properties", {}).get(
                "triplet_source_id"
            )
            if triplet_source_id:
                self._execute(
                    "MATCH (e:Entity {id: $entity_id}) "
                    "MERGE (c:Chunk {id: $chunk_id}) "
                    "MERGE (e)<-[:MENTIONS]-(c)",
                    {
                        "entity_id": entity["id"],
                        "chunk_id": triplet_source_id,
                    },
                )

    def upsert_relations(self, relations: List[Relation]) -> None:
        """Upsert relations into ArcadeDB."""
        for rel in relations:
            props = _clean_properties(rel.properties)
            cypher = (
                f"MATCH (source {{id: $source_id}}) "  # noqa: S608
                f"MATCH (target {{id: $target_id}}) "
                f"MERGE (source)-[r:`{rel.label}`]->(target) "
                f"SET r += $properties"
            )
            self._execute(
                cypher,
                {
                    "source_id": rel.source_id,
                    "target_id": rel.target_id,
                    "properties": props,
                },
            )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get(
        self,
        properties: Optional[dict] = None,
        ids: Optional[List[str]] = None,
    ) -> List[LabelledNode]:
        """Get nodes matching the given properties or IDs."""
        cypher = "MATCH (e) "
        params: dict[str, Any] = {}
        conditions: list[str] = []

        if ids:
            conditions.append("e.id IN $ids")
            params["ids"] = ids

        if properties:
            for i, (key, value) in enumerate(properties.items()):
                conditions.append(f"e.`{key}` = $prop_{i}")
                params[f"prop_{i}"] = value

        if conditions:
            cypher += "WHERE " + " AND ".join(conditions) + " "

        cypher += (
            "RETURN e.id AS name, labels(e)[0] AS node_type, "
            "e.label AS label, e AS props"
        )

        response = self._execute(cypher, params)
        nodes: list[LabelledNode] = []
        for record in response:
            raw_props = dict(record.get("props") or {})
            node_type = record.get("node_type", "")
            is_chunk = node_type == "Chunk" or "text" in raw_props

            cleaned = _strip_meta(_clean_properties(raw_props))

            if is_chunk:
                text = raw_props.get("text", "")
                nodes.append(
                    ChunkNode(
                        id_=record["name"],
                        text=text,
                        properties=cleaned,
                    )
                )
            else:
                nodes.append(
                    EntityNode(
                        name=record["name"],
                        label=record.get("label") or "entity",
                        properties=cleaned,
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
        """Get triplets matching the given criteria.

        Runs two queries (outgoing + incoming) because ArcadeDB may not
        support ``CALL {}`` sub-queries with ``UNION ALL``.
        """
        params: dict[str, Any] = {}
        where_parts: list[str] = []

        if entity_names:
            where_parts.append("e.name IN $entity_names")
            params["entity_names"] = entity_names
        if ids:
            where_parts.append("e.id IN $ids")
            params["ids"] = ids
        if properties:
            for i, (key, value) in enumerate(properties.items()):
                where_parts.append(f"e.`{key}` = $prop_{i}")
                params[f"prop_{i}"] = value

        where = "WHERE " + " AND ".join(where_parts) if where_parts else ""

        # Relationship type filter
        rel_filter = ""
        if relation_names:
            rel_types = "|".join(f"`{r}`" for r in relation_names)
            rel_filter = f":{rel_types}"

        return_clause = (
            "RETURN e.name AS source_id, e.label AS source_type, e AS source_props, "
            "type(r) AS rel_type, "
            "t.name AS target_id, t.label AS target_type, t AS target_props"
        )

        q_out = (
            f"MATCH (e:Entity) {where} "  # noqa: S608
            f"MATCH (e)-[r{rel_filter}]->(t:Entity) "
            f"{return_clause}"
        )

        q_in = (
            f"MATCH (e:Entity) {where} "  # noqa: S608
            f"MATCH (e)<-[r{rel_filter}]-(t:Entity) "
            "RETURN t.name AS source_id, t.label AS source_type, t AS source_props, "
            "type(r) AS rel_type, "
            "e.name AS target_id, e.label AS target_type, e AS target_props"
        )

        triples: list[Triplet] = []
        seen: set[tuple[str, str, str]] = set()

        for query in (q_out, q_in):
            data = self._execute(query, params)
            for record in data:
                key = (
                    record["source_id"],
                    record["rel_type"],
                    record["target_id"],
                )
                if key in seen:
                    continue
                seen.add(key)

                source_props = _strip_meta(
                    _strip_embedding(
                        _clean_properties(dict(record.get("source_props") or {}))
                    )
                )
                target_props = _strip_meta(
                    _strip_embedding(
                        _clean_properties(dict(record.get("target_props") or {}))
                    )
                )

                source = EntityNode(
                    name=record["source_id"],
                    label=record.get("source_type") or "entity",
                    properties=source_props,
                )
                target = EntityNode(
                    name=record["target_id"],
                    label=record.get("target_type") or "entity",
                    properties=target_props,
                )
                rel = Relation(
                    source_id=record["source_id"],
                    target_id=record["target_id"],
                    label=record["rel_type"],
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
        """Get depth-aware relationship map (replaces ``apoc.path.expand``)."""
        triples: list[Triplet] = []
        ignore_rels = ignore_rels or []
        ids = [node.id for node in graph_nodes]

        cypher = (
            f"MATCH (e:Entity) WHERE e.id IN $ids "  # noqa: S608
            f"MATCH (e)-[r*1..{depth}]-(other) "
            "UNWIND r AS rel "
            "WITH DISTINCT rel "
            "WITH startNode(rel) AS source, type(rel) AS rel_type, "
            "endNode(rel) AS target "
            "RETURN source.id AS source_id, source.label AS source_type, "
            "source AS source_props, rel_type, "
            "target.id AS target_id, target.label AS target_type, "
            "target AS target_props "
            "LIMIT $limit"
        )

        data = self._execute(cypher, {"ids": ids, "limit": limit})

        for record in data:
            if record["rel_type"] in ignore_rels:
                continue

            source_props = _strip_meta(
                _strip_embedding(
                    _clean_properties(dict(record.get("source_props") or {}))
                )
            )
            target_props = _strip_meta(
                _strip_embedding(
                    _clean_properties(dict(record.get("target_props") or {}))
                )
            )

            source = EntityNode(
                name=record["source_id"] or "",
                label=record.get("source_type") or "entity",
                properties=source_props,
            )
            target = EntityNode(
                name=record["target_id"] or "",
                label=record.get("target_type") or "entity",
                properties=target_props,
            )
            rel = Relation(
                source_id=record["source_id"] or "",
                target_id=record["target_id"] or "",
                label=record["rel_type"],
            )
            triples.append([source, rel, target])

        return triples

    def structured_query(
        self, query: str, param_map: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Execute a Cypher query and return sanitised results."""
        param_map = param_map or {}
        result = self._execute(query, param_map)
        if self.sanitize_query_output:
            return [value_sanitize(el) for el in result]
        return result

    def vector_query(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> Tuple[List[LabelledNode], List[float]]:
        """Vector similarity search using ArcadeDB's native HNSW index.

        Falls back to Python-side cosine similarity when native search is
        unavailable.
        """
        if not query.query_embedding:
            return [], []

        limit = query.similarity_top_k or 10

        # Build filter conditions
        filters = "1 = 1"
        if query.filters and query.filters.filters:
            conditions = [
                f"e.{f.key} {f.operator.value} {f.value}"
                for f in query.filters.filters
            ]
            filters = (
                f" {query.filters.condition.value} ".join(conditions)
                .replace("==", "=")
            )

        # Try native ArcadeDB vector search (SQL via Bolt)
        embedding_str = ", ".join(str(x) for x in query.query_embedding)
        try:
            sql = (
                f"SELECT *, $similarity AS score FROM Entity "  # noqa: S608
                f"WHERE embedding NEAR [{embedding_str}] "
                f"LIMIT {limit}"
            )
            data = self._execute(sql)
        except Exception:
            logger.debug(
                "Native vector search unavailable, falling back to brute-force"
            )
            data = self._brute_force_vector_search(query.query_embedding, limit)

        nodes: list[LabelledNode] = []
        scores: list[float] = []
        for record in data:
            props = _clean_properties(dict(record))
            name = props.pop("name", props.pop("id", ""))
            label = props.pop("label", "entity")
            score = props.pop("score", 0.0)
            props.pop("embedding", None)

            node = EntityNode(
                name=str(name),
                label=label or "entity",
                properties=props,
            )
            nodes.append(node)
            scores.append(float(score))

        return nodes, scores

    def _brute_force_vector_search(
        self, embedding: list[float], limit: int
    ) -> list[dict]:
        """Python-side cosine similarity fallback for vector search."""
        data = self._execute(
            "MATCH (e:Entity) WHERE e.embedding IS NOT NULL RETURN e"
        )

        scored = []
        for record in data:
            node = dict(record.get("e") or {})
            node_embedding = node.get("embedding")
            if node_embedding and isinstance(node_embedding, list):
                score = _cosine_similarity(embedding, node_embedding)
                node["score"] = score
                scored.append(node)

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:limit]

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete(
        self,
        entity_names: Optional[List[str]] = None,
        relation_names: Optional[List[str]] = None,
        properties: Optional[dict] = None,
        ids: Optional[List[str]] = None,
    ) -> None:
        """Delete matching nodes and/or relations."""
        if entity_names:
            self._execute(
                "MATCH (n) WHERE n.name IN $names DETACH DELETE n",
                {"names": entity_names},
            )

        if ids:
            self._execute(
                "MATCH (n) WHERE n.id IN $ids DETACH DELETE n",
                {"ids": ids},
            )

        if relation_names:
            for rel_name in relation_names:
                self._execute(f"MATCH ()-[r:`{rel_name}`]->() DELETE r")

        if properties:
            conditions = []
            params: dict[str, Any] = {}
            for i, (key, value) in enumerate(properties.items()):
                conditions.append(f"e.`{key}` = $prop_{i}")
                params[f"prop_{i}"] = value
            self._execute(
                "MATCH (e) WHERE " + " AND ".join(conditions) + " DETACH DELETE e",
                params,
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the Bolt driver connection."""
        if self._driver is not None:
            self._driver.close()

    def __del__(self) -> None:
        try:
            if self._driver is not None:
                self._driver.close()
        except Exception:  # noqa: BLE001, S110
            pass


# Convenience alias
ArcadeDBPGStore = ArcadeDBPropertyGraphStore
