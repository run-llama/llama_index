import hashlib
import logging
import re
import time
from typing import Callable, Dict, List, Optional, Any, cast

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from neo4j import GraphDatabase

# Configure Logging
logger = logging.getLogger("EntityUpserter")
logger.setLevel(logging.INFO)


# --- 1. Normalization Logic (The Soul of Deduplication) ---
def normalize_entity_name(name: str) -> str:
    """
    Standardize entity names to prevent duplication (e.g., 'Back-Prop' vs 'back prop').
    Policy: Lowercase + Remove special chars + Strip.
    """
    if not name:
        return ""
    name = name.lower().strip()
    name = re.sub(r"[^\w\s]", "", name)
    return re.sub(r"\s+", " ", name)


def deterministic_id_for_name(norm_name: str) -> int:
    """
    Generates a deterministic 64-bit integer ID from the NORMALIZED name.
    Strictly depends on normalized string to ensure collision for variations.
    """
    if not norm_name:
        return 0
    h = hashlib.sha1(norm_name.encode("utf-8")).hexdigest()
    return int(h[:16], 16)


def merge_vectors(existing: Optional[List[float]], new: List[float]) -> List[float]:
    """Average two vectors."""
    if not existing:
        return new
    if not new:
        return existing
    if len(existing) != len(new):
        return new
    return [(a + b) / 2.0 for a, b in zip(existing, new)]


# --- 2. Robust Qdrant Search ---
def find_point_by_normalized_name(
    client: QdrantClient, collection: str, norm_name: str
) -> Optional[Dict[str, Any]]:
    """
    Search Qdrant by 'normalized_name' (Payload Field).
    Handles different qdrant-client return formats (Object vs Dict).
    """
    try:
        flt = rest.Filter(
            must=[
                rest.FieldCondition(
                    key="normalized_name", match=rest.MatchValue(value=norm_name)
                )
            ]
        )

        # Try-Catch for SDK version compatibility (scroll_filter vs filter)
        try:
            res = client.scroll(
                collection_name=collection,
                scroll_filter=flt,
                limit=1,
                with_payload=True,
                with_vectors=True,
            )
        except TypeError:
            # Fallback for older/newer SDKs
            res = client.scroll(
                collection_name=collection,
                filter=flt,
                limit=1,
                with_payload=True,
                with_vectors=True,
            )

        # Parse Result (Tuple: (points, offset))
        points = res[0] if isinstance(res, tuple) else res

        if points:
            p = cast(Any, points[0])
            # Handle Object vs Dict access
            pid = getattr(p, "id", None) if hasattr(p, "id") else p.get("id")
            payload = (
                getattr(p, "payload", None)
                if hasattr(p, "payload")
                else p.get("payload")
            )
            vector = (
                getattr(p, "vector", None) if hasattr(p, "vector") else p.get("vector")
            )

            return {"id": pid, "payload": payload, "vector": vector}

    except Exception as e:
        logger.debug(f"Search failed for {norm_name}: {e}")
    return None


def upsert_entities_to_qdrant(
    client: QdrantClient,
    collection: str,
    entities: List[Dict],
    embed_fn: Optional[Callable[[str], List[float]]] = None,
    batch_size: int = 64,
) -> Dict[str, Any]:
    """
    Batch Upsert to Qdrant with Strict Normalization & Idempotency.
    """
    stats = {"inserted": 0, "updated": 0, "skipped": 0, "time_ms": 0}
    start_time = time.time()
    points_batch = []

    for ent in entities:
        raw_name = ent.get("name")
        if not raw_name:
            stats["skipped"] += 1
            continue

        # [Strict Policy] Use Normalized Name for ID & Search
        norm_name = normalize_entity_name(raw_name)
        if not norm_name:
            stats["skipped"] += 1
            continue

        props = ent.get("properties", {}) or {}
        new_vec = ent.get("vector")

        # Embedding: Use RAW name for better semantic representation
        if (not new_vec) and embed_fn:
            try:
                new_vec = embed_fn(raw_name)
            except Exception:
                new_vec = None

        if not new_vec:
            stats["skipped"] += 1
            continue

        # Generate ID from NORMALIZED name
        pid = deterministic_id_for_name(norm_name)

        # Check existence by NORMALIZED name
        existing = find_point_by_normalized_name(client, collection, norm_name)

        if existing:
            final_vec = merge_vectors(existing.get("vector"), new_vec)
            # Update payload, keep normalized_name constant
            final_payload = {
                **(existing.get("payload") or {}),
                **props,
                "name": raw_name,
                "normalized_name": norm_name,
            }

            # Use existing ID to be safe
            point = rest.PointStruct(
                id=existing["id"], vector=final_vec, payload=final_payload
            )
            stats["updated"] += 1
        else:
            # INSERT
            final_payload = {"name": raw_name, "normalized_name": norm_name, **props}
            point = rest.PointStruct(id=pid, vector=new_vec, payload=final_payload)
            stats["inserted"] += 1

        points_batch.append(point)

        # Batch Push
        if len(points_batch) >= batch_size:
            try:
                client.upsert(collection_name=collection, points=points_batch)
            except Exception as e:
                logger.error(f"Qdrant batch error: {e}")
            points_batch = []

    # Final Flush
    if points_batch:
        try:
            client.upsert(collection_name=collection, points=points_batch)
        except Exception as e:
            logger.error(f"Qdrant final batch error: {e}")

    stats["time_ms"] = int((time.time() - start_time) * 1000)
    return stats


def neo4j_merge_entities(
    neo4j_uri: str,
    user: str,
    password: str,
    entities: List[Dict],
    batch_size: int = 200,
) -> Dict[str, Any]:
    """Batch MERGE to Neo4j with Normalization Property."""
    stats = {"upserted": 0, "time_ms": 0}
    start_time = time.time()

    try:
        driver = GraphDatabase.driver(neo4j_uri, auth=(user, password))
    except Exception:
        return stats

    try:
        with driver.session() as session:
            for i in range(0, len(entities), batch_size):
                batch = entities[i : i + batch_size]
                rows = []
                for e in batch:
                    raw_name = e.get("name")
                    if not raw_name:
                        continue

                    safe_props = {
                        k: v
                        for k, v in e.get("properties", {}).items()
                        if isinstance(v, (str, int, float, bool))
                    }

                    # Store normalized name for querying
                    safe_props["normalized_name"] = normalize_entity_name(raw_name)

                    rows.append({"name": raw_name, "props": safe_props})

                # Cypher UNWIND MERGE
                query = """
                UNWIND $rows AS row
                MERGE (e:Entity {name: row.name})
                SET e += row.props
                RETURN count(e) as c
                """
                try:
                    session.run(query, parameters={"rows": rows})
                    stats["upserted"] += len(rows)
                except Exception as e:
                    logger.error(f"Neo4j batch error: {e}")
    finally:
        driver.close()

    stats["time_ms"] = int((time.time() - start_time) * 1000)
    return stats


def upsert_entities(
    qdrant_client: QdrantClient,
    qdrant_collection: str,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    entities: List[Dict],
    embed_fn: Optional[Callable[[str], List[float]]] = None,
    qdrant_batch_size: int = 64,
) -> Dict[str, Dict]:
    """Main Entry Point."""
    logger.info(
        f"🔄 Upserting {len(entities)} entities (deduplicated by normalized name)..."
    )

    q_metrics = upsert_entities_to_qdrant(
        qdrant_client, qdrant_collection, entities, embed_fn, qdrant_batch_size
    )
    n_metrics = neo4j_merge_entities(neo4j_uri, neo4j_user, neo4j_password, entities)

    logger.info(
        f"✅ [Qdrant] +{q_metrics['inserted']} / ~{q_metrics['updated']} ({q_metrics['time_ms']}ms)"
    )
    logger.info(
        f"✅ [Neo4j] Merged: {n_metrics['upserted']} ({n_metrics['time_ms']}ms)"
    )

    return {"qdrant": q_metrics, "neo4j": n_metrics}
