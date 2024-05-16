from typing import Any, List, Dict, Optional
import neo4j

from llama_index.core.graph_stores.types import (
    LabelledPropertyGraphStore,
    Triplet,
    LabelledNode,
    Relation,
    EntityNode,
    ChunkNode
)

class Neo4jLPGStore(LabelledPropertyGraphStore):
    """
    Simple Labelled Property Graph Store.

    This class implements a simple in-memory labelled property graph store.

    Args:
        graph (Optional[LabelledPropertyGraph]): Labelled property graph to initialize the store.
    """

    supports_structured_queries: bool = False
    supports_vector_queries: bool = False

    def __init__(
        self,
        username: str,
        password: str,
        url: str,
        database: Optional[str] = "neo4j"
    ) -> None:
        self._driver = neo4j.GraphDatabase.driver(url, auth=(username, password))
        self._database = database

    def get(
        self,
        properties: Optional[dict] = None,
        ids: Optional[List[str]] = None,
    ) -> List[LabelledNode]:
        """Get nodes."""
        print(properties, ids)

    def upsert_nodes(self, nodes: List[LabelledNode]) -> None:
        # Lists to hold separated types
        entity_nodes = []
        chunk_nodes = []
        # Sort by type
        for item in nodes:
            if isinstance(item, EntityNode):
                entity_nodes.append(item)
            elif isinstance(item, ChunkNode):
                chunk_nodes.append(item)
            else:
                # Log that we do not support these types of nodes
                # Or raise an error?
                pass
        if chunk_nodes:
            self.database_query("""
            UNWIND $data AS row 
            MERGE (c:Chunk {id: row.id_}) 
            SET c.text = row.text
            WITH c, row
            SET c += row.properties
            WITH c, row.embedding AS embedding
            WHERE embedding IS NOT NULL
            CALL db.create.setNodeVectorProperty(c, 'embedding', embedding)
            RETURN count(*)
            """, param_map={"data": [el.dict() for el in chunk_nodes]})
        if entity_nodes:
            self.database_query("""
            UNWIND $data AS row
            MERGE (e:`__Entity__` {name: row.name})
            SET e += apoc.map.clean(row.properties, ['triplet_source_id'], [])
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
            MERGE (e)-[:SOURCE_CHUNK]->(c)
            """, param_map={"data": [el.dict() for el in entity_nodes]})

    def upsert_relations(self, relations: List[Relation]) -> None:
        """Add relations."""
        params = [r.dict() for r in relations]
        self.database_query("""
        UNWIND $data AS row
        MERGE (source:`__Entity__` {name: row.source_id})
        MERGE (target:`__Entity__` {name: row.target_id})
        WITH source, target, row
        CALL apoc.merge.relationship(source, row.label, row.properties, {}, target) YIELD rel
        RETURN count(*)        
        """, param_map={"data": params})

    def get_triplets(
            self,
            entity_names: Optional[List[str]] = None,
            relation_names: Optional[List[str]] = None,
            properties: Optional[dict] = None,
            ids: Optional[List[str]] = None) -> List[Triplet]:
        # TODO: Handle ids
        cypher_statement = "MATCH (e:`__Entity__`) "
        params = {}
        if entity_names or properties or ids:
            cypher_statement += "WHERE "
        if entity_names:
            cypher_statement += "e.name in $entity_names "
            params["entity_names"] = entity_names

        if properties:
            prop_list = []
            for i, prop in enumerate(properties):
                prop_list.append(f"e.`{prop}` = $property_{i}")
                params[f"property_{i}"] = entity_names
            cypher_statement += " AND ".join(prop_list)


        # TODO: Add node properties?
        return_statement = f"""
        WITH e
        CALL {{
            WITH e
            MATCH (e)-[r{':`' + '`|`'.join(relation_names) + '`' if relation_names else ''}]->(t)
            RETURN e.name AS source_id, [l in labels(e) WHERE l <> 'Entity' | l][0] AS source_type,
                   type(r) AS type,
                   t.name AS target_id, [l in labels(t) WHERE l <> 'Entity' | l][0] AS target_type
            UNION ALL
            MATCH (e)<-[r{':`' + '`|`'.join(relation_names) + '`' if relation_names else ''}]-(t)
            RETURN t.name AS source_id, [l in labels(t) WHERE l <> 'Entity' | l][0] AS source_type,
                   type(r) AS type,
                   e.name AS target_id, [l in labels(e) WHERE l <> 'Entity' | l][0] AS target_type
        }}
        RETURN source_id, source_type, type, target_id, target_type"""
        cypher_statement += return_statement
        data = self.database_query(cypher_statement, param_map=params)
        triples = []
        for record in data:
            source = LabelledNode(name=record["source_id"], type=record["source_type"])
            target = LabelledNode(name=record["target_id"], type=record["target_type"])
            rel = Relation(source_id=record["source_id"], target_id=record["target_id"], label=record["type"])
            triples.append([source, rel, target])
        return triples
    
    def database_query(self, query: str, param_map: Optional[Dict[str, Any]] = {}) -> Any:
        with self._driver.session(database=self._database) as session:
            result = session.run(query, param_map)
            return [d.data() for d in result]

    def delete(
        self,
        entity_names: Optional[List[str]] = None,
        relation_names: Optional[List[str]] = None,
        properties: Optional[dict] = None,
        ids: Optional[List[str]] = None,
    ) -> None:
        """Delete matching data."""
    def get_rel_map(self ):
        pass
    def get_schema(self):
        pass
    def persist(self):
        pass
    def structured_query(self):
        pass
    def upsert_triplets(self):
        pass
    def vector_query(self):
        pass
