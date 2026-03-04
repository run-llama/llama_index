from abc import abstractmethod
import logging
from llama_index.core.prompts import PromptTemplate
from llama_index.core.graph_stores.prompts import DEFAULT_CYPHER_TEMPALTE
from typing import Any, Dict, List, Tuple, Optional
from llama_index.core.graph_stores.types import (
    LabelledNode,
    PropertyGraphStore,
    Relation,
    ChunkNode,
    EntityNode,
    Triplet,
)
from llama_index.core.vector_stores.types import VectorStoreQuery
from .neptune import remove_empty_values, refresh_schema

logger = logging.getLogger(__name__)
BASE_ENTITY_LABEL = "__Entity__"
BASE_NODE_LABEL = "__Node__"


class NeptuneBasePropertyGraph(PropertyGraphStore):
    supports_structured_queries: bool = True
    text_to_cypher_template: PromptTemplate = DEFAULT_CYPHER_TEMPALTE
    schema = None
    structured_schema = None

    def __init__() -> None:
        pass

    @property
    def client(self) -> Any:
        return self._client

    def get(
        self, properties: Dict = None, ids: List[str] = None, exact_match: bool = True
    ) -> List[LabelledNode]:
        """
        Get the nodes from the graph.

        Args:
            properties (Dict | None, optional): The properties to retrieve. Defaults to None.
            ids (List[str] | None, optional): A list of ids to find in the graph. Defaults to None.
            exact_match (bool, optional): Whether to do exact match on properties. Defaults to True.

        Returns:
            List[LabelledNode]: A list of nodes returned

        """
        cypher_statement = "MATCH (e) "

        params = {}
        if properties or ids:
            cypher_statement += "WHERE "

        if ids:
            if exact_match:
                cypher_statement += "e.id IN $ids "
            else:
                cypher_statement += "WHERE size([x IN $ids where toLower(e.id) CONTAINS toLower(x)]) > 0 "
            params["ids"] = ids

        if properties:
            prop_list = []
            for i, prop in enumerate(properties):
                prop_list.append(f"e.`{prop}` = $property_{i}")
                params[f"property_{i}"] = properties[prop]
            cypher_statement += " AND ".join(prop_list)

        return_statement = (
            """
        WITH e
        RETURN e.id AS name,
               [l in labels(e) WHERE l <> '"""
            + BASE_ENTITY_LABEL
            + """' | l][0] AS type,
               e{.* , embedding: Null, id: Null} AS properties
        """
        )
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
        """
        Get the triplets of the entities in the graph.

        Args:
            entity_names (Optional[List[str]], optional): The entity names to find. Defaults to None.
            relation_names (Optional[List[str]], optional): The relation names to follow. Defaults to None.
            properties (Optional[dict], optional): The properties to return. Defaults to None.
            ids (Optional[List[str]], optional): The ids to search on. Defaults to None.

        Returns:
            List[Triplet]: A list of triples

        """
        cypher_statement = f"MATCH (e:`{BASE_ENTITY_LABEL}`) "

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
            MATCH (e)-[r{":`" + "`|`".join(relation_names) + "`" if relation_names else ""}]->(t:{BASE_ENTITY_LABEL})
            RETURN e.name AS source_id, [l in labels(e) WHERE l <> '{BASE_ENTITY_LABEL}' | l][0] AS source_type,
                   e{{.* , embedding: Null, name: Null}} AS source_properties,
                   type(r) AS type,
                   r{{.*}} AS rel_properties,
                   t.name AS target_id, [l in labels(t) WHERE l <> '{BASE_ENTITY_LABEL}' | l][0] AS target_type,
                   t{{.* , embedding: Null, name: Null}} AS target_properties"""
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
                properties=remove_empty_values(record["rel_properties"]),
            )
            triples.append([source, rel, target])
        return triples

    def get_rel_map(
        self,
        graph_nodes: List[LabelledNode],
        depth: int = 2,
        limit: int = 30,
        ignore_rels: List[str] = None,
    ) -> List[Tuple[Any]]:
        """
        Get a depth aware map of relations.

        Args:
            graph_nodes (List[LabelledNode]): The nodes
            depth (int, optional): The depth to traverse. Defaults to 2.
            limit (int, optional): The limit of numbers to return. Defaults to 30.
            ignore_rels (List[str] | None, optional): Relations to ignore. Defaults to None.

        Returns:
            List[Tuple[LabelledNode | Relation]]: The node/relationship pairs

        """
        triples = []

        ids = [node.id for node in graph_nodes]
        # Needs some optimization
        if len(ids) > 0:
            logger.debug(f"get_rel_map() ids: {ids}")
            response = self.structured_query(
                f"""
                WITH $ids AS id_list
                UNWIND range(0, size(id_list) - 1) AS idx
                MATCH (e:`{BASE_ENTITY_LABEL}`)
                WHERE e.id = id_list[idx]
                MATCH p=(e)-[r*1..{depth}]-(other)
                WHERE size([rel in relationships(p) WHERE type(rel) <> 'MENTIONS']) = size(relationships(p))
                UNWIND relationships(p) AS rel
                WITH distinct rel, idx
                WITH startNode(rel) AS source,
                    type(rel) AS type,
                    endNode(rel) AS endNode,
                    idx
                LIMIT {limit}
                RETURN source.id AS source_id, [l in labels(source) WHERE l <> '{BASE_ENTITY_LABEL}' | l][0] AS source_type,
                    source{{.* , embedding: Null, id: Null}} AS source_properties,
                    type,
                    endNode.id AS target_id, [l in labels(endNode) WHERE l <> '{BASE_ENTITY_LABEL}' | l][0] AS target_type,
                    endNode{{.* , embedding: Null, id: Null}} AS target_properties,
                    idx
                ORDER BY idx
                LIMIT {limit}
                """,
                param_map={"ids": ids},
            )
        else:
            response = []
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

    @abstractmethod
    def upsert_nodes(self, nodes: List[LabelledNode]) -> None:
        raise NotImplementedError

    def upsert_relations(self, relations: List[Relation]) -> None:
        """
        Upsert relations in the graph.

        Args:
            relations (List[Relation]): Relations to upsert

        """
        for r in relations:
            self.structured_query(
                """
                WITH $data AS row
                MERGE (source {id: row.source_id})
                ON CREATE SET source:Chunk
                MERGE (target {id: row.target_id})
                ON CREATE SET target:Chunk
                MERGE (source)-[r:`"""
                + r.label
                + """`]->(target)
                SET r+= removeKeyFromMap(row.properties, '')
                RETURN count(*)
                """,
                param_map={"data": r.dict()},
            )

    def delete(
        self,
        entity_names: List[str] = None,
        relation_names: List[str] = None,
        properties: Dict = None,
        ids: List[str] = None,
    ) -> None:
        """
        Delete data matching the criteria.

        Args:
            entity_names (List[str] | None, optional): The entity names to delete. Defaults to None.
            relation_names (List[str] | None, optional): The relation names to delete. Defaults to None.
            properties (Dict | None, optional): The properties to remove. Defaults to None.
            ids (List[str] | None, optional): The ids to remove. Defaults to None.

        """
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

    @abstractmethod
    def structured_query(self, query: str, param_map: Dict[str, Any] = None) -> Any:
        raise NotImplementedError

    def query(self, query: str, params: dict = {}) -> Dict[str, Any]:
        """
        Run the query.

        Args:
            query (str): The query to run
            params (dict, optional): The query parameters. Defaults to {}.

        Returns:
            Dict[str, Any]: The query results

        """
        return self.structured_query(query, params)

    @abstractmethod
    def vector_query(self, query: VectorStoreQuery, **kwargs: Any) -> Tuple[List[Any]]:
        raise NotImplementedError

    @abstractmethod
    def _get_summary(self) -> Dict:
        raise NotImplementedError

    def get_schema(self, refresh: bool = False) -> Any:
        """Get the schema of the graph store."""
        if refresh or not self.schema:
            schema = refresh_schema(self.query, self._get_summary())
            self.schema = schema["schema_str"]
            self.structured_schema = schema["structured_schema"]
        return self.structured_schema

    def get_schema_str(self, refresh: bool = False) -> str:
        """
        Get the schema as a string.

        Args:
            refresh (bool, optional): True to force refresh of the schema. Defaults to False.

        Returns:
            str: A string description of the schema

        """
        if refresh or not self.schema:
            schema = refresh_schema(self.query, self._get_summary())
            self.schema = schema["schema_str"]
            self.structured_schema = schema["structured_schema"]

        return self.schema
