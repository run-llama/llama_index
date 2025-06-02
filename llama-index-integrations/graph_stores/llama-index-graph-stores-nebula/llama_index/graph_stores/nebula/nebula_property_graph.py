from typing import Any, List, Dict, Optional, Tuple

from llama_index.core.graph_stores.types import (
    PropertyGraphStore,
    Triplet,
    LabelledNode,
    Relation,
    EntityNode,
    ChunkNode,
)
from llama_index.core.graph_stores.utils import (
    clean_string_values,
    value_sanitize,
    LIST_LIMIT,
)
from llama_index.core.vector_stores.types import VectorStoreQuery
from nebula3.gclient.net.SessionPool import SessionPool
from nebula3.gclient.net.base import BaseExecutor
from nebula3.data.ResultSet import ResultSet
from jinja2 import Template

from llama_index.graph_stores.nebula.utils import (
    build_param_map,
    remove_empty_values,
    url_scheme_parse,
)


QUOTE = '"'
EXHAUSTIVE_SEARCH_LIMIT = 10000
# Threshold for returning all available prop values in graph schema
DISTINCT_VALUE_LIMIT = 10

META_NODE_LABEL_PREFIX = "__meta__label__"  # Not yet used, could be applied later

# DDL Design
# - Entity__ is used to store the extracted entity name(readable id)
# - Chunk__ is used to store the extracted chunk text
# - Node__ is used to store the node label
# - Props__ is used to store the LlamaIndex Node Metadata
# - Relation__ is used to store the relation
# - __meta__node_label__ is used to store the node labels,
#     we use EDGE for this due to we could leverage dangling edges to make it mostly invisible
# - __meta__rel_label__ is used to store the relation labels
#     the starting and ending vertices will be human-readable from node_label && META_NODE_LABEL_PREFIX
#     due to they are dangling edges, so they are mostly invisible


DDL = Template(
    """
CREATE TAG IF NOT EXISTS `Entity__` (`name` STRING);
CREATE TAG IF NOT EXISTS `Chunk__` (`text` STRING);
CREATE TAG IF NOT EXISTS `Node__` (`label` STRING);
CREATE TAG IF NOT EXISTS `Props__` ({{props_schema}});
CREATE EDGE IF NOT EXISTS `Relation__` (`label` STRING{% if props_schema != "" %}, {{props_schema}}{% endif%});

CREATE EDGE IF NOT EXISTS `__meta__node_label__` (`label` STRING, `props_json` STRING);
CREATE EDGE IF NOT EXISTS `__meta__rel_label__` (`label` STRING, `props_json` STRING);
"""
)

# TODO: need to define Props__ Indexes based on all the properties
INDEX_DDL = """
CREATE TAG INDEX IF NOT EXISTS idx_Entity__ ON `Entity__`(`name`(256));
CREATE TAG INDEX IF NOT EXISTS idx_Chunk__ ON `Chunk__`(`text`(256));
CREATE TAG INDEX IF NOT EXISTS idx_Node__ ON `Node__`(`label`(256));
CREATE EDGE INDEX IF NOT EXISTS idx_Relation__ ON `Relation__`(`label`(256));

CREATE EDGE INDEX IF NOT EXISTS idx_meta__node_label__ ON `__meta__node_label__`(`label`(256));
CREATE EDGE INDEX IF NOT EXISTS idx_meta__rel_label__ ON `__meta__rel_label__`(`label`(256));
"""

# Hard coded default schema, which is union of
# document metadata: `file_path` STRING, `file_name` STRING, `file_type` STRING, `file_size` INT, `creation_date` STRING, `last_modified_date` STRING
# llamaindex_node: `_node_content` STRING, `_node_type` STRING, `document_id` STRING, `doc_id` STRING, `ref_doc_id` STRING
# introduced by PropertyGraph: `triplet_source_id` STRING
DEFAULT_PROPS_SCHEMA = "`file_path` STRING, `file_name` STRING, `file_type` STRING, `file_size` INT, `creation_date` STRING, `last_modified_date` STRING, `_node_content` STRING, `_node_type` STRING, `document_id` STRING, `doc_id` STRING, `ref_doc_id` STRING, `triplet_source_id` STRING"


class NebulaPropertyGraphStore(PropertyGraphStore):
    """
    NebulaGraph Property Graph Store.

    This class implements a NebulaGraph property graph store.

    You could go with NebulaGraph-lite freely on Google Colab.
    - https://github.com/nebula-contrib/nebulagraph-lite
    Or Install with Docker Extension(search in the Docker Extension marketplace) on your local machine.

    Examples:
        `pip install llama-index-graph-stores-nebula`
        `pip install jupyter-nebulagraph`

        Create a new NebulaGraph Space with Basic Schema:

        ```jupyter
        %load_ext ngql
        %ngql --address 127.0.0.1 --port 9669 --user root --password nebula
        %ngql CREATE SPACE IF NOT EXISTS llamaindex_nebula_property_graph(vid_type=FIXED_STRING(256));
        ```

    """

    _space: str
    _client: BaseExecutor
    sanitize_query_output: bool
    enhanced_schema: bool

    def __init__(
        self,
        space: str,
        client: Optional[BaseExecutor] = None,
        username: str = "root",
        password: str = "nebula",
        url: str = "nebula://localhost:9669",
        overwrite: bool = False,
        props_schema: str = DEFAULT_PROPS_SCHEMA,
        refresh_schema: bool = True,
        sanitize_query_output: bool = False,  # We don't put Embedding-Like values as Properties
        enhanced_schema: bool = False,
    ) -> None:
        self.sanitize_query_output = sanitize_query_output
        self.enhanced_schema = enhanced_schema

        self._space = space
        if client:
            self._client = client
        else:
            session_pool = SessionPool(
                username,
                password,
                self._space,
                [url_scheme_parse(url)],
            )
            session_pool.init()
            self._client = session_pool
        self._client.execute(DDL.render(props_schema=props_schema))
        self._client.execute(INDEX_DDL)
        if overwrite:
            self._client.execute(f"CLEAR SPACE {self._space};")

        self.structured_schema = {}
        if refresh_schema:
            try:
                self.refresh_schema()
            except Exception:
                # fails to refresh for the first time
                pass

        self.supports_structured_queries = True

    @property
    def client(self):
        """Client of NebulaGraph."""
        return self._client

    def _execute(self, query: str) -> ResultSet:
        return self._client.execute(query)

    def refresh_schema(self) -> None:
        """
        Refresh schema.

        Example data of self.structured_schema
        {
            "node_props": {
                "Person": [
                    {"property": "name", "type": "STRING", "comment": "The name of the person"},
                    {"property": "age", "type": "INTEGER", "comment": "The age of the person"},
                    {"property": "dob", "type": "DATE", "comment": "The date of birth of the person"}
                ],
                "Company": [
                    {"property": "name", "type": "STRING", "comment": "The name of the company"},
                    {"property": "founded", "type": "DATE", "comment": "The date of foundation of the company"}
                ]
            },
            "rel_props": {
                "WORKS_AT": [
                    {"property": "since", "type": "DATE", "comment": "The date when the person started working at the company"}
                ],
                "MANAGES": [
                    {"property": "since", "type": "DATE", "comment": "The date when the person started managing the company"}
                ]
            },
            "relationships": [
                {"start": "Person", "type": "WORKS_AT", "end": "Company"},
                {"start": "Person", "type": "MANAGES", "end": "Company"}
            ]
        }
        """
        tags_schema = {}
        edge_types_schema = {}
        relationships = []

        for node_label in self.structured_query(
            "MATCH ()-[node_label:`__meta__node_label__`]->() "
            "RETURN node_label.label AS name, "
            "JSON_EXTRACT(node_label.props_json) AS props"
        ):
            tags_schema[node_label["name"]] = []
            # TODO: add properties to tags_schema

        for rel_label in self.structured_query(
            "MATCH ()-[rel_label:`__meta__rel_label__`]->() "
            "RETURN rel_label.label AS name, "
            "src(rel_label) AS src, dst(rel_label) AS dst, "
            "JSON_EXTRACT(rel_label.props_json) AS props"
        ):
            edge_types_schema[rel_label["name"]] = []
            # TODO: add properties to edge_types_schema
            relationships.append(
                {
                    "start": rel_label["src"],
                    "type": rel_label["name"],
                    "end": rel_label["dst"],
                }
            )

        self.structured_schema = {
            "node_props": tags_schema,
            "rel_props": edge_types_schema,
            "relationships": relationships,
            # TODO: need to check necessarity of meta data here
        }

    def upsert_nodes(self, nodes: List[LabelledNode]) -> None:
        # meta tag Entity__ is used to store the entity name
        # meta tag Chunk__ is used to store the chunk text
        # other labels are used to store the entity properties
        # which must be created before upserting the nodes

        # Lists to hold separated types
        entity_list: List[EntityNode] = []
        chunk_list: List[ChunkNode] = []
        other_list: List[LabelledNode] = []

        # Sort by type
        for item in nodes:
            if isinstance(item, EntityNode):
                entity_list.append(item)
            elif isinstance(item, ChunkNode):
                chunk_list.append(item)
            else:
                other_list.append(item)

        if chunk_list:
            # TODO: need to double check other properties if any(it seems for now only text is there)
            # model chunk as tag and perform upsert
            # i.e. INSERT VERTEX `Chunk__` (`text`) VALUES "foo":("hello world"), "baz":("lorem ipsum");
            insert_query = "INSERT VERTEX `Chunk__` (`text`) VALUES "
            for i, chunk in enumerate(chunk_list):
                insert_query += f'"{chunk.id}":($chunk_{i}),'
            insert_query = insert_query[:-1]  # Remove trailing comma
            self.structured_query(
                insert_query,
                param_map={
                    f"chunk_{i}": chunk.text for i, chunk in enumerate(chunk_list)
                },
            )

        if entity_list:
            # model with tag Entity__ and other tags(label) if applicable
            # need to add properties as well, for extractors like SchemaLLMPathExtractor there is no properties
            # NebulaGraph is Schema-Full, so we need to be strong schema mindset to abstract this.
            # i.e.
            # INSERT VERTEX Entity__ (name) VALUES "foo":("bar"), "baz":("qux");
            # INSERT VERTEX Person (name) VALUES "foo":("bar"), "baz":("qux");

            # The meta tag Entity__ is used to store the entity name
            insert_query = "INSERT VERTEX `Entity__` (`name`) VALUES "
            for i, entity in enumerate(entity_list):
                insert_query += f'"{entity.id}":($entity_{i}),'
            insert_query = insert_query[:-1]  # Remove trailing comma
            self.structured_query(
                insert_query,
                param_map={
                    f"entity_{i}": entity.name for i, entity in enumerate(entity_list)
                },
            )

        # Create tags for each LabelledNode
        # This could be revisited, if we don't have any properties for labels, mapping labels to
        # Properties of tag: Entity__ is also feasible.
        schema_ensurence_cache = set()
        for i, entity in enumerate(nodes):
            keys, values_k, values_params = self._construct_property_query(
                entity.properties
            )
            stmt = f'INSERT VERTEX Props__ ({keys}) VALUES "{entity.id}":({values_k});'
            self.structured_query(
                stmt,
                param_map=values_params,
            )
            stmt = (
                f'INSERT VERTEX Node__ (label) VALUES "{entity.id}":("{entity.label}");'
            )
            # if entity.label not in schema_ensurence_cache:
            #     if ensure_node_meta_schema(
            #         entity.label, self.structured_schema, self.client, entity.properties
            #     ):
            #         self.refresh_schema()
            #         schema_ensurence_cache.add(entity.label)
            self.structured_query(stmt)

    def _construct_property_query(self, properties: Dict[str, Any]):
        keys = ",".join([f"`{k}`" for k in properties])
        values_k = ""
        values_params: Dict[Any] = {}
        for idx, v in enumerate(properties.values()):
            values_k += f"$kv_{idx},"
            values_params[f"kv_{idx}"] = v
        values_k = values_k[:-1]
        return keys, values_k, values_params

    def upsert_relations(self, relations: List[Relation]) -> None:
        """Add relations."""
        schema_ensurence_cache = set()
        for relation in relations:
            keys, values_k, values_params = self._construct_property_query(
                relation.properties
            )
            stmt = f'INSERT EDGE `Relation__` (`label`,{keys}) VALUES "{relation.source_id}"->"{relation.target_id}":("{relation.label}",{values_k});'
            # if relation.label not in schema_ensurence_cache:
            #     if ensure_relation_meta_schema(
            #         relation.source_id,
            #         relation.target_id,
            #         relation.label,
            #         self.structured_schema,
            #         self.client,
            #         relation.properties,
            #     ):
            #         self.refresh_schema()
            #         schema_ensurence_cache.add(relation.label)
            self.structured_query(stmt, param_map=values_params)

    def get(
        self,
        properties: Optional[dict] = None,
        ids: Optional[List[str]] = None,
    ) -> List[LabelledNode]:
        """Get nodes."""
        if not (properties or ids):
            return []
        else:
            return self._get(properties, ids)

    def _get(
        self,
        properties: Optional[dict] = None,
        ids: Optional[List[str]] = None,
    ) -> List[LabelledNode]:
        """Get nodes."""
        cypher_statement = "MATCH (e:Node__) "
        if properties or ids:
            cypher_statement += "WHERE "
        params = {}

        if ids:
            cypher_statement += f"id(e) in $all_id "
            params[f"all_id"] = ids
        if properties:
            for i, prop in enumerate(properties):
                cypher_statement += f"e.Props__.`{prop}` == $property_{i} AND "
                params[f"property_{i}"] = properties[prop]
            cypher_statement = cypher_statement[:-5]  # Remove trailing AND

        return_statement = """
        RETURN id(e) AS name,
               e.Node__.label AS type,
               properties(e.Props__) AS properties,
               properties(e) AS all_props
        """
        cypher_statement += return_statement
        cypher_statement = cypher_statement.replace("\n", " ")

        response = self.structured_query(cypher_statement, param_map=params)

        nodes = []
        for record in response:
            if "text" in record["all_props"]:
                node = ChunkNode(
                    id_=record["name"],
                    label=record["type"],
                    text=record["all_props"]["text"],
                    properties=remove_empty_values(record["properties"]),
                )
            elif "name" in record["all_props"]:
                node = EntityNode(
                    id_=record["name"],
                    label=record["type"],
                    name=record["all_props"]["name"],
                    properties=remove_empty_values(record["properties"]),
                )
            else:
                node = EntityNode(
                    name=record["name"],
                    type=record["type"],
                    properties=remove_empty_values(record["properties"]),
                )
            nodes.append(node)
        return nodes

    def get_all_nodes(self) -> List[LabelledNode]:
        return self._get()

    def get_triplets(
        self,
        entity_names: Optional[List[str]] = None,
        relation_names: Optional[List[str]] = None,
        properties: Optional[dict] = None,
        ids: Optional[List[str]] = None,
    ) -> List[Triplet]:
        cypher_statement = "MATCH (e:`Entity__`)-[r:`Relation__`]->(t:`Entity__`) "
        if not (entity_names or relation_names or properties or ids):
            return []
        else:
            cypher_statement += "WHERE "
        params = {}

        if entity_names:
            cypher_statement += (
                f"e.Entity__.name in $entities OR t.Entity__.name in $entities"
            )
            params[f"entities"] = entity_names
        if relation_names:
            cypher_statement += f"r.label in $relations "
            params[f"relations"] = relation_names
        if properties:
            pass
        if ids:
            cypher_statement += f"id(e) in $all_id OR id(t) in $all_id"
            params[f"all_id"] = ids
        if properties:
            v0_matching = ""
            v1_matching = ""
            edge_matching = ""
            for i, prop in enumerate(properties):
                v0_matching += f"e.Props__.`{prop}` == $property_{i} AND "
                v1_matching += f"t.Props__.`{prop}` == $property_{i} AND "
                edge_matching += f"r.`{prop}` == $property_{i} AND "
                params[f"property_{i}"] = properties[prop]
            v0_matching = v0_matching[:-5]  # Remove trailing AND
            v1_matching = v1_matching[:-5]  # Remove trailing AND
            edge_matching = edge_matching[:-5]  # Remove trailing AND
            cypher_statement += (
                f"({v0_matching}) OR ({edge_matching}) OR ({v1_matching})"
            )

        return_statement = f"""
        RETURN id(e) AS source_id, e.Node__.label AS source_type,
                properties(e.Props__) AS source_properties,
                r.label AS type,
                properties(r) AS rel_properties,
                id(t) AS target_id, t.Node__.label AS target_type,
                properties(t.Props__) AS target_properties
        """
        cypher_statement += return_statement
        cypher_statement = cypher_statement.replace("\n", " ")

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
            rel_properties = remove_empty_values(record["rel_properties"])
            rel_properties.pop("label")
            rel = Relation(
                source_id=record["source_id"],
                target_id=record["target_id"],
                label=record["type"],
                properties=rel_properties,
            )
            triples.append((source, rel, target))
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
            MATCH (e:`Entity__`)
            WHERE id(e) in $ids
            MATCH p=(e)-[r*1..{depth}]-(other)
            WHERE ALL(rel in relationships(p) WHERE rel.`label` <> 'MENTIONS')
            UNWIND relationships(p) AS rel
            WITH distinct rel
            WITH startNode(rel) AS source,
                rel.`label` AS type,
                endNode(rel) AS endNode
            MATCH (v) WHERE id(v)==id(source) WITH v AS source, type, endNode
            MATCH (v) WHERE id(v)==id(endNode) WITH source, type, v AS endNode
            RETURN id(source) AS source_id, source.`Node__`.`label` AS source_type,
                    properties(source.`Props__`) AS source_properties,
                    type,
                    id(endNode) AS target_id, endNode.`Node__`.`label` AS target_type,
                    properties(endNode.`Props__`) AS target_properties
            LIMIT {limit}
            """,
            param_map={"ids": ids},
        )

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
        if not param_map:
            result = self._client.execute(query)
        else:
            result = self._client.execute_parameter(query, build_param_map(param_map))
        if not result.is_succeeded():
            raise Exception(
                "NebulaGraph query failed:",
                result.error_msg(),
                "Statement:",
                query,
                "Params:",
                param_map,
            )
        full_result = [
            {
                key: result.row_values(row_index)[i].cast_primitive()
                for i, key in enumerate(result.keys())
            }
            for row_index in range(result.row_size())
        ]
        if self.sanitize_query_output:
            # Not applicable for NebulaGraph for now though
            return value_sanitize(full_result)

        return full_result

    def delete(
        self,
        entity_names: Optional[List[str]] = None,
        relation_names: Optional[List[str]] = None,
        properties: Optional[dict] = None,
        ids: Optional[List[str]] = None,
    ) -> None:
        """Delete matching data."""
        ans_ids: List[str] = []
        if entity_names:
            trips = self.get_triplets(
                entity_names=entity_names,
            )
            for trip in trips:
                if isinstance(trip[0], EntityNode) and trip[0].name in entity_names:
                    ans_ids.append(trip[0].id)
                if isinstance(trip[2], EntityNode) and trip[2].name in entity_names:
                    ans_ids.append(trip[2].id)
        if relation_names:
            trips = self.get_triplets(
                relation_names=relation_names,
            )
            for trip in trips:
                ans_ids += [trip[0].id, trip[2].id, trip[1].source_id]
        if properties:
            nodes = self.get(properties=properties)
            ans_ids += [node.id for node in nodes]
        if ids:
            nodes = self.get(ids=ids)
            ans_ids += [node.id for node in nodes]
        ans_ids = list(set(ans_ids))
        for id in ans_ids or []:
            self.structured_query(f'DELETE VERTEX "{id}" WITH EDGE;')

    def _enhanced_schema_cypher(
        self,
        label_or_type: str,
        properties: List[Dict[str, Any]],
        exhaustive: bool,
        is_relationship: bool = False,
    ) -> str:
        """Get enhanced schema information."""

    def get_schema(self, refresh: bool = False) -> Any:
        if refresh:
            self.refresh_schema()

        return self.structured_schema

    def get_schema_str(self, refresh: bool = False) -> str:
        schema = self.get_schema(refresh=refresh)

        formatted_node_props = []
        formatted_rel_props = []

        if self.enhanced_schema:
            # Enhanced formatting for nodes
            for node_type, properties in schema["node_props"].items():
                formatted_node_props.append(f"- **{node_type}**")
                for prop in properties:
                    example = ""
                    if prop["type"] == "string" and prop.get("values"):
                        if prop.get("distinct_count", 11) > DISTINCT_VALUE_LIMIT:
                            example = (
                                f'Example: "{clean_string_values(prop["values"][0])}"'
                                if prop["values"]
                                else ""
                            )
                        else:  # If less than 10 possible values return all
                            example = (
                                (
                                    "Available options: "
                                    f"{[clean_string_values(el) for el in prop['values']]}"
                                )
                                if prop["values"]
                                else ""
                            )

                    elif prop["type"] in [
                        # TODO: Add all numeric types
                        "int64",
                        "int32",
                        "int16",
                        "int8",
                        "uint64",
                        "uint32",
                        "uint16",
                        "uint8",
                        "date",
                        "datetime",
                        "timestamp",
                        "float",
                        "double",
                    ]:
                        if prop.get("min") is not None:
                            example = f"Min: {prop['min']}, Max: {prop['max']}"
                        else:
                            example = (
                                f'Example: "{prop["values"][0]}"'
                                if prop.get("values")
                                else ""
                            )
                    formatted_node_props.append(
                        f"  - `{prop['property']}`: {prop['type']} {example}"
                    )

            # Enhanced formatting for relationships
            for rel_type, properties in schema["rel_props"].items():
                formatted_rel_props.append(f"- **{rel_type}**")
                for prop in properties:
                    example = ""
                    if prop["type"] == "string":
                        if prop.get("distinct_count", 11) > DISTINCT_VALUE_LIMIT:
                            example = (
                                f'Example: "{clean_string_values(prop["values"][0])}"'
                                if prop.get("values")
                                else ""
                            )
                        else:  # If less than 10 possible values return all
                            example = (
                                (
                                    "Available options: "
                                    f"{[clean_string_values(el) for el in prop['values']]}"
                                )
                                if prop.get("values")
                                else ""
                            )
                    elif prop["type"] in [
                        "int",
                        "int64",
                        "int32",
                        "int16",
                        "int8",
                        "uint64",
                        "uint32",
                        "uint16",
                        "uint8",
                        "float",
                        "double",
                        "date",
                        "datetime",
                        "timestamp",
                    ]:
                        if prop.get("min"):  # If we have min/max
                            example = f"Min: {prop['min']}, Max:  {prop['max']}"
                        else:  # return a single value
                            example = (
                                f'Example: "{prop["values"][0]}"'
                                if prop.get("values")
                                else ""
                            )
                    elif prop["type"] == "LIST":
                        # Skip embeddings
                        if prop["min_size"] > LIST_LIMIT:
                            continue
                        example = f"Min Size: {prop['min_size']}, Max Size: {prop['max_size']}"
                    formatted_rel_props.append(
                        f"  - `{prop['property']}: {prop['type']}` {example}"
                    )
        else:
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

    def vector_query(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> Tuple[List[LabelledNode], List[float]]:
        raise NotImplementedError(
            "Vector query not implemented for NebulaPropertyGraphStore."
        )
