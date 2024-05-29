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

DDL = Template(
    """
CREATE TAG IF NOT EXISTS `Entity__` (`name` STRING);
CREATE TAG IF NOT EXISTS `Chunk__` (`text` STRING);
CREATE TAG IF NOT EXISTS `Node__` (`label` STRING);
CREATE TAG IF NOT EXISTS `Props__` ({{props_schema}});
CREATE EDGE IF NOT EXISTS `Relation__` (`label` STRING{% if props_schema != "" %}, {{props_schema}}{% endif%});
"""
)


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
        props_schema: str = "",
        refresh_schema: bool = True,
        sanitize_query_output: bool = False,  # We don't put Embedding-Like values as Properties
        enhanced_schema: bool = False,
    ) -> None:
        self.sanitize_query_output = sanitize_query_output
        self.enhcnaced_schema = enhanced_schema

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
        if overwrite:
            self._client.execute(f"CLEAR SPACE {self._space};")

        self.structured_schema = {}
        if refresh_schema:
            self.refresh_schema()

        self.supports_structured_queries = True

    @property
    def client(self):
        """client of NebulaGraph."""
        return self._client

    def _execute(self, query: str) -> ResultSet:
        return self._client.execute(query)

    def refresh_schema(self) -> None:
        """
        Example data of self.structured_schema:
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
        for tag in self._execute("SHOW TAGS").column_values("Name"):
            tag_name = tag.cast()
            tags_schema[tag_name] = []

            r = self._execute(f"DESCRIBE TAG `{tag_name}`")
            props, types, comments = (
                r.column_values("Field"),
                r.column_values("Type"),
                r.column_values("Comment"),
            )
            for i in range(r.row_size()):
                prop_name = props[i].cast()
                prop_type = types[i].cast()
                prop_comment = comments[i].cast()
                prop = {"property": prop_name, "type": prop_type}
                if prop_comment:
                    prop["comment"] = prop_comment
                tags_schema[tag_name].append(prop)

        # TODO: Add the sample data for tags(good for Text2cypher)

        for edge_type in self._execute("SHOW EDGES").column_values("Name"):
            edge_type_name = edge_type.cast()
            edge_types_schema[edge_type_name] = []
            r = self._execute(f"DESCRIBE EDGE `{edge_type_name}`")
            props, types, comments = (
                r.column_values("Field"),
                r.column_values("Type"),
                r.column_values("Comment"),
            )
            for i in range(r.row_size()):
                prop_name = props[i].cast()
                prop_type = types[i].cast()
                prop_comment = comments[i].cast()
                prop = {"property": prop_name, "type": prop_type}
                if prop_comment:
                    prop["comment"] = prop_comment
                edge_types_schema[edge_type_name].append(prop)

            # build relationships types
            # TODO: handle dangling vertices in a strategy to Sample more edges with valid vertices
            # For now, we ignore the tag of dangling vertices by leaving it empty.
            rel_query_sample_edge = f"""
MATCH ()-[e:`{ edge_type_name }`]->()
RETURN [src(e), dst(e)] AS sample_edge LIMIT 1
"""
            sample_edge = self._execute(rel_query_sample_edge).column_values(
                "sample_edge"
            )
            if len(sample_edge) == 0:
                continue
            src_id, dst_id = sample_edge[0].cast()
            quote = "" if self.vid_type == "INT64" else QUOTE
            _rel_query_edge_type = f"""
MATCH (m)-[:{ edge_type_name }]->(n)
  WHERE id(m) == { quote }{ src_id }{ quote } AND id(n) == { quote }{ dst_id }{ quote }
RETURN "(" + (CASE WHEN size(tags(m)) == 0 THEN "" ELSE ":" + tags(m)[0] END) + ")-[:{ edge_type_name }]->(" + (CASE WHEN size(tags(n)) == 0 THEN "" ELSE ":" + tags(n)[0] END) + ")" AS rels
"""

            r = self._execute(_rel_query_edge_type).column_values("rels")
            if len(r) > 0:
                relationships.append(r[0].cast())

        self.structured_schema = {
            "node_props": tags_schema,
            "rel_props": edge_types_schema,
            "relationships": relationships,
            # TODO: need to check necessarity of meta data here
        }

        # Update node info
        # TODO need to look into this enhanced schema
        # for node in schema_counts[0].get("nodes", []):
        #     node_props = self.structured_schema["node_props"].get(node["name"])
        #     if not node_props:  # The node has no properties
        #         continue
        #     enhanced_cypher = self._enhanced_schema_cypher(
        #         node["name"], node_props, node["count"] < EXHAUSTIVE_SEARCH_LIMIT
        #     )
        #     enhanced_info = self.structured_query(enhanced_cypher)[0]["output"]
        #     for prop in node_props:
        #         if prop["property"] in enhanced_info:
        #             prop.update(enhanced_info[prop["property"]])
        # # Update rel info
        # for rel in schema_counts[0].get("relationships", []):

        #     rel_props = self.structured_schema["rel_props"].get(rel["name"])
        #     if not rel_props:  # The rel has no properties
        #         continue
        #     enhanced_cypher = self._enhanced_schema_cypher(
        #         rel["name"],
        #         rel_props,
        #         rel["count"] < EXHAUSTIVE_SEARCH_LIMIT,
        #         is_relationship=True,
        #     )

        #     enhanced_info = self.structured_query(enhanced_cypher)[0]["output"]
        #     for prop in rel_props:
        #         if prop["property"] in enhanced_info:
        #             prop.update(enhanced_info[prop["property"]])

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
            # NebulaGraph is Schema-ful, so we need to be strong schema mindset to abstract this.
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
        # This could be revisted, if we don't have any properties for labels, mapping labels to
        # Properties of tag: Entity__ is also feasible.
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
            self.structured_query(stmt)

    def _construct_property_query(self, properties: Dict[str, Any]):
        keys = ",".join([f"`{k}`" for k in properties.keys()])
        values_k = ""
        values_params: Dict[Any] = {}
        for idx, v in enumerate(properties.values()):
            values_k += f"$kv_{idx},"
            values_params[f"kv_{idx}"] = v
        values_k = values_k[:-1]
        return keys, values_k, values_params

    def upsert_relations(self, relations: List[Relation]) -> None:
        """Add relations."""
        # TODO: Handle ad-hoc schema ensuring, now assuming all relations are present in the schema
        for relation in relations:
            keys, values_k, values_params = self._construct_property_query(
                relation.properties
            )
            stmt = f'INSERT EDGE `Relation__` (`label`,{keys}) VALUES "{relation.source_id}"->"{relation.target_id}":("{relation.label}",{values_k});'
            self.structured_query(stmt, param_map=values_params)

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
            cypher_statement += "id(e) in $ids "
            params["ids"] = ids

        if properties:
            prop_list = []
            for i, prop in enumerate(properties):
                prop_list.append(f"e.`{prop}` = $property_{i}")
                params[f"property_{i}"] = properties[prop]
            cypher_statement += " AND ".join(prop_list)

        return_statement = """
        WITH e
        RETURN id(e) AS name,
               [l in labels(e) WHERE l <> 'Entity__' | l][0] AS type,
               properties(e) AS properties
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
        cypher_statement = "MATCH (e:`Entity__`) "

        params = {}
        if entity_names or properties or ids:
            cypher_statement += "WHERE "

        if entity_names:
            raise NotImplementedError("Filtering by entity names is not supported yet.")
        if relation_names:
            raise NotImplementedError(
                "Filtering by relation names is not supported yet."
            )

        if ids:
            cypher_statement += "id(e) in $ids "
            params["ids"] = ids

        if properties:
            prop_list = []
            for i, prop in enumerate(properties):
                prop_list.append(f"e.`{prop}` = $property_{i}")
                params[f"property_{i}"] = properties[prop]
            cypher_statement += " AND ".join(prop_list)

        return_statement = f"""
        MATCH (e)-[r:Relation__]->(t)
        RETURN id(e) AS source_id, e.Node__.label AS source_type,
                properties(e.Props__) AS source_properties,
                r.label AS type,
                properties(r) AS rel_properties,
                id(t) AS target_id, t.Node__.label AS target_type,
                properties(t.Props__) AS target_properties
        """
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
            rel_properties = remove_empty_values(record["rel_properties"])
            rel_properties.pop("label")
            rel = Relation(
                source_id=record["source_id"],
                target_id=record["target_id"],
                label=record["type"],
                properties=rel_properties,
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
            MATCH (e:`Entity__`)
            WHERE id(e) in $ids
            MATCH p=(e)-[r*1..{depth}]-(other)
            WHERE ALL(rel in relationships(p) WHERE type(rel) <> 'MENTIONS')
            UNWIND relationships(p) AS rel
            WITH distinct rel
            WITH startNode(rel) AS source,
                type(rel) AS type,
                endNode(rel) AS endNode
            RETURN id(source) AS source_id, [l in labels(source) WHERE l <> 'Entity__' | l][0] AS source_type,
                    properties(source) AS source_properties,
                    type,
                    id(endNode) AS target_id, [l in labels(endNode) WHERE l <> 'Entity__' | l][0] AS target_type,
                    properties(endNode) AS target_properties
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
            raise ValueError(f"Query failed: {result.error_msg()}")
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
        if entity_names:
            self.structured_query(
                "LOOKUP ON `Entity__` WHERE `Entity__`.name IN $entity_names | DELETE VERTEX $-.VertexID",
                param_map={"entity_names": entity_names},
            )

        for id in ids or []:
            self.structured_query(f'DELETE VERTEX "{id}" WITH EDGE;')
        if relation_names:
            for relation_name in relation_names:
                self.structured_query(
                    f"LOOKUP ON `Relation__` WHERE `Relation__`.`label` IN $relation_name | DELETE EDGE $-.EdgeID",
                    param_map={"relation_name": relation_name},
                )
        if properties:
            cypher = "MATCH (e:`Entity__`) WHERE "
            prop_list = []
            params = {}
            for i, prop in enumerate(properties):
                prop_list.append(f"e.`Entity__`.`{prop}` == $property_{i}")
                params[f"property_{i}"] = properties[prop]
            cypher += " AND ".join(prop_list)
            cypher += " RETURN id(e) AS id"
            ids_dict_list = self.structured_query(
                cypher,
                param_map=params,
            )
            ids = [el["id"] for el in ids_dict_list]
            for id in ids or []:
                self.structured_query(f'DELETE VERTEX "{id}" WITH EDGE;')

    def _enhanced_schema_cypher(
        self,
        label_or_type: str,
        properties: List[Dict[str, Any]],
        exhaustive: bool,
        is_relationship: bool = False,
    ) -> str:
        """Get enhanced schema information."""
        pass

    def get_schema(self, refresh: bool = False) -> Any:
        if refresh:
            self.refresh_schema()

        return self.structured_schema

    def get_schema_str(self, refresh: bool = False) -> str:
        schema = self.get_schema(refresh=refresh)

        formatted_node_props = []
        formatted_rel_props = []

        if self.enhcnaced_schema:
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
                                    f'{[clean_string_values(el) for el in prop["values"]]}'
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
                            example = f'Min: {prop["min"]}, Max: {prop["max"]}'
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
                                    f'{[clean_string_values(el) for el in prop["values"]]}'
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
                            example = f'Min: {prop["min"]}, Max:  {prop["max"]}'
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
                        example = f'Min Size: {prop["min_size"]}, Max Size: {prop["max_size"]}'
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
    ) -> Tuple[List[LabelledNode] | List[float]]:
        raise NotImplementedError(
            "Vector query not implemented for NebulaPropertyGraphStore."
        )
