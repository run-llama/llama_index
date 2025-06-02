import logging
from typing import Any, Dict, List, Optional
from llama_index.core.graph_stores.types import (
    LabelledNode,
    PropertyGraphStore,
    EntityNode,
    Relation,
    ChunkNode,
    Triplet,
)

query_executor = None
query_builder = None
# Prefix for properties that are in the client metadata
PROPERTY_PREFIX = "lm_"

TEXT_PROPERTY = "text"  # Property name for the text
UNIQUEID_PROPERTY = "uniqueid"  # Property name for the unique id

BATCHSIZE = 1000

logger = logging.getLogger(__name__)


def get_entity(client, label, id):
    """Get entity by id."""
    query = [
        {
            "FindEntity": {
                "_ref": 1,
                "constraints": {UNIQUEID_PROPERTY: ["==", id]},
                "results": {"all_properties": True},
            }
        }
    ]
    if label is not None:
        query[0]["FindEntity"]["with_class"] = label
    result, response, _ = query_executor(
        client,
        query,
    )
    assert result == 0, response
    if (
        "entities" in response[0]["FindEntity"]
        and len(response[0]["FindEntity"]["entities"]) > 0
    ):
        return response[0]["FindEntity"]["entities"][0]
    return None


def changed(entity, properties):
    """Check if properties have changed."""
    to_update = {}
    to_delete = []
    if entity is None:
        return properties, to_delete
    for k, v in properties.items():
        if k not in entity:
            to_update[k] = v
        elif entity[k] != v:
            to_update[k] = v
    for k, v in entity.items():
        if k not in properties and not k.startswith("_") and k not in ["id", "name"]:
            to_delete.append(k)
    return to_update, to_delete


def query_for_ids(command: str, ids: List[str]) -> List[dict]:
    """Create a query for a list of ids."""
    constraints = {}
    constraints.setdefault("any", {UNIQUEID_PROPERTY: ["in", ids]})
    if command == "FindEntity":
        query = [{command: {"results": {"all_properties": True}}}]
    else:
        query = [{command: {}}]
    if len(constraints) > 0:
        query[0][command]["constraints"] = constraints
    return query


def query_for_properties(command: str, properties: dict) -> List[dict]:
    """Create a query for a list of properties."""
    constraints = {}
    for k, v in properties.items():
        constraints.setdefault("all", {k: ["==", v]})
    query = [{command: {"results": {"all_properties": True}}}]
    if len(constraints) > 0:
        query[0][command]["constraints"] = constraints
    return query


class ApertureDBGraphStore(PropertyGraphStore):
    """
    ApertureDB graph store.

    Args:
        config (dict): Configuration for the graph store.
        **kwargs: Additional keyword arguments.

    """

    flat_metadata: bool = True

    @property
    def client(self) -> Any:
        """Get client."""
        return self._client

    def __init__(self, *args, **kwargs) -> None:
        try:
            from aperturedb.CommonLibrary import create_connector, execute_query
            from aperturedb.Query import QueryBuilder
        except ImportError:
            raise ImportError(
                "ApertureDB is not installed. Please install it using "
                "'pip install --upgrade aperturedb'"
            )

        self._client = create_connector()
        global query_executor
        query_executor = execute_query
        global query_builder
        query_builder = QueryBuilder

    def get_rel_map(
        self,
        subjs: List[LabelledNode],
        depth: int = 2,
        limit: int = 30,
        ignore_rels: Optional[List[str]] = None,
    ) -> List[Triplet]:
        """Get depth-aware rel map."""
        if subjs is None or len(subjs) == 0:
            return []
        if depth <= 0:
            return []
        rel_map = []
        ignore_rels = ignore_rels or []
        for s in subjs:
            query = [
                query_builder.find_command(
                    oclass=s.label,
                    params={
                        "_ref": 1,
                        "constraints": {UNIQUEID_PROPERTY: ["==", s.id]},
                        "results": {"all_properties": True, "limit": limit},
                    },
                )
            ]
            for i in range(1, 2):
                query.extend(
                    [
                        {
                            "FindEntity": {
                                "_ref": i + 1,
                                "is_connected_to": {"ref": i, "direction": "out"},
                                "results": {"all_properties": True, "limit": limit},
                            }
                        },
                        {
                            "FindConnection": {
                                "src": i,
                                "results": {"all_properties": True, "limit": limit},
                            }
                        },
                    ]
                )
            result, response, _ = query_executor(
                self._client,
                query,
            )
            assert result == 0, response

            adjacent_nodes = []
            if "entities" in response[0]["FindEntity"]:
                for entity in response[0]["FindEntity"]["entities"]:
                    for c, ce in zip(
                        response[1]["FindEntity"]["entities"],
                        response[2]["FindConnection"]["connections"],
                    ):
                        if ce[UNIQUEID_PROPERTY] in ignore_rels:
                            continue
                        source = EntityNode(
                            name=entity[UNIQUEID_PROPERTY],
                            label=entity["label"],
                            properties=entity,
                        )

                        target = EntityNode(
                            name=c[UNIQUEID_PROPERTY],
                            label=c["label"],
                            properties=c,
                        )

                        relation = Relation(
                            source_id=c[UNIQUEID_PROPERTY],
                            target_id=c[UNIQUEID_PROPERTY],
                            label=ce[UNIQUEID_PROPERTY],
                        )
                        adjacent_nodes.append(target)
                        rel_map.append([source, relation, target])
                    rel_map.extend(self.get_rel_map(adjacent_nodes, depth - 1))
        return rel_map

    def delete(
        self,
        entity_names: Optional[List[str]] = None,
        relation_names: Optional[List[str]] = None,
        properties: Optional[dict] = None,
        ids: Optional[List[str]] = None,
    ) -> None:
        """Delete nodes."""
        if ids and len(ids) > 0:
            query = query_for_ids("DeleteEntity", [id.capitalize() for id in ids])
            result, response, _ = query_executor(
                self._client,
                query,
            )
            assert result == 0, response
        if properties and len(properties) > 0:
            query = query_for_properties("DeleteEntity", properties)
            result, response, _ = query_executor(
                self._client,
                query,
            )
            assert result == 0, response
        if entity_names and len(entity_names) > 0:
            for name in entity_names:
                query = [
                    {
                        "DeleteEntity": {
                            "with_class": name,
                            "constraints": {"_uniqueid": ["!=", "0.0.0"]},
                        }
                    }
                ]
                result, response, _ = query_executor(
                    self._client,
                    query,
                )
                assert result == 0, response
        if relation_names and len(relation_names) > 0:
            for relation_name in set(relation_names):
                query = [
                    {
                        "DeleteConnection": {
                            "with_class": relation_name,
                            "constraints": {"_uniqueid": ["!=", "0.0.0"]},
                        }
                    }
                ]
                result, response, _ = query_executor(
                    self._client,
                    query,
                )
                assert result == 0, response

    def get(
        self, properties: Optional[dict] = None, ids: Optional[List[str]] = None
    ) -> List[LabelledNode]:
        entities = []
        if ids and len(ids) > 0:
            query = query_for_ids("FindEntity", ids)
            result, response, _ = query_executor(
                self._client,
                query,
            )
            assert result == 0, response
            entities.extend(response[0]["FindEntity"].get("entities", []))

        elif properties and len(properties) > 0:
            query = query_for_properties("FindEntity", properties)
            result, response, _ = query_executor(
                self._client,
                query,
            )
            assert result == 0, response
            entities.extend(response[0]["FindEntity"].get("entities", []))

        else:
            query = [
                {
                    "FindEntity": {
                        "results": {"all_properties": True, "limit": BATCHSIZE}
                    }
                }
            ]
            result, response, _ = query_executor(
                self._client,
                query,
            )
            assert result == 0, response
            entities.extend(response[0]["FindEntity"].get("entities", []))

        response = []
        if len(entities) > 0:
            for e in entities:
                if e["label"] == "text_chunk":
                    node = ChunkNode(
                        properties={
                            "_node_content": e["node_content"],
                            "_node_type": e["node_type"],
                        },
                        text=e["text"],
                        id=e[UNIQUEID_PROPERTY],
                    )
                else:
                    node = EntityNode(
                        label=e["label"], properties=e, name=e[UNIQUEID_PROPERTY]
                    )
                response.append(node)

        return response

    def get_triplets(
        self, entity_names=None, relation_names=None, properties=None, ids=None
    ):
        raise NotImplementedError("get_triplets is not implemented")

    def structured_query(
        self, query: str, param_map: Optional[Dict[str, Any]] = None
    ) -> Any:
        query = [{query: param_map}]
        blobs = []
        result, response, _ = query_executor(self._client, query, blobs)
        assert result == 0, response
        return response

    def upsert_nodes(self, nodes: List[EntityNode]) -> List[str]:
        ids = []
        data = []

        for node in nodes:
            # TODO: nodes can be of type EntityNode or ChunkNode
            properties = node.properties
            id = node.id.capitalize()
            if isinstance(node, ChunkNode):
                sane_props = {
                    "text": node.text,
                }
                for k, v in node.properties.items():
                    if k.startswith("_"):
                        sane_props[k[1:]] = v
                properties = sane_props

            entity = get_entity(self._client, node.label, id)
            combined_properties = properties | {
                UNIQUEID_PROPERTY: id,
                "label": node.label,
            }

            command = None
            if entity is None:
                command = {
                    "AddEntity": {
                        "class": node.label,
                        "if_not_found": {UNIQUEID_PROPERTY: ["==", id]},
                        "properties": combined_properties,
                    }
                }
            else:
                to_update, to_delete = changed(entity, combined_properties)
                if len(to_update) > 0 or len(to_delete) > 0:
                    command = {
                        "UpdateEntity": {
                            "constraints": {UNIQUEID_PROPERTY: ["==", id]},
                            "properties": to_update,
                            "remove_props": to_delete,
                        }
                    }

            if command is not None:
                query = [command]
                blobs = []
                result, response, _ = query_executor(self._client, query, blobs)
                assert result == 0, response
                data.append((query, blobs))
                ids.append(id)

        return ids

    def upsert_relations(self, relations: List[Relation]) -> None:
        """Upsert relations."""
        ids = []
        for i, r in enumerate(relations):
            query = [
                {
                    "FindEntity": {
                        "constraints": {
                            UNIQUEID_PROPERTY: ["==", r.source_id.capitalize()]
                        },
                        "_ref": 1,
                    }
                },
                {
                    "FindEntity": {
                        "constraints": {
                            UNIQUEID_PROPERTY: ["==", r.target_id.capitalize()]
                        },
                        "_ref": 2,
                    }
                },
                {
                    "AddConnection": {
                        "class": r.label,
                        "src": 1,
                        "dst": 2,
                        "properties": r.properties
                        | {
                            UNIQUEID_PROPERTY: f"{r.id}",
                            "src_id": r.source_id.capitalize(),
                            "dst_id": r.target_id.capitalize(),
                        },
                        "if_not_found": {
                            UNIQUEID_PROPERTY: ["==", f"{r.id}"],
                            "src_id": ["==", r.source_id.capitalize()],
                            "dst_id": ["==", r.target_id.capitalize()],
                        },
                    }
                },
            ]
            result, response, _ = query_executor(
                self._client, query, success_statuses=[0, 2]
            )
            assert result == 0, response
            ids.append(r.id)
        return ids

    def vector_query(self, query, **kwargs):
        raise NotImplementedError("vector_query is not implemented")
