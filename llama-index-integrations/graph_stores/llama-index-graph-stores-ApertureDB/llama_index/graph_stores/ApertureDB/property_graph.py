import logging
from typing import Any, List, Optional
from llama_index.core.graph_stores.types import (
    LabelledNode,
    PropertyGraphStore,
    EntityNode,
    Relation,
    ChunkNode,
    Triplet,
)
from aperturedb.CommonLibrary import create_connector, execute_query
from aperturedb.Query import QueryBuilder

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
    result, response, _ = execute_query(
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
        if k not in properties and not k.startswith("_"):
            to_delete.append(k)
    return to_update, to_delete


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
        self._client = create_connector()

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
                QueryBuilder.find_command(
                    oclass=s.label,
                    params={
                        "_ref": 1,
                        "constraints": {UNIQUEID_PROPERTY: ["==", s.id]},
                        "results": {"all_properties": True, "limit": limit},
                    },
                )
            ]
            for i in range(1, 2):
                query.extend([
                    {
                        "FindEntity": {
                            "_ref": i+1,
                            "is_connected_to": {
                                "ref": i,
                                "direction": "out"
                            },
                            "results": {"all_properties": True, "limit": limit},
                        }
                    },
                    {
                        "FindConnection": {
                            "src": i,
                            "results": {"all_properties": True, "limit": limit},
                        }
                    }
                ])
            result, response, _ = execute_query(
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
                    rel_map.extend(self.get_rel_map(adjacent_nodes, depth-1))
        return rel_map

    def delete(self, entity_names=None, relation_names=None, properties=None, ids=None):
        raise NotImplementedError("delete is not implemented")

    def get(self, properties=None, ids=None):
        constraints = {}
        entities = []
        if ids and len(ids) > 0:
            constraints.setdefault("any", {UNIQUEID_PROPERTY: ["in", ids]})
            query = [{"FindEntity": {"results": {"all_properties": True}}}]
            if len(constraints) > 0:
                query[0]["FindEntity"]["constraints"] = constraints
            result, response, _ = execute_query(
                self._client,
                query,
            )
            assert result == 0, response
            entities.extend(response[0]["FindEntity"].get("entities", []))

        elif properties and len(properties) > 0:
            for k, v in properties.items():
                constraints.setdefault("all", {k: ["==", v]})
            query = [{"FindEntity": {"results": {"all_properties": True}}}]
            if len(constraints) > 0:
                query[0]["FindEntity"]["constraints"] = constraints
            result, response, _ = execute_query(
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

    def structured_query(self, query, param_map=None):
        raise NotImplementedError("structured_query is not implemented")

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
                result, response, _ = execute_query(self._client, query, blobs)
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
                        },
                        "if_not_found": {
                            UNIQUEID_PROPERTY: ["==", f"{r.id}"],
                        },
                    }
                },
            ]
            result, response, _ = execute_query(
                self._client, query, success_statuses=[0, 2]
            )
            assert result == 0, response
            ids.append(r.id)
        return ids

    def vector_query(self, query, **kwargs):
        raise NotImplementedError("vector_query is not implemented")
