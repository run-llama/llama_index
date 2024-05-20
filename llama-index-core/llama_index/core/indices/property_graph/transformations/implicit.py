from typing import Any, List

from llama_index.core.schema import TransformComponent, BaseNode, NodeRelationship
from llama_index.core.graph_stores.types import Relation, KG_NODES_KEY, KG_RELATIONS_KEY


def get_node_rel_string(relationship: NodeRelationship) -> str:
    return str(relationship).split(".")[-1]


class ImplicitTripletExtractor(TransformComponent):
    def __call__(self, nodes: List[BaseNode], **kwargs: Any) -> List[BaseNode]:
        """Extract edges from node relationships."""
        for node in nodes:
            existing_relations = node.metadata.pop(KG_RELATIONS_KEY, [])
            existing_nodes = node.metadata.pop(KG_NODES_KEY, [])

            edges = []
            metadata = node.metadata.copy()
            if node.source_node:
                edges.append(
                    Relation(
                        target_id=node.source_node.node_id,
                        source_id=node.node_id,
                        label=get_node_rel_string(NodeRelationship.SOURCE),
                        properties=metadata,
                    )
                )

            if node.parent_node:
                edges.append(
                    Relation(
                        target_id=node.parent_node.node_id,
                        source_id=node.node_id,
                        label=get_node_rel_string(NodeRelationship.PARENT),
                        properties=metadata,
                    )
                )

            if node.prev_node:
                edges.append(
                    Relation(
                        target_id=node.prev_node.node_id,
                        source_id=node.node_id,
                        label=get_node_rel_string(NodeRelationship.PREVIOUS),
                        properties=metadata,
                    )
                )

            if node.next_node:
                edges.append(
                    Relation(
                        source_id=node.node_id,
                        target_id=node.next_node.node_id,
                        label=get_node_rel_string(NodeRelationship.NEXT),
                        properties=metadata,
                    )
                )

            if node.child_nodes:
                for child_node in node.child_nodes:
                    edges.append(
                        Relation(
                            source_id=node.node_id,
                            target_id=child_node.node_id,
                            label=get_node_rel_string(NodeRelationship.CHILD),
                            properties=metadata,
                        )
                    )

            # link all existing kg_nodes to the current text chunk
            for kg_node in existing_nodes:
                edges.append(
                    Relation(
                        target_id=node.id_,
                        source_id=kg_node.id,
                        label="SOURCE_CHUNK",
                    )
                )

            existing_relations.extend(edges)
            node.metadata["relations"] = existing_relations
            node.metadata["nodes"] = existing_nodes

        return nodes
