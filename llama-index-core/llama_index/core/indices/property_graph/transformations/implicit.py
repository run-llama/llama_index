from typing import Any, List

from llama_index.core.schema import TransformComponent, BaseNode, NodeRelationship


class ImplicitTripletExtractor(TransformComponent):
    def __call__(self, nodes: List[BaseNode], **kwargs: Any) -> List[BaseNode]:
        """Extract triplets from node relationships."""
        for node in nodes:
            triplets = []

            if node.source_node:
                triplets.append(
                    (node.id_, str(NodeRelationship.SOURCE), node.source_node.node_id)
                )

            if node.parent_node:
                triplets.append(
                    (node.id_, str(NodeRelationship.PARENT), node.parent_node.node_id)
                )

            if node.prev_node:
                triplets.append(
                    (node.id_, str(NodeRelationship.PREVIOUS), node.prev_node.node_id)
                )

            if node.next_node:
                triplets.append(
                    (node.id_, str(NodeRelationship.NEXT), node.next_node.node_id)
                )

            if node.child_nodes:
                for child_node in node.child_nodes:
                    triplets.append(
                        (node.id_, str(NodeRelationship.CHILD), child_node.node_id)
                    )

            existing_triplets = node.metadata.get("triplets", [])
            existing_triplets.extend(triplets)
            node.metadata["triplets"] = existing_triplets

        return nodes
