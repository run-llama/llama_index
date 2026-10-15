# Auto Merging Retriever

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, cast

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.indices.query.schema import QueryBundle
from llama_index.core.indices.utils import truncate_text
from llama_index.core.indices.vector_store.retrievers.retriever import (
    VectorIndexRetriever,
)
from llama_index.core.schema import (
    BaseNode,
    IndexNode,
    NodeWithScore,
    MetadataMode,
    QueryBundle,
)
from llama_index.core.storage.storage_context import StorageContext

logger = logging.getLogger(__name__)


class AutoMergingRetriever(BaseRetriever):
    """
    This retriever will try to merge context into parent context.

    The retriever first retrieves chunks from a vector store.
    Then, it will try to merge the chunks into a single context.

    """

    def __init__(
        self,
        vector_retriever: VectorIndexRetriever,
        storage_context: StorageContext,
        simple_ratio_thresh: float = 0.5,
        verbose: bool = False,
        callback_manager: Optional[CallbackManager] = None,
        object_map: Optional[dict] = None,
        objects: Optional[List[IndexNode]] = None,
    ) -> None:
        """Init params."""
        self._vector_retriever = vector_retriever
        self._storage_context = storage_context
        self._simple_ratio_thresh = simple_ratio_thresh
        super().__init__(
            callback_manager=callback_manager,
            object_map=object_map,
            objects=objects,
            verbose=verbose,
        )

    def _get_parents_and_merge(
        self, nodes: List[NodeWithScore]
    ) -> Tuple[List[NodeWithScore], bool]:
        """Get parents and merge nodes."""
        # retrieve all parent nodes
        parent_nodes: Dict[str, BaseNode] = {}
        parent_cur_children_dict: Dict[str, List[NodeWithScore]] = defaultdict(list)
        for node in nodes:
            if node.node.parent_node is None:
                continue
            parent_node_info = node.node.parent_node

            # Fetch actual parent node if doesn't exist in `parent_nodes` cache yet
            parent_node_id = parent_node_info.node_id
            if parent_node_id not in parent_nodes:
                parent_node = self._storage_context.docstore.get_document(
                    parent_node_id
                )
                parent_nodes[parent_node_id] = cast(BaseNode, parent_node)

            # add reference to child from parent
            parent_cur_children_dict[parent_node_id].append(node)

        # compute ratios and "merge" nodes
        # merging: delete some children nodes, add some parent nodes
        node_ids_to_delete = set()
        nodes_to_add: Dict[str, NodeWithScore] = {}
        for parent_node_id, parent_node in parent_nodes.items():
            parent_child_nodes = parent_node.child_nodes
            parent_num_children = len(parent_child_nodes) if parent_child_nodes else 1
            parent_cur_children = parent_cur_children_dict[parent_node_id]
            ratio = len(parent_cur_children) / parent_num_children

            # if ratio is high enough, merge
            if ratio > self._simple_ratio_thresh:
                node_ids_to_delete.update(
                    set({n.node.node_id for n in parent_cur_children})
                )

                parent_node_text = truncate_text(
                    parent_node.get_content(metadata_mode=MetadataMode.NONE), 100
                )
                info_str = (
                    f"> Merging {len(parent_cur_children)} nodes into parent node.\n"
                    f"> Parent node id: {parent_node_id}.\n"
                    f"> Parent node text: {parent_node_text}\n"
                )
                logger.info(info_str)
                if self._verbose:
                    print(info_str)

                # add parent node
                # can try averaging score across embeddings for now

                avg_score = sum(
                    [n.get_score() or 0.0 for n in parent_cur_children]
                ) / len(parent_cur_children)
                parent_node_with_score = NodeWithScore(
                    node=parent_node, score=avg_score
                )
                nodes_to_add[parent_node_id] = parent_node_with_score

        # delete old child nodes, add new parent nodes
        new_nodes = [n for n in nodes if n.node.node_id not in node_ids_to_delete]
        # add parent nodes
        new_nodes.extend(list(nodes_to_add.values()))

        is_changed = len(node_ids_to_delete) > 0

        return new_nodes, is_changed

    def _fill_in_nodes(
        self, nodes: List[NodeWithScore]
    ) -> Tuple[List[NodeWithScore], bool]:
        """Fill in nodes.

        Insert the single node that sits between two retrieved neighbours when it is
        missing from the results: node ``i`` points to it with its NEXT relationship
        and node ``i+1`` points back to it with its PREVIOUS relationship.

        The two references are compared by node id rather than by equality. Node
        parsers build the NEXT and PREVIOUS references at different points of
        processing, so the two snapshots can differ in ``metadata``/``hash`` for the
        same node (for example when the source ``Document`` carries metadata).
        Comparing the whole ``RelatedNodeInfo`` then never matches and the gap is
        silently not filled for real-world documents.
        """
        new_nodes = []
        is_changed = False
        # A node already returned by the retriever must not be inserted a second time
        # just because it also sits in a gap; compare by id and keep the original hit.
        existing_node_ids = {node.node.node_id for node in nodes}
        for idx, node in enumerate(nodes):
            new_nodes.append(node)
            if idx >= len(nodes) - 1:
                continue

            cur_node = cast(BaseNode, node.node)
            next_node_info = cur_node.next_node
            prev_node_info = nodes[idx + 1].node.prev_node
            # if there's exactly one node in the middle, add that to the queue
            if next_node_info is None or prev_node_info is None:
                continue
            if next_node_info.node_id != prev_node_info.node_id:
                continue

            next_node = self._storage_context.docstore.get_document(
                next_node_info.node_id
            )
            next_node = cast(BaseNode, next_node)
            if next_node.node_id in existing_node_ids:
                continue

            existing_node_ids.add(next_node.node_id)
            is_changed = True

            next_node_text = truncate_text(
                next_node.get_content(metadata_mode=MetadataMode.NONE), 100
            )
            info_str = (
                f"> Filling in node. Node id: {next_node.node_id}\n"
                f"> Node text: {next_node_text}\n"
            )
            logger.info(info_str)
            if self._verbose:
                print(info_str)

            # set score to be average of current node and next node
            avg_score = (node.get_score() + nodes[idx + 1].get_score()) / 2
            new_nodes.append(NodeWithScore(node=next_node, score=avg_score))
        return new_nodes, is_changed

    def _try_merging(
        self, nodes: List[NodeWithScore]
    ) -> Tuple[List[NodeWithScore], bool]:
        """Try different ways of merging nodes."""
        # first try filling in nodes
        nodes, is_changed_0 = self._fill_in_nodes(nodes)
        # then try merging nodes
        nodes, is_changed_1 = self._get_parents_and_merge(nodes)
        return nodes, is_changed_0 or is_changed_1

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        Retrieve nodes given a query.

        Implemented by the user.

        """
        initial_nodes = self._vector_retriever.retrieve(query_bundle)

        cur_nodes, is_changed = self._try_merging(initial_nodes)
        # cur_nodes, is_changed = self._get_parents_and_merge(initial_nodes)
        while is_changed:
            cur_nodes, is_changed = self._try_merging(cur_nodes)
            # cur_nodes, is_changed = self._get_parents_and_merge(cur_nodes)

        # sort by similarity
        cur_nodes.sort(key=lambda x: x.get_score(), reverse=True)

        return cur_nodes
