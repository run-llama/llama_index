"""Hierarchical node parser."""

from typing import Dict, List, Optional, Sequence

from llama_index.bridge.pydantic import Field
from llama_index.callbacks.base import CallbackManager
from llama_index.callbacks.schema import CBEventType, EventPayload
from llama_index.node_parser.extractors.metadata_extractors import MetadataExtractor
from llama_index.node_parser.interface import NodeParser
from llama_index.node_parser.node_utils import get_nodes_from_document
from llama_index.schema import BaseNode, Document, NodeRelationship
from llama_index.text_splitter import TextSplitter, get_default_text_splitter
from llama_index.utils import get_tqdm_iterable


def _add_parent_child_relationship(parent_node: BaseNode, child_node: BaseNode) -> None:
    """Add parent/child relationship between nodes."""
    child_list = parent_node.relationships.get(NodeRelationship.CHILD, [])
    child_list.append(child_node.as_related_node_info())
    parent_node.relationships[NodeRelationship.CHILD] = child_list

    child_node.relationships[
        NodeRelationship.PARENT
    ] = parent_node.as_related_node_info()


def get_leaf_nodes(nodes: List[BaseNode]) -> List[BaseNode]:
    """Get leaf nodes."""
    leaf_nodes = []
    for node in nodes:
        if NodeRelationship.CHILD not in node.relationships:
            leaf_nodes.append(node)
    return leaf_nodes


def get_root_nodes(nodes: List[BaseNode]) -> List[BaseNode]:
    """Get root nodes."""
    root_nodes = []
    for node in nodes:
        if NodeRelationship.PARENT not in node.relationships:
            root_nodes.append(node)
    return root_nodes


class HierarchicalNodeParser(NodeParser):
    """Hierarchical node parser.

    Splits a document into a recursive hierarchy Nodes using a TextSplitter.

    NOTE: this will return a hierarchy of nodes in a flat list, where there will be
    overlap between parent nodes (e.g. with a bigger chunk size), and child nodes
    per parent (e.g. with a smaller chunk size).

    For instance, this may return a list of nodes like:
    - list of top-level nodes with chunk size 2048
    - list of second-level nodes, where each node is a child of a top-level node,
        chunk size 512
    - list of third-level nodes, where each node is a child of a second-level node,
        chunk size 128

    Args:
        text_splitter (Optional[TextSplitter]): text splitter
        include_metadata (bool): whether to include metadata in nodes
        include_prev_next_rel (bool): whether to include prev/next relationships

    """

    chunk_sizes: Optional[List[int]] = Field(
        default=None,
        description=(
            "The chunk sizes to use when splitting documents, in order of level."
        ),
    )
    text_splitter_ids: List[str] = Field(
        default_factory=list,
        description=(
            "List of ids for the text splitters to use when splitting documents, "
            + "in order of level (first id used for first level, etc.)."
        ),
    )
    text_splitter_map: Dict[str, TextSplitter] = Field(
        description="Map of text splitter id to text splitter.",
    )
    include_metadata: bool = Field(
        default=True, description="Whether or not to consider metadata when splitting."
    )
    include_prev_next_rel: bool = Field(
        default=True, description="Include prev/next node relationships."
    )
    metadata_extractor: Optional[MetadataExtractor] = Field(
        default=None, description="Metadata extraction pipeline to apply to nodes."
    )
    callback_manager: CallbackManager = Field(
        default_factory=CallbackManager, exclude=True
    )

    @classmethod
    def from_defaults(
        cls,
        chunk_sizes: Optional[List[int]] = None,
        text_splitter_ids: Optional[List[str]] = None,
        text_splitter_map: Optional[Dict[str, TextSplitter]] = None,
        include_metadata: bool = True,
        include_prev_next_rel: bool = True,
        callback_manager: Optional[CallbackManager] = None,
        metadata_extractor: Optional[MetadataExtractor] = None,
    ) -> "HierarchicalNodeParser":
        callback_manager = callback_manager or CallbackManager([])

        if text_splitter_ids is None:
            if chunk_sizes is None:
                chunk_sizes = [2048, 512, 128]

            text_splitter_ids = [
                f"chunk_size_{chunk_size}" for chunk_size in chunk_sizes
            ]
            text_splitter_map = {}
            for chunk_size, text_splitter_id in zip(chunk_sizes, text_splitter_ids):
                text_splitter_map[text_splitter_id] = get_default_text_splitter(
                    chunk_size=chunk_size,
                    callback_manager=callback_manager,
                )
        else:
            if chunk_sizes is not None:
                raise ValueError(
                    "Cannot specify both text_splitter_ids and chunk_sizes."
                )
            if text_splitter_map is None:
                raise ValueError(
                    "Must specify text_splitter_map if using text_splitter_ids."
                )

        return cls(
            chunk_sizes=chunk_sizes,
            text_splitter_ids=text_splitter_ids,
            text_splitter_map=text_splitter_map,
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel,
            callback_manager=callback_manager,
            metadata_extractor=metadata_extractor,
        )

    @classmethod
    def class_name(cls) -> str:
        return "HierarchicalNodeParser"

    def _recursively_get_nodes_from_nodes(
        self,
        nodes: List[BaseNode],
        level: int,
        show_progress: bool = False,
    ) -> List[BaseNode]:
        """Recursively get nodes from nodes."""
        if level >= len(self.text_splitter_ids):
            raise ValueError(
                f"Level {level} is greater than number of text "
                f"splitters ({len(self.text_splitter_ids)})."
            )

        # first split current nodes into sub-nodes
        nodes_with_progress = get_tqdm_iterable(
            nodes, show_progress, "Parsing documents into nodes"
        )
        sub_nodes = []
        for node in nodes_with_progress:
            cur_sub_nodes = get_nodes_from_document(
                node,
                self.text_splitter_map[self.text_splitter_ids[level]],
                self.include_metadata,
                include_prev_next_rel=self.include_prev_next_rel,
            )
            # add parent relationship from sub node to parent node
            # add child relationship from parent node to sub node
            # NOTE: Only add relationships if level > 0, since we don't want to add
            # relationships for the top-level document objects that we are splitting
            if level > 0:
                for sub_node in cur_sub_nodes:
                    _add_parent_child_relationship(
                        parent_node=node,
                        child_node=sub_node,
                    )

            sub_nodes.extend(cur_sub_nodes)

        # now for each sub-node, recursively split into sub-sub-nodes, and add
        if level < len(self.text_splitter_ids) - 1:
            sub_sub_nodes = self._recursively_get_nodes_from_nodes(
                sub_nodes,
                level + 1,
                show_progress=show_progress,
            )
        else:
            sub_sub_nodes = []

        return sub_nodes + sub_sub_nodes

    def get_nodes_from_documents(
        self,
        documents: Sequence[Document],
        show_progress: bool = False,
    ) -> List[BaseNode]:
        """Parse document into nodes.

        Args:
            documents (Sequence[Document]): documents to parse
            include_metadata (bool): whether to include metadata in nodes

        """
        with self.callback_manager.event(
            CBEventType.NODE_PARSING, payload={EventPayload.DOCUMENTS: documents}
        ) as event:
            all_nodes: List[BaseNode] = []
            documents_with_progress = get_tqdm_iterable(
                documents, show_progress, "Parsing documents into nodes"
            )

            # TODO: a bit of a hack rn for tqdm
            for doc in documents_with_progress:
                nodes_from_doc = self._recursively_get_nodes_from_nodes([doc], 0)
                all_nodes.extend(nodes_from_doc)

            if self.metadata_extractor is not None:
                all_nodes = self.metadata_extractor.process_nodes(all_nodes)

            event.on_end(payload={EventPayload.NODES: all_nodes})

        return all_nodes
