"""Tree-based index."""

from typing import Any, Optional, Sequence

# from gpt_index.data_structs.data_structs import IndexGraph
from gpt_index.data_structs.data_structs_v2 import IndexGraph
from gpt_index.data_structs.node_v2 import Node
from gpt_index.indices.base import BaseGPTIndex, QueryMap
from gpt_index.indices.common_tree.base import GPTTreeIndexBuilder
from gpt_index.indices.query.schema import QueryMode
from gpt_index.indices.tree.embedding_query import GPTTreeIndexEmbeddingQuery
from gpt_index.indices.tree.leaf_query import GPTTreeIndexLeafQuery
from gpt_index.indices.tree.retrieve_query import GPTTreeIndexRetQuery
from gpt_index.indices.tree.summarize_query import GPTTreeIndexSummarizeQuery
from gpt_index.indices.service_context import ServiceContext
from gpt_index.indices.tree.inserter import GPTTreeIndexInserter
from gpt_index.prompts.default_prompts import (
    DEFAULT_INSERT_PROMPT,
    DEFAULT_SUMMARY_PROMPT,
)
from gpt_index.prompts.prompts import SummaryPrompt, TreeInsertPrompt

REQUIRE_TREE_MODES = {
    QueryMode.DEFAULT,
    QueryMode.EMBEDDING,
    QueryMode.RETRIEVE,
}


class GPTTreeIndex(BaseGPTIndex[IndexGraph]):
    """GPT Tree Index.

    The tree index is a tree-structured index, where each node is a summary of
    the children nodes. During index construction, the tree is constructed
    in a bottoms-up fashion until we end up with a set of root_nodes.

    There are a few different options during query time (see :ref:`Ref-Query`).
    The main option is to traverse down the tree from the root nodes.
    A secondary answer is to directly synthesize the answer from the root nodes.

    Args:
        summary_template (Optional[SummaryPrompt]): A Summarization Prompt
            (see :ref:`Prompt-Templates`).
        insert_prompt (Optional[TreeInsertPrompt]): An Tree Insertion Prompt
            (see :ref:`Prompt-Templates`).
        num_children (int): The number of children each node should have.
        build_tree (bool): Whether to build the tree during index construction.

    """

    index_struct_cls = IndexGraph

    def __init__(
        self,
        nodes: Optional[Sequence[Node]] = None,
        index_struct: Optional[IndexGraph] = None,
        service_context: Optional[ServiceContext] = None,
        summary_template: Optional[SummaryPrompt] = None,
        insert_prompt: Optional[TreeInsertPrompt] = None,
        num_children: int = 10,
        build_tree: bool = True,
        use_async: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        # need to set parameters before building index in base class.
        self.num_children = num_children
        self.summary_template = summary_template or DEFAULT_SUMMARY_PROMPT
        self.insert_prompt: TreeInsertPrompt = insert_prompt or DEFAULT_INSERT_PROMPT
        self.build_tree = build_tree
        self._use_async = use_async
        super().__init__(
            nodes=nodes,
            index_struct=index_struct,
            service_context=service_context,
            **kwargs,
        )

    @classmethod
    def get_query_map(self) -> QueryMap:
        """Get query map."""
        return {
            QueryMode.DEFAULT: GPTTreeIndexLeafQuery,
            QueryMode.EMBEDDING: GPTTreeIndexEmbeddingQuery,
            QueryMode.RETRIEVE: GPTTreeIndexRetQuery,
            QueryMode.SUMMARIZE: GPTTreeIndexSummarizeQuery,
        }

    def _validate_build_tree_required(self, mode: QueryMode) -> None:
        """Check if index supports modes that require trees."""
        if mode in REQUIRE_TREE_MODES and not self.build_tree:
            raise ValueError(
                "Index was constructed without building trees, "
                f"but mode {mode} requires trees."
            )

    def _preprocess_query(self, mode: QueryMode, query_kwargs: Any) -> None:
        """Query mode to class."""
        super()._preprocess_query(mode, query_kwargs)
        self._validate_build_tree_required(mode)

    def _build_index_from_nodes(self, nodes: Sequence[Node]) -> IndexGraph:
        """Build the index from nodes."""
        index_builder = GPTTreeIndexBuilder(
            self.num_children,
            self.summary_template,
            service_context=self._service_context,
            use_async=self._use_async,
            docstore=self._docstore,
        )
        index_graph = index_builder.build_from_nodes(nodes, build_tree=self.build_tree)
        return index_graph

    def _insert(self, nodes: Sequence[Node], **insert_kwargs: Any) -> None:
        """Insert a document."""
        # TODO: allow to customize insert prompt
        inserter = GPTTreeIndexInserter(
            self.index_struct,
            num_children=self.num_children,
            insert_prompt=self.insert_prompt,
            summary_prompt=self.summary_template,
            service_context=self._service_context,
            docstore=self._docstore,
        )
        inserter.insert(nodes)

    def _delete(self, doc_id: str, **delete_kwargs: Any) -> None:
        """Delete a document."""
        raise NotImplementedError("Delete not implemented for tree index.")
