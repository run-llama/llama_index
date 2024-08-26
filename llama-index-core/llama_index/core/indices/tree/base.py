"""Tree-based index."""

from enum import Enum
from typing import Any, Dict, Optional, Sequence, Union

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.embeddings.base import BaseEmbedding

# from llama_index.core.data_structs.data_structs import IndexGraph
from llama_index.core.data_structs.data_structs import IndexGraph
from llama_index.core.indices.base import BaseIndex
from llama_index.core.indices.common_tree.base import GPTTreeIndexBuilder
from llama_index.core.indices.tree.inserter import TreeIndexInserter
from llama_index.core.llms.llm import LLM
from llama_index.core.prompts import BasePromptTemplate
from llama_index.core.prompts.default_prompts import (
    DEFAULT_INSERT_PROMPT,
    DEFAULT_SUMMARY_PROMPT,
)
from llama_index.core.schema import BaseNode, IndexNode
from llama_index.core.settings import Settings
from llama_index.core.storage.docstore.types import RefDocInfo


class TreeRetrieverMode(str, Enum):
    SELECT_LEAF = "select_leaf"
    SELECT_LEAF_EMBEDDING = "select_leaf_embedding"
    ALL_LEAF = "all_leaf"
    ROOT = "root"


REQUIRE_TREE_MODES = {
    TreeRetrieverMode.SELECT_LEAF,
    TreeRetrieverMode.SELECT_LEAF_EMBEDDING,
    TreeRetrieverMode.ROOT,
}


class TreeIndex(BaseIndex[IndexGraph]):
    """Tree Index.

    The tree index is a tree-structured index, where each node is a summary of
    the children nodes. During index construction, the tree is constructed
    in a bottoms-up fashion until we end up with a set of root_nodes.

    There are a few different options during query time (see :ref:`Ref-Query`).
    The main option is to traverse down the tree from the root nodes.
    A secondary answer is to directly synthesize the answer from the root nodes.

    Args:
        summary_template (Optional[BasePromptTemplate]): A Summarization Prompt
            (see :ref:`Prompt-Templates`).
        insert_prompt (Optional[BasePromptTemplate]): An Tree Insertion Prompt
            (see :ref:`Prompt-Templates`).
        num_children (int): The number of children each node should have.
        build_tree (bool): Whether to build the tree during index construction.
        show_progress (bool): Whether to show progress bars. Defaults to False.

    """

    index_struct_cls = IndexGraph

    def __init__(
        self,
        nodes: Optional[Sequence[BaseNode]] = None,
        objects: Optional[Sequence[IndexNode]] = None,
        index_struct: Optional[IndexGraph] = None,
        llm: Optional[LLM] = None,
        summary_template: Optional[BasePromptTemplate] = None,
        insert_prompt: Optional[BasePromptTemplate] = None,
        num_children: int = 10,
        build_tree: bool = True,
        use_async: bool = False,
        show_progress: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        # need to set parameters before building index in base class.
        self.num_children = num_children
        self.summary_template = summary_template or DEFAULT_SUMMARY_PROMPT
        self.insert_prompt: BasePromptTemplate = insert_prompt or DEFAULT_INSERT_PROMPT
        self.build_tree = build_tree
        self._use_async = use_async
        self._llm = llm or Settings.llm
        super().__init__(
            nodes=nodes,
            index_struct=index_struct,
            show_progress=show_progress,
            objects=objects,
            **kwargs,
        )

    def as_retriever(
        self,
        retriever_mode: Union[str, TreeRetrieverMode] = TreeRetrieverMode.SELECT_LEAF,
        embed_model: Optional[BaseEmbedding] = None,
        **kwargs: Any,
    ) -> BaseRetriever:
        # NOTE: lazy import
        from llama_index.core.indices.tree.all_leaf_retriever import (
            TreeAllLeafRetriever,
        )
        from llama_index.core.indices.tree.select_leaf_embedding_retriever import (
            TreeSelectLeafEmbeddingRetriever,
        )
        from llama_index.core.indices.tree.select_leaf_retriever import (
            TreeSelectLeafRetriever,
        )
        from llama_index.core.indices.tree.tree_root_retriever import (
            TreeRootRetriever,
        )

        self._validate_build_tree_required(TreeRetrieverMode(retriever_mode))

        if retriever_mode == TreeRetrieverMode.SELECT_LEAF:
            return TreeSelectLeafRetriever(self, object_map=self._object_map, **kwargs)
        elif retriever_mode == TreeRetrieverMode.SELECT_LEAF_EMBEDDING:
            embed_model = embed_model or Settings.embed_model
            return TreeSelectLeafEmbeddingRetriever(
                self, embed_model=embed_model, object_map=self._object_map, **kwargs
            )
        elif retriever_mode == TreeRetrieverMode.ROOT:
            return TreeRootRetriever(self, object_map=self._object_map, **kwargs)
        elif retriever_mode == TreeRetrieverMode.ALL_LEAF:
            return TreeAllLeafRetriever(self, object_map=self._object_map, **kwargs)
        else:
            raise ValueError(f"Unknown retriever mode: {retriever_mode}")

    def _validate_build_tree_required(self, retriever_mode: TreeRetrieverMode) -> None:
        """Check if index supports modes that require trees."""
        if retriever_mode in REQUIRE_TREE_MODES and not self.build_tree:
            raise ValueError(
                "Index was constructed without building trees, "
                f"but retriever mode {retriever_mode} requires trees."
            )

    def _build_index_from_nodes(self, nodes: Sequence[BaseNode]) -> IndexGraph:
        """Build the index from nodes."""
        index_builder = GPTTreeIndexBuilder(
            self.num_children,
            self.summary_template,
            llm=self._llm,
            use_async=self._use_async,
            show_progress=self._show_progress,
            docstore=self._docstore,
        )
        return index_builder.build_from_nodes(nodes, build_tree=self.build_tree)

    def _insert(self, nodes: Sequence[BaseNode], **insert_kwargs: Any) -> None:
        """Insert a document."""
        # TODO: allow to customize insert prompt
        inserter = TreeIndexInserter(
            self.index_struct,
            llm=self._llm,
            num_children=self.num_children,
            insert_prompt=self.insert_prompt,
            summary_prompt=self.summary_template,
            docstore=self._docstore,
        )
        inserter.insert(nodes)

    def _delete_node(self, node_id: str, **delete_kwargs: Any) -> None:
        """Delete a node."""
        raise NotImplementedError("Delete not implemented for tree index.")

    @property
    def ref_doc_info(self) -> Dict[str, RefDocInfo]:
        """Retrieve a dict mapping of ingested documents and their nodes+metadata."""
        node_doc_ids = list(self.index_struct.all_nodes.values())
        nodes = self.docstore.get_nodes(node_doc_ids)

        all_ref_doc_info = {}
        for node in nodes:
            ref_node = node.source_node
            if not ref_node:
                continue

            ref_doc_info = self.docstore.get_ref_doc_info(ref_node.node_id)
            if not ref_doc_info:
                continue

            all_ref_doc_info[ref_node.node_id] = ref_doc_info
        return all_ref_doc_info


# legacy
GPTTreeIndex = TreeIndex
