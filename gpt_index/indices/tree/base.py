"""Tree-based index."""

from typing import Any, Optional, Sequence

from gpt_index.indices.base import DOCUMENTS_INPUT, BaseGPTIndex
from gpt_index.indices.common.tree.base import GPTTreeIndexBuilder
from gpt_index.indices.data_structs import IndexGraph
from gpt_index.indices.query.schema import QueryMode
from gpt_index.indices.tree.inserter import GPTIndexInserter
from gpt_index.langchain_helpers.chain_wrapper import LLMPredictor
from gpt_index.prompts.default_prompts import (
    DEFAULT_INSERT_PROMPT,
    DEFAULT_SUMMARY_PROMPT,
)
from gpt_index.prompts.prompts import SummaryPrompt, TreeInsertPrompt
from gpt_index.schema import BaseDocument

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
        documents: Optional[Sequence[DOCUMENTS_INPUT]] = None,
        index_struct: Optional[IndexGraph] = None,
        summary_template: Optional[SummaryPrompt] = None,
        insert_prompt: Optional[TreeInsertPrompt] = None,
        num_children: int = 10,
        llm_predictor: Optional[LLMPredictor] = None,
        build_tree: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        # need to set parameters before building index in base class.
        self.num_children = num_children
        self.summary_template = summary_template or DEFAULT_SUMMARY_PROMPT
        self.insert_prompt: TreeInsertPrompt = insert_prompt or DEFAULT_INSERT_PROMPT
        self.build_tree = build_tree
        super().__init__(
            documents=documents,
            index_struct=index_struct,
            llm_predictor=llm_predictor,
            **kwargs,
        )

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

    def _build_index_from_documents(
        self, documents: Sequence[BaseDocument], verbose: bool = False
    ) -> IndexGraph:
        """Build the index from documents."""
        # do simple concatenation
        index_builder = GPTTreeIndexBuilder(
            self.num_children,
            self.summary_template,
            self._llm_predictor,
            self._prompt_helper,
        )
        index_graph = index_builder.build_from_text(
            documents, build_tree=self.build_tree, verbose=verbose
        )
        return index_graph

    def _insert(self, document: BaseDocument, **insert_kwargs: Any) -> None:
        """Insert a document."""
        # TODO: allow to customize insert prompt
        inserter = GPTIndexInserter(
            self.index_struct,
            num_children=self.num_children,
            insert_prompt=self.insert_prompt,
            summary_prompt=self.summary_template,
            llm_predictor=self._llm_predictor,
            prompt_helper=self._prompt_helper,
        )
        inserter.insert(document)

    def delete(self, document: BaseDocument) -> None:
        """Delete a document."""
        raise NotImplementedError("Delete not implemented for tree index.")
