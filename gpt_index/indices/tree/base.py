"""Tree-based index."""

from typing import Any, Dict, Optional, Sequence

from gpt_index.indices.base import (
    DEFAULT_MODE,
    DOCUMENTS_INPUT,
    EMBEDDING_MODE,
    BaseGPTIndex,
)
from gpt_index.indices.data_structs import IndexGraph, Node
from gpt_index.indices.prompt_helper import PromptHelper
from gpt_index.indices.query.base import BaseGPTIndexQuery
from gpt_index.indices.query.tree.embedding_query import GPTTreeIndexEmbeddingQuery
from gpt_index.indices.query.tree.leaf_query import GPTTreeIndexLeafQuery
from gpt_index.indices.query.tree.retrieve_query import GPTTreeIndexRetQuery
from gpt_index.indices.tree.inserter import GPTIndexInserter
from gpt_index.indices.utils import get_sorted_node_list
from gpt_index.langchain_helpers.chain_wrapper import LLMPredictor
from gpt_index.prompts.base import Prompt, validate_prompt
from gpt_index.prompts.default_prompts import (
    DEFAULT_INSERT_PROMPT,
    DEFAULT_SUMMARY_PROMPT,
)
from gpt_index.schema import BaseDocument
from gpt_index.utils import llm_token_counter

RETRIEVE_MODE = "retrieve"


class GPTTreeIndexBuilder:
    """GPT tree index builder.

    Helper class to build the tree-structured index.

    """

    def __init__(
        self,
        num_children: int,
        summary_prompt: Prompt,
        llm_predictor: Optional[LLMPredictor],
        prompt_helper: Optional[PromptHelper],
    ) -> None:
        """Initialize with params."""
        if num_children < 2:
            raise ValueError("Invalid number of children.")
        self.num_children = num_children
        self.summary_prompt = summary_prompt
        self._prompt_helper = prompt_helper or PromptHelper()
        self._text_splitter = self._prompt_helper.get_text_splitter_given_prompt(
            self.summary_prompt, self.num_children
        )
        self._llm_predictor = llm_predictor or LLMPredictor()

    def _get_nodes_from_document(
        self, start_idx: int, document: BaseDocument
    ) -> Dict[int, Node]:
        """Add document to index."""
        text_chunks = self._text_splitter.split_text(document.get_text())
        doc_nodes = {
            (start_idx + i): Node(
                text=t, index=(start_idx + i), ref_doc_id=document.get_doc_id()
            )
            for i, t in enumerate(text_chunks)
        }
        return doc_nodes

    def build_from_text(self, documents: Sequence[BaseDocument]) -> IndexGraph:
        """Build from text.

        Returns:
            IndexGraph: graph object consisting of all_nodes, root_nodes

        """
        all_nodes: Dict[int, Node] = {}
        for d in documents:
            all_nodes.update(self._get_nodes_from_document(len(all_nodes), d))

        # instantiate all_nodes from initial text chunks
        root_nodes = self._build_index_from_nodes(all_nodes, all_nodes)
        return IndexGraph(all_nodes=all_nodes, root_nodes=root_nodes)

    def _build_index_from_nodes(
        self, cur_nodes: Dict[int, Node], all_nodes: Dict[int, Node]
    ) -> Dict[int, Node]:
        """Consolidates chunks recursively, in a bottoms-up fashion."""
        cur_node_list = get_sorted_node_list(cur_nodes)
        cur_index = len(all_nodes)
        new_node_dict = {}
        print(
            f"> Building index from nodes: {len(cur_nodes) // self.num_children} chunks"
        )
        for i in range(0, len(cur_node_list), self.num_children):
            print(f"{i}/{len(cur_nodes)}")
            cur_nodes_chunk = cur_node_list[i : i + self.num_children]
            text_chunk = self._prompt_helper.get_text_from_nodes(
                cur_nodes_chunk, prompt=self.summary_prompt
            )

            new_summary, _ = self._llm_predictor.predict(
                self.summary_prompt, text=text_chunk
            )

            print(f"> {i}/{len(cur_nodes)}, summary: {new_summary}")
            new_node = Node(
                text=new_summary,
                index=cur_index,
                child_indices={n.index for n in cur_nodes_chunk},
            )
            new_node_dict[cur_index] = new_node
            cur_index += 1

        all_nodes.update(new_node_dict)

        if len(new_node_dict) <= self.num_children:
            return new_node_dict
        else:
            return self._build_index_from_nodes(new_node_dict, all_nodes)


class GPTTreeIndex(BaseGPTIndex[IndexGraph]):
    """GPT Tree Index.

    The tree index is a tree-structured index, where each node is a summary of
    the children nodes. During index construction, the tree is constructed
    in a bottoms-up fashion until we end up with a set of root_nodes.

    There are a few different options during query time (see :ref:`Ref-Query`).
    The main option is to traverse down the tree from the root nodes.
    A secondary answer is to directly synthesize the answer from the root nodes.

    Args:
        summary_template (Optional[Prompt]): A Summarization Prompt
            (see :ref:`Prompt-Templates`).
        insert_prompt (Optional[Prompt]): An Tree Insertion Prompt
            (see :ref:`Prompt-Templates`).

    """

    index_struct_cls = IndexGraph

    def __init__(
        self,
        documents: Optional[Sequence[DOCUMENTS_INPUT]] = None,
        index_struct: Optional[IndexGraph] = None,
        summary_template: Optional[Prompt] = None,
        insert_prompt: Optional[Prompt] = None,
        query_str: Optional[str] = None,
        num_children: int = 10,
        llm_predictor: Optional[LLMPredictor] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        # need to set parameters before building index in base class.
        self.num_children = num_children
        summary_template = summary_template or DEFAULT_SUMMARY_PROMPT
        # if query_str is specified, then we try to load into summary template
        if query_str is not None:
            summary_template = summary_template.partial_format(query_str=query_str)

        self.summary_template = summary_template
        self.insert_prompt = insert_prompt or DEFAULT_INSERT_PROMPT
        validate_prompt(self.summary_template, ["text"], ["query_str"])
        super().__init__(
            documents=documents,
            index_struct=index_struct,
            llm_predictor=llm_predictor,
            **kwargs,
        )

    def _mode_to_query(self, mode: str, **query_kwargs: Any) -> BaseGPTIndexQuery:
        """Query mode to class."""
        if mode == DEFAULT_MODE:
            query: BaseGPTIndexQuery = GPTTreeIndexLeafQuery(
                self.index_struct, **query_kwargs
            )
        elif mode == RETRIEVE_MODE:
            query = GPTTreeIndexRetQuery(self.index_struct, **query_kwargs)
        elif mode == EMBEDDING_MODE:
            query = GPTTreeIndexEmbeddingQuery(self.index_struct, **query_kwargs)
        else:
            raise ValueError(f"Invalid query mode: {mode}.")
        return query

    @llm_token_counter("build_index_from_documents")
    def build_index_from_documents(
        self, documents: Sequence[BaseDocument]
    ) -> IndexGraph:
        """Build the index from documents."""
        # do simple concatenation
        index_builder = GPTTreeIndexBuilder(
            self.num_children,
            self.summary_template,
            self._llm_predictor,
            self._prompt_helper,
        )
        index_graph = index_builder.build_from_text(documents)
        return index_graph

    def _insert(self, document: BaseDocument, **insert_kwargs: Any) -> None:
        """Insert a document."""
        # TODO: allow to customize insert prompt
        inserter = GPTIndexInserter(
            self.index_struct,
            num_children=self.num_children,
            summary_prompt=self.summary_template,
            insert_prompt=self.insert_prompt,
            prompt_helper=self._prompt_helper,
        )
        inserter.insert(document)

    def delete(self, document: BaseDocument) -> None:
        """Delete a document."""
        raise NotImplementedError("Delete not implemented for tree index.")
