"""Tree-based index."""

from typing import Any, Dict, Optional, Sequence

from gpt_index.constants import MAX_CHUNK_OVERLAP, MAX_CHUNK_SIZE, NUM_OUTPUTS
from gpt_index.embeddings.openai import EMBED_MAX_TOKEN_LIMIT
from gpt_index.indices.base import (
    DEFAULT_MODE,
    DOCUMENTS_INPUT,
    EMBEDDING_MODE,
    BaseGPTIndex,
)
from gpt_index.indices.data_structs import IndexGraph, Node
from gpt_index.indices.query.base import BaseGPTIndexQuery
from gpt_index.indices.tree.embedding_query import GPTTreeIndexEmbeddingQuery
from gpt_index.indices.tree.inserter import GPTIndexInserter
from gpt_index.indices.tree.leaf_query import GPTTreeIndexLeafQuery
from gpt_index.indices.tree.retrieve_query import GPTTreeIndexRetQuery
from gpt_index.indices.utils import (
    get_chunk_size_given_prompt,
    get_sorted_node_list,
    get_text_from_nodes,
)
from gpt_index.langchain_helpers.chain_wrapper import LLMPredictor
from gpt_index.langchain_helpers.text_splitter import TokenTextSplitter
from gpt_index.prompts.base import Prompt, validate_prompt
from gpt_index.prompts.default_prompts import (
    DEFAULT_INSERT_PROMPT,
    DEFAULT_SUMMARY_PROMPT,
)
from gpt_index.schema import BaseDocument

RETRIEVE_MODE = "retrieve"


class GPTTreeIndexBuilder:
    """GPT tree index builder.

    Helper class to build the tree-structured index.

    """

    def __init__(
        self,
        num_children: int = 10,
        summary_prompt: Prompt = DEFAULT_SUMMARY_PROMPT,
        llm_predictor: Optional[LLMPredictor] = None,
    ) -> None:
        """Initialize with params."""
        if num_children < 2:
            raise ValueError("Invalid number of children.")
        self.num_children = num_children
        self.summary_prompt = summary_prompt
        chunk_size = get_chunk_size_given_prompt(
            summary_prompt.format(text=""),
            MAX_CHUNK_SIZE,
            num_children,
            NUM_OUTPUTS,
            embedding_limit=EMBED_MAX_TOKEN_LIMIT,
        )
        self.text_splitter = TokenTextSplitter(
            separator=" ",
            chunk_size=chunk_size,
            chunk_overlap=MAX_CHUNK_OVERLAP // num_children,
        )
        self._llm_predictor = llm_predictor or LLMPredictor()

    def _get_nodes_from_document(
        self, start_idx: int, document: BaseDocument
    ) -> Dict[int, Node]:
        """Add document to index."""
        text_chunks = self.text_splitter.split_text(document.get_text())
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
            text_chunk = get_text_from_nodes(cur_nodes_chunk)

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
    """GPT Index."""

    index_struct_cls = IndexGraph

    def __init__(
        self,
        documents: Optional[Sequence[DOCUMENTS_INPUT]] = None,
        index_struct: Optional[IndexGraph] = None,
        summary_template: Prompt = DEFAULT_SUMMARY_PROMPT,
        insert_prompt: Prompt = DEFAULT_INSERT_PROMPT,
        query_str: Optional[str] = None,
        num_children: int = 10,
        llm_predictor: Optional[LLMPredictor] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        # need to set parameters before building index in base class.
        self.num_children = num_children
        # if query_str is specified, then we try to load into summary template
        if query_str is not None:
            summary_template = summary_template.partial_format(query_str=query_str)
        self.summary_template = summary_template
        self.insert_prompt = insert_prompt
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

    def build_index_from_documents(
        self, documents: Sequence[BaseDocument]
    ) -> IndexGraph:
        """Build the index from documents."""
        # do simple concatenation
        index_builder = GPTTreeIndexBuilder(
            num_children=self.num_children,
            summary_prompt=self.summary_template,
            llm_predictor=self._llm_predictor,
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
        )
        inserter.insert(document)

    def delete(self, document: BaseDocument) -> None:
        """Delete a document."""
        raise NotImplementedError("Delete not implemented for tree index.")
