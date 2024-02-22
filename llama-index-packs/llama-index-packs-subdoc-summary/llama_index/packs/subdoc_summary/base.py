"""Subdoc Summary."""

from typing import Any, Dict, List, Optional, List

from llama_index.core.llama_pack import BaseLlamaPack
from llama_index.core.schema import Document
from llama_index.core.text_splitter import SentenceSplitter
from llama_index.core.utils import print_text
from llama_index.core import SummaryIndex, VectorStoreIndex
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms import LLM


DEFAULT_SUMMARY_PROMPT_STR = """\
Please give a concise summary of the context in 1-2 sentences.
"""


class SubDocSummaryPack(BaseLlamaPack):
    """Pack for injecting sub-doc metadata into each chunk."""

    def __init__(
        self,
        documents: List[Document],
        parent_chunk_size: int = 8192,
        parent_chunk_overlap: int = 512,
        child_chunk_size: int = 512,
        child_chunk_overlap: int = 32,
        summary_prompt_str: str = DEFAULT_SUMMARY_PROMPT_STR,
        verbose: bool = False,
        embed_model: Optional[BaseEmbedding] = None,
        llm: Optional[LLM] = None,
    ) -> None:
        """Init params."""
        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size = child_chunk_size

        self.parent_splitter = SentenceSplitter(
            chunk_size=parent_chunk_size, chunk_overlap=parent_chunk_overlap
        )
        self.child_splitter = SentenceSplitter(
            chunk_size=child_chunk_size, chunk_overlap=child_chunk_overlap
        )

        self.summary_prompt_str = summary_prompt_str
        self.embed_model = embed_model
        self.llm = llm

        parent_nodes = self.parent_splitter.get_nodes_from_documents(documents)
        all_child_nodes = []
        # For each parent node, extract the child nodes and print the text
        for idx, parent_node in enumerate(parent_nodes):
            if verbose:
                print_text(
                    f"> Processing parent chunk {idx + 1} of {len(parent_nodes)}\n",
                    color="blue",
                )
            # get summary
            summary_index = SummaryIndex([parent_node])
            summary_query_engine = summary_index.as_query_engine(
                response_mode="tree_summarize"
            )
            parent_summary = summary_query_engine.query(DEFAULT_SUMMARY_PROMPT_STR)
            if verbose:
                print_text(f"Extracted summary: {parent_summary}\n", color="pink")

            # attach summary to all child nodes
            child_nodes = self.child_splitter.get_nodes_from_documents([parent_node])
            for child_node in child_nodes:
                child_node.metadata["context_summary"] = str(parent_summary)

            all_child_nodes.extend(child_nodes)

        # build vector index for child nodes
        self.vector_index = VectorStoreIndex(
            all_child_nodes, embed_model=self.embed_model
        )
        self.vector_retriever = self.vector_index.as_retriever()
        self.vector_query_engine = self.vector_index.as_query_engine(llm=llm)

        self.verbose = verbose

    def get_modules(self) -> Dict[str, Any]:
        """Get modules."""
        return {
            "vector_index": self.vector_index,
            "vector_retriever": self.vector_retriever,
            "vector_query_engine": self.vector_query_engine,
        }

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Run the pipeline."""
        return self.vector_query_engine.query(*args, **kwargs)
