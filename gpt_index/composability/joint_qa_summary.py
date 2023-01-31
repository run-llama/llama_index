"""Joint QA Summary graph."""


from typing import Sequence

from gpt_index.composability.base import BaseGraphBuilder, Graph
from gpt_index.indices.base import DOCUMENTS_INPUT, BaseGPTIndex
from gpt_index.indices.list.base import GPTListIndex
from gpt_index.indices.tree.base import GPTTreeIndex
from gpt_index.indices.vector_store.simple import GPTSimpleVectorIndex

DEFAULT_SUMMARY_TEXT = "Use this index for summarization queries"
DEFAULT_QA_TEXT = (
    "Use this index for queries that require retrieval of specific "
    "context from documents."
)


class QASummaryGraphBuilder(BaseGraphBuilder):
    """Joint QA Summary graph builder."""

    def build_graph_from_documents(
        self,
        documents: Sequence[DOCUMENTS_INPUT],
        verbose: bool = False,
        summary_text: str = DEFAULT_SUMMARY_TEXT,
        qa_text: str = DEFAULT_QA_TEXT,
    ) -> "Graph":
        """Build graph from index."""

        # used for QA
        vector_index = GPTSimpleVectorIndex(
            documents,
            llm_predictor=self._llm_predictor,
            embed_model=self._embed_model,
            docstore=self._docstore,
            index_registry=self._index_registry,
            prompt_helper=self._prompt_helper,
        )
        # used for summarization
        list_index = GPTListIndex(
            documents,
            llm_predictor=self._llm_predictor,
            embed_model=self._embed_model,
            docstore=self._docstore,
            index_registry=self._index_registry,
            prompt_helper=self._prompt_helper,
        )

        vector_index.set_text(qa_text)
        list_index.set_text(summary_text)

        # # used for top-level indices
        # top_index = GPTSimpleVectorIndex(
        #     [vector_index, list_index],
        #     llm_predictor=self._llm_predictor,
        #     embed_model=self._embed_model,
        #     docstore=self._docstore,
        #     index_registry=self._index_registry,
        #     prompt_helper=self._prompt_helper,
        # )

        # used for top-level indices
        top_index = GPTTreeIndex(
            [vector_index, list_index],
            llm_predictor=self._llm_predictor,
            embed_model=self._embed_model,
            docstore=self._docstore,
            index_registry=self._index_registry,
            prompt_helper=self._prompt_helper,
        )
        return Graph.build_from_index(top_index)
