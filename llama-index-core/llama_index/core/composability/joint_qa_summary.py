"""Joint QA Summary graph."""

from typing import List, Optional, Sequence

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.indices.list.base import SummaryIndex
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.core.ingestion import run_transformations
from llama_index.core.llms.llm import LLM
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.schema import Document, TransformComponent
from llama_index.core.settings import Settings
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.tools.query_engine import QueryEngineTool

DEFAULT_SUMMARY_TEXT = "Use this index for summarization queries"
DEFAULT_QA_TEXT = (
    "Use this index for queries that require retrieval of specific "
    "context from documents."
)


class QASummaryQueryEngineBuilder:
    """
    Joint QA Summary graph builder.

    Can build a graph that provides a unified query interface
    for both QA and summarization tasks.

    NOTE: this is a beta feature. The API may change in the future.

    Args:
        docstore (BaseDocumentStore): A BaseDocumentStore to use for storing nodes.
        summary_text (str): Text to use for the summary index.
        qa_text (str): Text to use for the QA index.
        node_parser (NodeParser): A NodeParser to use for parsing.

    """

    def __init__(
        self,
        llm: Optional[LLM] = None,
        embed_model: Optional[BaseEmbedding] = None,
        callback_manager: Optional[CallbackManager] = None,
        transformations: Optional[List[TransformComponent]] = None,
        storage_context: Optional[StorageContext] = None,
        summary_text: str = DEFAULT_SUMMARY_TEXT,
        qa_text: str = DEFAULT_QA_TEXT,
    ) -> None:
        """Init params."""
        self._llm = llm or Settings.llm
        self._callback_manager = callback_manager or Settings.callback_manager
        self._embed_model = embed_model or Settings.embed_model
        self._transformations = transformations or Settings.transformations

        self._storage_context = storage_context or StorageContext.from_defaults()
        self._summary_text = summary_text
        self._qa_text = qa_text

    def build_from_documents(
        self,
        documents: Sequence[Document],
    ) -> RouterQueryEngine:
        """Build query engine."""
        # parse nodes
        nodes = run_transformations(documents, self._transformations)  # type: ignore

        # ingest nodes
        self._storage_context.docstore.add_documents(nodes, allow_update=True)

        # build indices
        vector_index = VectorStoreIndex(
            nodes=nodes,
            transformations=self._transformations,
            embed_model=self._embed_model,
            storage_context=self._storage_context,
        )
        summary_index = SummaryIndex(nodes, storage_context=self._storage_context)

        vector_query_engine = vector_index.as_query_engine(llm=self._llm)
        list_query_engine = summary_index.as_query_engine(
            llm=self._llm, response_mode="tree_summarize"
        )

        # build query engine
        return RouterQueryEngine.from_defaults(
            llm=self._llm,
            query_engine_tools=[
                QueryEngineTool.from_defaults(
                    vector_query_engine, description=self._qa_text
                ),
                QueryEngineTool.from_defaults(
                    list_query_engine, description=self._summary_text
                ),
            ],
            select_multi=False,
        )
