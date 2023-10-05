"""Joint QA Summary graph."""


from typing import Optional, Sequence

from llama_index.indices.list.base import SummaryIndex
from llama_index.indices.service_context import ServiceContext
from llama_index.indices.vector_store import VectorStoreIndex
from llama_index.query_engine.router_query_engine import RouterQueryEngine
from llama_index.schema import Document
from llama_index.storage.storage_context import StorageContext
from llama_index.tools.query_engine import QueryEngineTool

DEFAULT_SUMMARY_TEXT = "Use this index for summarization queries"
DEFAULT_QA_TEXT = (
    "Use this index for queries that require retrieval of specific "
    "context from documents."
)


class QASummaryQueryEngineBuilder:
    """Joint QA Summary graph builder.

    Can build a graph that provides a unified query interface
    for both QA and summarization tasks.

    NOTE: this is a beta feature. The API may change in the future.

    Args:
        docstore (BaseDocumentStore): A BaseDocumentStore to use for storing nodes.
        service_context (ServiceContext): A ServiceContext to use for
            building indices.
        summary_text (str): Text to use for the summary index.
        qa_text (str): Text to use for the QA index.
        node_parser (NodeParser): A NodeParser to use for parsing.

    """

    def __init__(
        self,
        storage_context: Optional[StorageContext] = None,
        service_context: Optional[ServiceContext] = None,
        summary_text: str = DEFAULT_SUMMARY_TEXT,
        qa_text: str = DEFAULT_QA_TEXT,
    ) -> None:
        """Init params."""
        self._storage_context = storage_context or StorageContext.from_defaults()
        self._service_context = service_context or ServiceContext.from_defaults()
        self._summary_text = summary_text
        self._qa_text = qa_text

    def build_from_documents(
        self,
        documents: Sequence[Document],
    ) -> RouterQueryEngine:
        """Build query engine."""
        # parse nodes
        nodes = self._service_context.node_parser.get_nodes_from_documents(documents)

        # ingest nodes
        self._storage_context.docstore.add_documents(nodes, allow_update=True)

        # build indices
        vector_index = VectorStoreIndex(
            nodes,
            service_context=self._service_context,
            storage_context=self._storage_context,
        )
        summary_index = SummaryIndex(
            nodes,
            service_context=self._service_context,
            storage_context=self._storage_context,
        )

        vector_query_engine = vector_index.as_query_engine(
            service_context=self._service_context
        )
        list_query_engine = summary_index.as_query_engine(
            service_context=self._service_context,
            response_mode="tree_summarize",
        )

        # build query engine
        return RouterQueryEngine.from_defaults(
            query_engine_tools=[
                QueryEngineTool.from_defaults(
                    vector_query_engine, description=self._qa_text
                ),
                QueryEngineTool.from_defaults(
                    list_query_engine, description=self._summary_text
                ),
            ],
            service_context=self._service_context,
            select_multi=False,
        )
