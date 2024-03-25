from typing import Dict, Any, List

from docugami import Docugami
from llama_index.core.tools import BaseTool
from llama_index.core.llama_pack import BaseLlamaPack
from llama_index.core.agent import ReActAgent

from llama_index.packs.docugami_kg_rag.helpers.prompts import ASSISTANT_SYSTEM_MESSAGE
from llama_index.packs.docugami_kg_rag.config import (
    LARGE_CONTEXT_INSTRUCT_LLM,
    DEFAULT_USE_REPORTS,
)
from llama_index.packs.docugami_kg_rag.helpers.indexing import (
    read_all_local_index_state,
    index_docset,
)
from llama_index.packs.docugami_kg_rag.helpers.reports import (
    get_retrieval_tool_for_report,
)
from llama_index.packs.docugami_kg_rag.helpers.retrieval import (
    get_retrieval_tool_for_docset,
)


class DocugamiKgRagPack(BaseLlamaPack):
    """Docugami KG-RAG Pack.

    A pack for performing evaluation with your own RAG pipeline.

    """

    def __init__(self) -> None:
        self.docugami_client = Docugami()

    def list_docsets(self):
        """
        List your Docugami docsets and their docset name and ids.
        """
        docsets_response = self.docugami_client.docsets.list()
        for idx, docset in enumerate(docsets_response.docsets, start=1):
            print(f"{idx}: {docset.name} (ID: {docset.id})")

    def index_docset(self, docset_id: str, overwrite: bool = False):
        """
        Build the index for the docset and create the agent for it.
        """
        docsets_response = self.docugami_client.docsets.list()
        docset = next(
            (docset for docset in docsets_response.docsets if docset.id == docset_id),
            None,
        )

        if not docset:
            raise Exception(
                f"Docset with id {docset_id} does not exist in your workspace"
            )

        index_docset(docset_id, docset.name, overwrite)

    def build_agent_for_docset(
        self, docset_id: str, use_reports: bool = DEFAULT_USE_REPORTS
    ):
        local_state = read_all_local_index_state()

        tools: List[BaseTool] = []
        for docset_id in local_state:
            docset_state = local_state[docset_id]
            direct_retrieval_tool = get_retrieval_tool_for_docset(
                docset_id, docset_state
            )
            if direct_retrieval_tool:
                # Direct retrieval tool for each indexed docset (direct KG-RAG against semantic XML)
                tools.append(direct_retrieval_tool)

            if use_reports:
                for report in docset_state.reports:
                    # Report retrieval tool for each published report (user-curated views on semantic XML)
                    report_retrieval_tool = get_retrieval_tool_for_report(report)
                    if report_retrieval_tool:
                        tools.append(report_retrieval_tool)

        self.agent = ReActAgent.from_tools(
            tools,
            llm=LARGE_CONTEXT_INSTRUCT_LLM,
            verbose=True,
            context=ASSISTANT_SYSTEM_MESSAGE,
        )

    def get_modules(self) -> Dict[str, Any]:
        """Get modules."""
        return {
            "agent": self.agent,
        }

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Run the pipeline."""
        return self.agent.query(*args, **kwargs)
