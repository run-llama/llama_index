"""Multi-document agents Pack."""

from typing import Any, Dict, List

from llama_index.agent.openai import OpenAIAgent
from llama_index.agent.openai_legacy import FnRetrieverOpenAIAgent
from llama_index.core import Settings, SummaryIndex, VectorStoreIndex
from llama_index.core.llama_pack.base import BaseLlamaPack
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.objects import ObjectIndex, SimpleToolNodeMapping
from llama_index.core.schema import Document
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.llms.openai import OpenAI


class MultiDocumentAgentsPack(BaseLlamaPack):
    """Multi-document Agents pack.

    Given a set of documents, build our multi-document agents architecture.
    - setup a document agent over agent doc (capable of QA and summarization)
    - setup a top-level agent over doc agents

    """

    def __init__(
        self,
        docs: List[Document],
        doc_titles: List[str],
        doc_descriptions: List[str],
        **kwargs: Any,
    ) -> None:
        """Init params."""
        self.node_parser = SentenceSplitter()
        self.llm = OpenAI(temperature=0, model="gpt-3.5-turbo")
        Settings.llm = self.llm

        # Build agents dictionary
        self.agents = {}

        # this is for the baseline
        all_nodes = []

        # build agent for each document
        for idx, doc in enumerate(docs):
            doc_title = doc_titles[idx]
            doc_description = doc_descriptions[idx]
            nodes = self.node_parser.get_nodes_from_documents([doc])
            all_nodes.extend(nodes)

            # build vector index
            vector_index = VectorStoreIndex(nodes)

            # build summary index
            summary_index = SummaryIndex(nodes)
            # define query engines
            vector_query_engine = vector_index.as_query_engine()
            summary_query_engine = summary_index.as_query_engine()

            # define tools
            query_engine_tools = [
                QueryEngineTool(
                    query_engine=vector_query_engine,
                    metadata=ToolMetadata(
                        name="vector_tool",
                        description=(
                            "Useful for questions related to specific aspects of"
                            f" {doc_title}."
                        ),
                    ),
                ),
                QueryEngineTool(
                    query_engine=summary_query_engine,
                    metadata=ToolMetadata(
                        name="summary_tool",
                        description=(
                            "Useful for any requests that require a holistic summary"
                            f" of EVERYTHING about {doc_title}. "
                        ),
                    ),
                ),
            ]

            # build agent
            function_llm = OpenAI(model="gpt-4")
            agent = OpenAIAgent.from_tools(
                query_engine_tools,
                llm=function_llm,
                verbose=True,
                system_prompt=f"""\
        You are a specialized agent designed to answer queries about {doc_title}.
        You must ALWAYS use at least one of the tools provided when answering a question; do NOT rely on prior knowledge.\
        """,
            )

            self.agents[doc_title] = agent

        # build top-level, retrieval-enabled OpenAI Agent
        # define tool for each document agent
        all_tools = []
        for idx, doc in enumerate(docs):
            doc_title = doc_titles[idx]
            doc_description = doc_descriptions[idx]
            wiki_summary = (
                f"Use this tool if you want to answer any questions about {doc_title}.\n"
                f"Doc description: {doc_description}\n"
            )
            doc_tool = QueryEngineTool(
                query_engine=self.agents[doc_title],
                metadata=ToolMetadata(
                    name=f"tool_{doc_title}",
                    description=wiki_summary,
                ),
            )
            all_tools.append(doc_tool)

        tool_mapping = SimpleToolNodeMapping.from_objects(all_tools)
        self.obj_index = ObjectIndex.from_objects(
            all_tools,
            tool_mapping,
            VectorStoreIndex,
        )
        self.top_agent = FnRetrieverOpenAIAgent.from_retriever(
            self.obj_index.as_retriever(similarity_top_k=3),
            system_prompt=""" \
        You are an agent designed to answer queries about a set of given cities.
        Please always use the tools provided to answer a question. Do not rely on prior knowledge.\

        """,
            verbose=True,
        )

    def get_modules(self) -> Dict[str, Any]:
        """Get modules."""
        return {
            "top_agent": self.top_agent,
            "obj_index": self.obj_index,
            "doc_agents": self.agents,
        }

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Run the pipeline."""
        return self.top_agent.query(*args, **kwargs)
