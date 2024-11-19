"""Corrective RAG LlamaPack class."""

from typing import Any, Dict, List, Optional

from llama_index.core import VectorStoreIndex, SummaryIndex
from llama_index.core.async_utils import asyncio_run
from llama_index.core.llama_pack.base import BaseLlamaPack
from llama_index.llms.openai import OpenAI
from llama_index.core.schema import Document, NodeWithScore
from llama_index.core.query_pipeline.query import QueryPipeline
from llama_index.tools.tavily_research.base import TavilyToolSpec
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.prompts import PromptTemplate

from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    step,
    Workflow,
    Context,
    Event,
)

DEFAULT_RELEVANCY_PROMPT_TEMPLATE = PromptTemplate(
    template="""As a grader, your task is to evaluate the relevance of a document retrieved in response to a user's question.

    Retrieved Document:
    -------------------
    {context_str}

    User Question:
    --------------
    {query_str}

    Evaluation Criteria:
    - Consider whether the document contains keywords or topics related to the user's question.
    - The evaluation should not be overly stringent; the primary objective is to identify and filter out clearly irrelevant retrievals.

    Decision:
    - Assign a binary score to indicate the document's relevance.
    - Use 'yes' if the document is relevant to the question, or 'no' if it is not.

    Please provide your binary score ('yes' or 'no') below to indicate the document's relevance to the user question."""
)

DEFAULT_TRANSFORM_QUERY_TEMPLATE = PromptTemplate(
    template="""Your task is to refine a query to ensure it is highly effective for retrieving relevant search results. \n
    Analyze the given input to grasp the core semantic intent or meaning. \n
    Original Query:
    \n ------- \n
    {query_str}
    \n ------- \n
    Your goal is to rephrase or enhance this query to improve its search performance. Ensure the revised query is concise and directly aligned with the intended search objective. \n
    Respond with the optimized query only:"""
)


class RetrieveEvent(Event):
    """Retrieve event (gets retrieved nodes)."""

    retrieved_nodes: List[NodeWithScore]


class RelevanceEvalEvent(Event):
    """Relevance evaluation event (gets results of relevance evaluation)."""

    relevant_results: List[str]


class TextExtractEvent(Event):
    """Text extract event. Extracts relevant text and concatenates."""

    relevant_text: str


class QueryEvent(Event):
    """Query event. Queries given relevant text and search text."""

    relevant_text: str
    search_text: str


class CorrectiveRAGWorkflow(Workflow):
    @step(pass_context=True)
    async def ingest(self, ctx: Context, ev: StartEvent) -> Optional[StopEvent]:
        """Ingest step (for ingesting docs and initializing index)."""
        documents: Optional[List[Document]] = ev.get("documents")
        tavily_ai_apikey: Optional[str] = ev.get("tavily_ai_apikey")

        if any(i is None for i in [documents, tavily_ai_apikey]):
            return None

        llm = OpenAI(model="gpt-4")
        ctx.data["relevancy_pipeline"] = QueryPipeline(
            chain=[DEFAULT_RELEVANCY_PROMPT_TEMPLATE, llm]
        )
        ctx.data["transform_query_pipeline"] = QueryPipeline(
            chain=[DEFAULT_TRANSFORM_QUERY_TEMPLATE, llm]
        )

        ctx.data["llm"] = llm
        ctx.data["index"] = VectorStoreIndex.from_documents(documents)
        ctx.data["tavily_tool"] = TavilyToolSpec(api_key=tavily_ai_apikey)

        return StopEvent()

    @step(pass_context=True)
    async def retrieve(self, ctx: Context, ev: StartEvent) -> Optional[RetrieveEvent]:
        """Retrieve the relevant nodes for the query."""
        query_str = ev.get("query_str")
        retriever_kwargs = ev.get("retriever_kwargs", {})

        if query_str is None:
            return None

        if "index" not in ctx.data or "tavily_tool" not in ctx.data:
            raise ValueError(
                "Index and tavily tool must be constructed. Run with 'documents' and 'tavily_ai_apikey' params first."
            )

        retriever: BaseRetriever = ctx.data["index"].as_retriever(**retriever_kwargs)
        result = retriever.retrieve(query_str)
        ctx.data["retrieved_nodes"] = result
        ctx.data["query_str"] = query_str
        return RetrieveEvent(retrieved_nodes=result)

    @step(pass_context=True)
    async def eval_relevance(
        self, ctx: Context, ev: RetrieveEvent
    ) -> RelevanceEvalEvent:
        """Evaluate relevancy of retrieved documents with the query."""
        retrieved_nodes = ev.retrieved_nodes
        query_str = ctx.data["query_str"]

        relevancy_results = []
        for node in retrieved_nodes:
            relevancy = ctx.data["relevancy_pipeline"].run(
                context_str=node.text, query_str=query_str
            )
            relevancy_results.append(relevancy.message.content.lower().strip())

        ctx.data["relevancy_results"] = relevancy_results
        return RelevanceEvalEvent(relevant_results=relevancy_results)

    @step(pass_context=True)
    async def extract_relevant_texts(
        self, ctx: Context, ev: RelevanceEvalEvent
    ) -> TextExtractEvent:
        """Extract relevant texts from retrieved documents."""
        retrieved_nodes = ctx.data["retrieved_nodes"]
        relevancy_results = ev.relevant_results

        relevant_texts = [
            retrieved_nodes[i].text
            for i, result in enumerate(relevancy_results)
            if result == "yes"
        ]

        result = "\n".join(relevant_texts)
        return TextExtractEvent(relevant_text=result)

    @step(pass_context=True)
    async def transform_query_pipeline(
        self, ctx: Context, ev: TextExtractEvent
    ) -> QueryEvent:
        """Search the transformed query with Tavily API."""
        relevant_text = ev.relevant_text
        relevancy_results = ctx.data["relevancy_results"]
        query_str = ctx.data["query_str"]

        # If any document is found irrelevant, transform the query string for better search results.
        if "no" in relevancy_results:
            transformed_query_str = (
                ctx.data["transform_query_pipeline"]
                .run(query_str=query_str)
                .message.content
            )
            # Conduct a search with the transformed query string and collect the results.
            search_results = ctx.data["tavily_tool"].search(
                transformed_query_str, max_results=5
            )
            search_text = "\n".join([result.text for result in search_results])
        else:
            search_text = ""

        return QueryEvent(relevant_text=relevant_text, search_text=search_text)

    @step(pass_context=True)
    async def query_result(self, ctx: Context, ev: QueryEvent) -> StopEvent:
        """Get result with relevant text."""
        relevant_text = ev.relevant_text
        search_text = ev.search_text
        query_str = ctx.data["query_str"]

        documents = [Document(text=relevant_text + "\n" + search_text)]
        index = SummaryIndex.from_documents(documents)
        query_engine = index.as_query_engine()
        result = query_engine.query(query_str)
        return StopEvent(result=result)


class CorrectiveRAGPack(BaseLlamaPack):
    def __init__(self, documents: List[Document], tavily_ai_apikey: str) -> None:
        """Init params."""
        self._wf = CorrectiveRAGWorkflow()

        asyncio_run(
            self._wf.run(documents=documents, tavily_ai_apikey=tavily_ai_apikey)
        )

        self.llm = OpenAI(model="gpt-4")
        self.index = self._wf.get_context("ingest").data["index"]

    def get_modules(self) -> Dict[str, Any]:
        """Get modules."""
        return {"llm": self.llm, "index": self.index}

    def run(self, query_str: str, **kwargs: Any) -> Any:
        """Run the pipeline."""
        return asyncio_run(self._wf.run(query_str=query_str, retriever_kwargs=kwargs))
