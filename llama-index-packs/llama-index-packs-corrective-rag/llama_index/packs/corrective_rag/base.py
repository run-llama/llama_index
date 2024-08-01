"""Corrective RAG LlamaPack class."""

from typing import Any, Dict, List

from llama_index.core import VectorStoreIndex, SummaryIndex
from llama_index.core.llama_pack.base import BaseLlamaPack
from llama_index.llms.openai import OpenAI
from llama_index.core.schema import Document, NodeWithScore
from llama_index.core.query_pipeline.query import QueryPipeline
from llama_index.tools.tavily_research.base import TavilyToolSpec
from llama_index.core.prompts import PromptTemplate

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


class CorrectiveRAGPack(BaseLlamaPack):
    def __init__(self, documents: List[Document], tavily_ai_apikey: str) -> None:
        """Init params."""
        llm = OpenAI(model="gpt-4")
        self.relevancy_pipeline = QueryPipeline(
            chain=[DEFAULT_RELEVANCY_PROMPT_TEMPLATE, llm]
        )
        self.transform_query_pipeline = QueryPipeline(
            chain=[DEFAULT_TRANSFORM_QUERY_TEMPLATE, llm]
        )

        self.llm = llm
        self.index = VectorStoreIndex.from_documents(documents)
        self.tavily_tool = TavilyToolSpec(api_key=tavily_ai_apikey)

    def get_modules(self) -> Dict[str, Any]:
        """Get modules."""
        return {"llm": self.llm, "index": self.index}

    def retrieve_nodes(self, query_str: str, **kwargs: Any) -> List[NodeWithScore]:
        """Retrieve the relevant nodes for the query."""
        retriever = self.index.as_retriever(**kwargs)
        return retriever.retrieve(query_str)

    def evaluate_relevancy(
        self, retrieved_nodes: List[Document], query_str: str
    ) -> List[str]:
        """Evaluate relevancy of retrieved documents with the query."""
        relevancy_results = []
        for node in retrieved_nodes:
            relevancy = self.relevancy_pipeline.run(
                context_str=node.text, query_str=query_str
            )
            relevancy_results.append(relevancy.message.content.lower().strip())
        return relevancy_results

    def extract_relevant_texts(
        self, retrieved_nodes: List[NodeWithScore], relevancy_results: List[str]
    ) -> str:
        """Extract relevant texts from retrieved documents."""
        relevant_texts = [
            retrieved_nodes[i].text
            for i, result in enumerate(relevancy_results)
            if result == "yes"
        ]
        return "\n".join(relevant_texts)

    def search_with_transformed_query(self, query_str: str) -> str:
        """Search the transformed query with Tavily API."""
        search_results = self.tavily_tool.search(query_str, max_results=5)
        return "\n".join([result.text for result in search_results])

    def get_result(self, relevant_text: str, search_text: str, query_str: str) -> Any:
        """Get result with relevant text."""
        documents = [Document(text=relevant_text + "\n" + search_text)]
        index = SummaryIndex.from_documents(documents)
        query_engine = index.as_query_engine()
        return query_engine.query(query_str)

    def run(self, query_str: str, **kwargs: Any) -> Any:
        """Run the pipeline."""
        # Retrieve nodes based on the input query string.
        retrieved_nodes = self.retrieve_nodes(query_str, **kwargs)

        # Evaluate the relevancy of each retrieved document in relation to the query string.
        relevancy_results = self.evaluate_relevancy(retrieved_nodes, query_str)
        # Extract texts from documents that are deemed relevant based on the evaluation.
        relevant_text = self.extract_relevant_texts(retrieved_nodes, relevancy_results)

        # Initialize search_text variable to handle cases where it might not get defined.
        search_text = ""

        # If any document is found irrelevant, transform the query string for better search results.
        if "no" in relevancy_results:
            transformed_query_str = self.transform_query_pipeline.run(
                query_str=query_str
            ).message.content
            # Conduct a search with the transformed query string and collect the results.
            search_text = self.search_with_transformed_query(transformed_query_str)

        # Compile the final result. If there's additional search text from the transformed query,
        # it's included; otherwise, only the relevant text from the initial retrieval is returned.
        if search_text:
            return self.get_result(relevant_text, search_text, query_str)
        else:
            return self.get_result(relevant_text, "", query_str)
