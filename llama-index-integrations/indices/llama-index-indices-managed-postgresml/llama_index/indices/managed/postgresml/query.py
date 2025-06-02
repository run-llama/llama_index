from typing import Any, Optional, Dict, Generator, AsyncGenerator
from copy import deepcopy
from llama_index.core.async_utils import run_async_tasks
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.response.schema import (
    RESPONSE_TYPE,
    Response,
    AsyncStreamingResponse,
    StreamingResponse,
)
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.prompts.base import ChatPromptTemplate, BasePromptTemplate
from llama_index.core.prompts.mixin import PromptDictType, PromptMixinType
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.indices.managed.postgresml.retriever import PostgresMLRetriever


PROMPTS = {
    "text_qa_template": ChatPromptTemplate(
        message_templates=[
            ChatMessage(content="You are a helpful chatbot", role=MessageRole.SYSTEM),
            ChatMessage(
                content="""Context information is below.
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge,
Query: {query_str}
Answer: """,
                role=MessageRole.USER,
            ),
        ]
    )
}


class AsyncJsonGenerator(Generator, AsyncGenerator):
    def __init__(self, rag_stream_results) -> None:
        self.rag_stream_results = rag_stream_results
        self.rag_stream = None

    def asend(self):
        raise Exception("asend is not implemented")

    def send(self):
        raise Exception("send is not implemented")

    def athrow(self):
        raise Exception("athrow is not implemented")

    def throw(self):
        raise Exception("throw is not implemented")

    def __iter__(self) -> "AsyncJsonGenerator":
        return self

    def __aiter__(self) -> "AsyncJsonGenerator":
        return self

    def __next__(self) -> str:
        try:
            return run_async_tasks([self.__anext__()])[0]
        except StopAsyncIteration:
            raise StopIteration

    async def __anext__(self) -> str:
        if not self.rag_stream:
            self.rag_stream = self.rag_stream_results.stream()
        result = await self.rag_stream.__anext__()
        if len(result) > 0:
            return result[0]
        else:
            return ""


class PostgresMLQueryEngine(BaseQueryEngine):
    """Retriever query engine for PostgresML."""

    def __init__(
        self,
        retriever: PostgresMLRetriever,
        streaming: Optional[bool] = False,
        callback_manager: Optional[CallbackManager] = None,
        pgml_query: Optional[Dict[str, Any]] = None,
        vector_search_limit: Optional[int] = 4,
        vector_search_rerank: Optional[Dict[str, Any]] = None,
        vector_search_document: Optional[Dict[str, Any]] = {"keys": ["id", "metadata"]},
        model: Optional[str] = "meta-llama/Meta-Llama-3-8B-Instruct",
        model_parameters: Optional[Dict[str, Any]] = {"max_tokens": 2048},
        **kwargs,
    ) -> None:
        self._retriever = retriever
        self._streaming = streaming
        self._prompts = deepcopy(PROMPTS)
        self._pgml_query = pgml_query
        self._vector_search_limit = vector_search_limit
        self._vector_search_rerank = vector_search_rerank
        self._vector_search_document = vector_search_document
        self._model = model
        self._model_parameters = model_parameters
        super().__init__(callback_manager=callback_manager)

    @classmethod
    def from_args(
        cls,
        retriever: PostgresMLRetriever,
        **kwargs: Any,
    ) -> "PostgresMLQueryEngine":
        """
        Initialize a PostgresMLQueryEngine object.".

        Args:
            retriever (PostgresMLRetriever): A PostgresML retriever object.

        """
        return cls(retriever=retriever, **kwargs)

    def with_retriever(self, retriever: PostgresMLRetriever) -> "PostgresMLQueryEngine":
        return PostgresMLQueryEngine(
            retriever=retriever,
        )

    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Answer a query."""
        if self._streaming:
            async_token_gen = run_async_tasks([self._do_query(query_bundle)])[0]
            return StreamingResponse(response_gen=async_token_gen.response_gen)
        else:
            return run_async_tasks([self._do_query(query_bundle)])[0]

    async def _aquery(self, query_bundle: QueryBundle, **kwargs) -> RESPONSE_TYPE:
        """Answer an async query."""
        return await self._do_query(query_bundle, **kwargs)

    async def _do_query(
        self, query_bundle: Optional[QueryBundle] = None
    ) -> RESPONSE_TYPE:
        query = self._pgml_query
        if not query:
            if not query_bundle:
                raise Exception(
                    "Must provide either query or query_bundle to query and aquery"
                )

            # {CONTEXT} gets replaced with the correct context in the SQL query generated by the pgml SDK
            messages = self._prompts["text_qa_template"].format_messages(
                context_str="{CONTEXT}", query_str=query_bundle.query_str
            )
            messages = [
                {"role": m["role"].value, "content": m["content"]}
                for m in [m.dict() for m in messages]
            ]

            model_parameters = deepcopy(self._model_parameters)
            model_parameters["model"] = self._model
            model_parameters["messages"] = messages

            if self._vector_search_rerank is not None:
                self._vector_search_rerank = self._vector_search_rerank | {
                    "query": query_bundle.query_str
                }

            query = {
                "CONTEXT": {
                    "vector_search": {
                        "query": {
                            "fields": {
                                "content": {
                                    "query": query_bundle.query_str,
                                    "parameters": {"prompt": "query: "},
                                },
                            },
                        },
                        "document": self._vector_search_document,
                        "limit": self._vector_search_limit,
                        "rerank": self._vector_search_rerank,
                    },
                    "aggregate": {"join": "\n"},
                },
                "chat": model_parameters,
            }

        if self._streaming:
            # The pgml SDK does not currently return sources for streaming
            results = await self._retriever._index.collection.rag_stream(
                query,
                self._retriever._index.pipeline,
            )
            return AsyncStreamingResponse(response_gen=AsyncJsonGenerator(results))
        else:
            results = await self._retriever._index.collection.rag(
                query,
                self._retriever._index.pipeline,
            )
            source_nodes = [
                NodeWithScore(
                    node=TextNode(
                        id_=r["document"]["id"],
                        text=r["chunk"],
                        metadata=r["document"]["metadata"],
                    ),
                    score=r["score"],
                )
                if self._vector_search_rerank is None
                else NodeWithScore(
                    node=TextNode(
                        id_=r["document"]["id"],
                        text=r["chunk"],
                        metadata=r["document"]["metadata"],
                    ),
                    score=r["rerank_score"],
                )
                for r in results["sources"]["CONTEXT"]
            ]
            return Response(response=results["rag"][0], source_nodes=source_nodes)

    def _get_prompt_modules(self) -> PromptMixinType:
        """Get prompt modules."""
        return {}

    def _get_prompts(self) -> Dict[str, BasePromptTemplate]:
        """Get prompts."""
        return self._prompts

    def _update_prompts(self, prompts_dict: PromptDictType):
        """Update prompts."""
        for key in prompts_dict:
            self._prompts[key] = prompts_dict[key]
