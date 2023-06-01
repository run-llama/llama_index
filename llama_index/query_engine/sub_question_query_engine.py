import asyncio
import logging
from typing import List, Optional, Sequence, cast

from langchain.input import get_color_mapping, print_text

from llama_index.async_utils import run_async_tasks
from llama_index.callbacks.base import CallbackManager
from llama_index.data_structs.node import Node, NodeWithScore
from llama_index.indices.query.base import BaseQueryEngine
from llama_index.indices.query.response_synthesis import ResponseSynthesizer
from llama_index.indices.query.schema import QueryBundle
from llama_index.indices.service_context import ServiceContext
from llama_index.question_gen.llm_generators import LLMQuestionGenerator
from llama_index.question_gen.types import BaseQuestionGenerator, SubQuestion
from llama_index.response.schema import RESPONSE_TYPE
from llama_index.tools.query_engine import QueryEngineTool

logger = logging.getLogger(__name__)


class SubQuestionQueryEngine(BaseQueryEngine):
    """Sub question query engine.

    A query engine that breaks down a complex query (e.g. compare and contrast) into
        many sub questions and their target query engine for execution.
        After executing all sub questions, all responses are gathered and sent to
        response synthesizer to produce the final response.

    Args:
        question_gen (BaseQuestionGenerator): A module for generating sub questions
            given a complex question and tools.
        response_synthesizer (ResponseSynthesizer): A response synthesizer for
            generating the final response
        query_engine_tools (Sequence[QueryEngineTool]): Tools to answer the
            sub questions.
        verbose (bool): whether to print intermediate questions and answers.
            Defaults to True
        use_async (bool): whether to execute the sub questions with asyncio.
            Defaults to True
    """

    def __init__(
        self,
        question_gen: BaseQuestionGenerator,
        response_synthesizer: ResponseSynthesizer,
        query_engine_tools: Sequence[QueryEngineTool],
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = True,
        use_async: bool = False,
    ) -> None:
        self._question_gen = question_gen
        self._response_synthesizer = response_synthesizer
        self._metadatas = [x.metadata for x in query_engine_tools]
        self._query_engines = {
            tool.metadata.name: tool.query_engine for tool in query_engine_tools
        }
        self._verbose = verbose
        self._use_async = use_async
        super().__init__(callback_manager)

    @classmethod
    def from_defaults(
        cls,
        query_engine_tools: Sequence[QueryEngineTool],
        question_gen: Optional[BaseQuestionGenerator] = None,
        response_synthesizer: Optional[ResponseSynthesizer] = None,
        service_context: Optional[ServiceContext] = None,
        verbose: bool = True,
        use_async: bool = True,
    ) -> "SubQuestionQueryEngine":
        callback_manager = None
        if len(query_engine_tools) > 0:
            callback_manager = query_engine_tools[0].query_engine.callback_manager

        question_gen = question_gen or LLMQuestionGenerator.from_defaults(
            service_context=service_context
        )
        synth = response_synthesizer or ResponseSynthesizer.from_args(
            callback_manager=callback_manager,
            service_context=service_context,
        )

        return cls(
            question_gen,
            synth,
            query_engine_tools,
            callback_manager=callback_manager,
            verbose=verbose,
            use_async=use_async,
        )

    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        sub_questions = self._question_gen.generate(self._metadatas, query_bundle)

        if self._verbose:
            print_text(f"Generated {len(sub_questions)} sub questions.\n")
            colors = get_color_mapping([str(i) for i in range(len(sub_questions))])

        if self._use_async:
            tasks = [
                self._aquery_subq(sub_q, color=colors[str(ind)])
                for ind, sub_q in enumerate(sub_questions)
            ]

            nodes_all = run_async_tasks(tasks)
            nodes_all = cast(List[Optional[NodeWithScore]], nodes_all)
        else:
            nodes_all = [
                self._query_subq(sub_q, color=colors[str(ind)])
                for ind, sub_q in enumerate(sub_questions)
            ]

        # filter out sub questions that failed
        nodes: List[NodeWithScore] = list(filter(None, nodes_all))

        return self._response_synthesizer.synthesize(
            query_bundle=query_bundle,
            nodes=nodes,
        )

    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        sub_questions = await self._question_gen.agenerate(
            self._metadatas, query_bundle
        )

        if self._verbose:
            print_text(f"Generated {len(sub_questions)} sub questions.\n")
            colors = get_color_mapping([str(i) for i in range(len(sub_questions))])

        tasks = [
            self._aquery_subq(sub_q, color=colors[str(ind)])
            for ind, sub_q in enumerate(sub_questions)
        ]
        nodes_all = await asyncio.gather(*tasks)
        nodes_all = cast(List[Optional[NodeWithScore]], nodes_all)

        # filter out sub questions that failed
        nodes = list(filter(None, nodes_all))

        return await self._response_synthesizer.asynthesize(
            query_bundle=query_bundle,
            nodes=nodes,
        )

    async def _aquery_subq(
        self, sub_q: SubQuestion, color: Optional[str] = None
    ) -> Optional[NodeWithScore]:
        try:
            question = sub_q.sub_question
            query_engine = self._query_engines[sub_q.tool_name]

            if self._verbose:
                print_text(f"[{sub_q.tool_name}] Q: {question}\n", color=color)

            response = await query_engine.aquery(question)
            response_text = str(response)
            node_text = f"Sub question: {question}\nResponse: {response_text}"

            if self._verbose:
                print_text(f"[{sub_q.tool_name}] A: {response_text}\n", color=color)

            return NodeWithScore(Node(text=node_text))
        except ValueError:
            logger.warn(f"[{sub_q.tool_name}] Failed to run {question}")
            return None

    def _query_subq(
        self, sub_q: SubQuestion, color: Optional[str] = None
    ) -> Optional[NodeWithScore]:
        try:
            question = sub_q.sub_question
            query_engine = self._query_engines[sub_q.tool_name]

            if self._verbose:
                print_text(f"[{sub_q.tool_name}] Q: {question}\n", color=color)

            response = query_engine.query(question)
            response_text = str(response)
            node_text = f"Sub question: {question}\nResponse: {response_text}"

            if self._verbose:
                print_text(f"[{sub_q.tool_name}] A: {response_text}\n", color=color)

            return NodeWithScore(Node(text=node_text))
        except ValueError:
            logger.warn(f"[{sub_q.tool_name}] Failed to run {question}")
            return None
