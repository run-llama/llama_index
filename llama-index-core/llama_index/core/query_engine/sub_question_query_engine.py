import logging
from typing import List, Optional, Sequence

from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.llms.llm import LLM
from llama_index.core.prompts.mixin import PromptMixinType
from llama_index.core.question_gen.llm_generators import LLMQuestionGenerator
from llama_index.core.question_gen.types import BaseQuestionGenerator, SubQuestion
from llama_index.core.response_synthesizers import (
    BaseSynthesizer,
    get_response_synthesizer,
)
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.settings import Settings
from llama_index.core.tools.query_engine import QueryEngineTool
from llama_index.core.utils import get_color_mapping, print_text
from llama_index.core.workflow import (
    Context,
    StartEvent,
    StopEvent,
    step,
    Event,
)

logger = logging.getLogger(__name__)


class SubQuestionAnswerPair(BaseModel):
    """
    Pair of the sub question and optionally its answer (if its been answered yet).
    """

    sub_q: SubQuestion
    answer: Optional[str] = None
    sources: List[NodeWithScore] = Field(default_factory=list)


class SubQuestionEvent(Event):
    """Sub question event."""

    query: SubQuestion
    color: Optional[str] = None


class SubQuestionAnswerEvent(Event):
    """
    Pair of the sub question and optionally its answer (if its been answered yet).
    """

    qa_pair: SubQuestionAnswerPair


class SubQuestionQueryEngine(BaseQueryEngine):
    """Sub question query engine.

    A query engine that breaks down a complex query (e.g. compare and contrast) into
        many sub questions and their target query engine for execution.
        After executing all sub questions, all responses are gathered and sent to
        response synthesizer to produce the final response.

    Args:
        question_gen (BaseQuestionGenerator): A module for generating sub questions
            given a complex question and tools.
        response_synthesizer (BaseSynthesizer): A response synthesizer for
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
        response_synthesizer: BaseSynthesizer,
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

    def _get_prompt_modules(self) -> PromptMixinType:
        """Get prompt sub-modules."""
        return {
            "question_gen": self._question_gen,
            "response_synthesizer": self._response_synthesizer,
        }

    @classmethod
    def from_defaults(
        cls,
        query_engine_tools: Sequence[QueryEngineTool],
        llm: Optional[LLM] = None,
        question_gen: Optional[BaseQuestionGenerator] = None,
        response_synthesizer: Optional[BaseSynthesizer] = None,
        verbose: bool = True,
        use_async: bool = True,
    ) -> "SubQuestionQueryEngine":
        callback_manager = Settings.callback_manager
        if len(query_engine_tools) > 0:
            callback_manager = query_engine_tools[0].query_engine.callback_manager

        llm = llm or Settings.llm
        if question_gen is None:
            try:
                from llama_index.question_gen.openai import (
                    OpenAIQuestionGenerator,
                )  # pants: no-infer-dep

                # try to use OpenAI function calling based question generator.
                # if incompatible, use general LLM question generator
                question_gen = OpenAIQuestionGenerator.from_defaults(llm=llm)

            except ImportError as e:
                raise ImportError(
                    "`llama-index-question-gen-openai` package cannot be found. "
                    "Please install it by using `pip install `llama-index-question-gen-openai`"
                )
            except ValueError:
                question_gen = LLMQuestionGenerator.from_defaults(llm=llm)

        synth = response_synthesizer or get_response_synthesizer(
            llm=llm,
            callback_manager=callback_manager,
            use_async=use_async,
        )

        return cls(
            question_gen,
            synth,
            query_engine_tools,
            callback_manager=callback_manager,
            verbose=verbose,
            use_async=use_async,
        )

    @step
    async def generate_sub_questions(
        self, ctx: Context, ev: StartEvent
    ) -> SubQuestionEvent | None:
        """Generate sub questions."""
        sub_questions = await self._question_gen.agenerate(
            self._metadatas, ev.query_bundle
        )
        colors = get_color_mapping([str(i) for i in range(len(sub_questions))])

        ctx.data["num_sub_questions"] = len(sub_questions)
        ctx.data["query_bundle"] = ev.query_bundle

        for ind, sub_q in enumerate(sub_questions):
            self.send_event(SubQuestionEvent(query=sub_q, color=colors[str(ind)]))

        if self._verbose:
            print_text(f"Generated {len(sub_questions)} sub questions.\n")

        return None

    @step
    async def query_sub_question(
        self, ctx: Context, ev: SubQuestionEvent
    ) -> SubQuestionAnswerEvent | None:
        """Query sub question."""
        qa_pair = await self._aquery_subq(ev.query, color=ev.color)
        return SubQuestionAnswerEvent(qa_pair=qa_pair)

    @step
    async def gather(self, ctx: Context, ev: SubQuestionAnswerEvent) -> StopEvent:
        """Gather sub questions."""
        subq_events = ctx.collect_events(
            ev, ctx.data["num_sub_questions"] * [SubQuestionAnswerEvent]
        )
        if not subq_events:
            return None
        # filter out sub questions that failed
        qa_pairs = [ev.qa_pair for ev in subq_events if ev.qa_pair is not None]
        nodes = [self._construct_node(pair) for pair in qa_pairs]
        source_nodes = [node for qa_pair in qa_pairs for node in qa_pair.sources]
        response = await self._response_synthesizer.asynthesize(
            query=ctx.data["query_bundle"],
            nodes=nodes,
            additional_source_nodes=source_nodes,
        )
        return StopEvent(result=response)

    def _construct_node(self, qa_pair: SubQuestionAnswerPair) -> NodeWithScore:
        node_text = (
            f"Sub question: {qa_pair.sub_q.sub_question}\nResponse: {qa_pair.answer}"
        )
        return NodeWithScore(node=TextNode(text=node_text))

    async def _aquery_subq(
        self, sub_q: SubQuestion, color: Optional[str] = None
    ) -> Optional[SubQuestionAnswerPair]:
        try:
            with self.callback_manager.event(
                CBEventType.SUB_QUESTION,
                payload={EventPayload.SUB_QUESTION: SubQuestionAnswerPair(sub_q=sub_q)},
            ) as event:
                question = sub_q.sub_question
                query_engine = self._query_engines[sub_q.tool_name]

                if self._verbose:
                    print_text(f"[{sub_q.tool_name}] Q: {question}\n", color=color)

                response = await query_engine.aquery(question)
                response_text = str(response)

                if self._verbose:
                    print_text(f"[{sub_q.tool_name}] A: {response_text}\n", color=color)

                qa_pair = SubQuestionAnswerPair(
                    sub_q=sub_q, answer=response_text, sources=response.source_nodes
                )

                event.on_end(payload={EventPayload.SUB_QUESTION: qa_pair})

            return qa_pair
        except ValueError:
            logger.warning(f"[{sub_q.tool_name}] Failed to run {question}")
            return None

    def _query_subq(
        self, sub_q: SubQuestion, color: Optional[str] = None
    ) -> Optional[SubQuestionAnswerPair]:
        try:
            with self.callback_manager.event(
                CBEventType.SUB_QUESTION,
                payload={EventPayload.SUB_QUESTION: SubQuestionAnswerPair(sub_q=sub_q)},
            ) as event:
                question = sub_q.sub_question
                query_engine = self._query_engines[sub_q.tool_name]

                if self._verbose:
                    print_text(f"[{sub_q.tool_name}] Q: {question}\n", color=color)

                response = query_engine.query(question)
                response_text = str(response)

                if self._verbose:
                    print_text(f"[{sub_q.tool_name}] A: {response_text}\n", color=color)

                qa_pair = SubQuestionAnswerPair(
                    sub_q=sub_q, answer=response_text, sources=response.source_nodes
                )

                event.on_end(payload={EventPayload.SUB_QUESTION: qa_pair})

            return qa_pair
        except ValueError:
            logger.warning(f"[{sub_q.tool_name}] Failed to run {question}")
            return None
