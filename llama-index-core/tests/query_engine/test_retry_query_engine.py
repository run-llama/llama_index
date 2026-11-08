from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.response.schema import RESPONSE_TYPE, Response
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from llama_index.core.callbacks.schema import CBEventType
from llama_index.core.evaluation.base import EvaluationResult
from llama_index.core.indices.query.query_transform.feedback_transform import (
    FeedbackQueryTransformation,
)
from llama_index.core.prompts.mixin import PromptMixinType
from llama_index.core.query_engine.retry_query_engine import (
    RetryGuidelineQueryEngine,
    RetryQueryEngine,
)
from llama_index.core.schema import QueryBundle


class _DummyQueryEngine(BaseQueryEngine):
    def __init__(self, callback_manager: Optional[CallbackManager] = None) -> None:
        super().__init__(callback_manager)
        self.queries: List[str] = []

    def _get_prompt_modules(self) -> PromptMixinType:
        return {}

    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        self.queries.append(query_bundle.query_str)
        return Response(response=f"answer:{query_bundle.query_str}")

    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        return self._query(query_bundle)


class _RecordingCallbackHandler(BaseCallbackHandler):
    def __init__(self) -> None:
        super().__init__(event_starts_to_ignore=[], event_ends_to_ignore=[])
        self.trace_starts: int = 0

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        return event_id

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        pass

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        if trace_id == "query":
            self.trace_starts += 1

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        pass


def test_retry_guideline_query_engine_preserves_custom_query_transformer() -> None:
    """Verify RetryGuidelineQueryEngine preserves custom query_transformer across all retry steps."""
    base_engine = _DummyQueryEngine()
    guideline_evaluator = MagicMock()
    guideline_evaluator.evaluate_response.return_value = EvaluationResult(
        query="q", response="r", passing=False, feedback="try again"
    )

    transform_calls: List[str] = []

    class CustomFeedbackTransform(FeedbackQueryTransformation):
        def _run(self, query_bundle: QueryBundle, metadata: Dict) -> QueryBundle:
            transform_calls.append(query_bundle.query_str)
            return QueryBundle(query_str=f"{query_bundle.query_str}_custom_{len(transform_calls)}")

    custom_transformer = CustomFeedbackTransform(resynthesize_query=False)
    retry_engine = RetryGuidelineQueryEngine(
        query_engine=base_engine,
        guideline_evaluator=guideline_evaluator,
        resynthesize_query=False,
        max_retries=3,
        query_transformer=custom_transformer,
    )

    response = retry_engine.query("initial")
    assert len(transform_calls) == 3
    assert base_engine.queries == [
        "initial",
        "initial_custom_1",
        "initial_custom_1_custom_2",
        "initial_custom_1_custom_2_custom_3",
    ]
    assert str(response) == "answer:initial_custom_1_custom_2_custom_3"


def test_retry_query_engine_preserves_callback_manager() -> None:
    """Verify RetryQueryEngine forwards callback_manager to recursive retry instances."""
    handler = _RecordingCallbackHandler()
    cb_manager = CallbackManager([handler])
    base_engine = _DummyQueryEngine(callback_manager=cb_manager)
    evaluator = MagicMock()
    evaluator.evaluate_response.return_value = EvaluationResult(
        query="q", response="r", passing=False, feedback="retry"
    )

    retry_engine = RetryQueryEngine(
        query_engine=base_engine,
        evaluator=evaluator,
        max_retries=2,
        callback_manager=cb_manager,
    )

    retry_engine._query(QueryBundle("hello"))
    assert handler.trace_starts == 1


def test_retry_source_query_engine_preserves_callback_manager(monkeypatch: Any) -> None:
    """Verify RetrySourceQueryEngine forwards callback_manager to recursive retry instances."""
    from llama_index.core import Settings
    from llama_index.core.llms.mock import MockLLM
    from llama_index.core.query_engine.retry_source_query_engine import (
        RetrySourceQueryEngine,
    )
    from llama_index.core.schema import NodeWithScore, TextNode

    monkeypatch.setattr(Settings, "_llm", MockLLM())

    class _SourceEngine(_DummyQueryEngine):
        def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
            return Response(
                response="answer",
                source_nodes=[NodeWithScore(node=TextNode(text="ctx"), score=1.0)],
            )

    handler = _RecordingCallbackHandler()
    cb_manager = CallbackManager([handler])
    evaluator = MagicMock()
    evaluator.evaluate_response.return_value = EvaluationResult(
        query="q", response="r", passing=False, feedback="retry"
    )
    evaluator.evaluate.return_value = EvaluationResult(passing=True)

    retry_engine = RetrySourceQueryEngine(
        query_engine=_SourceEngine(),
        evaluator=evaluator,
        llm=MockLLM(),
        max_retries=1,
        callback_manager=cb_manager,
    )
    retry_engine._query(QueryBundle("initial"))
    assert handler.trace_starts == 1
