from unittest.mock import AsyncMock, MagicMock

import pytest

from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.response.schema import Response
from llama_index.core.callbacks import CallbackManager
from llama_index.core.query_engine import MultiStepQueryEngine
from llama_index.core.schema import QueryBundle


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("early_stopping", [False, True])
@pytest.mark.parametrize("num_steps", [0, 3])
async def test_multistep_honors_early_stopping(
    use_async: bool, early_stopping: bool, num_steps: int
) -> None:
    query_engine = MagicMock(spec=BaseQueryEngine)
    query_engine.callback_manager = CallbackManager()
    query_engine.query.return_value = Response(response="Sub-answer")
    transform = MagicMock(return_value=QueryBundle("Follow-up question"))
    synthesizer = MagicMock()
    synthesizer.synthesize.return_value = Response(response="Final answer")
    synthesizer.asynthesize = AsyncMock(return_value=Response(response="Final answer"))
    stop_fn = MagicMock(return_value=True)
    engine = MultiStepQueryEngine(
        query_engine=query_engine,
        query_transform=transform,
        response_synthesizer=synthesizer,
        early_stopping=early_stopping,
        num_steps=num_steps,
        stop_fn=stop_fn,
    )

    response = (
        await engine.aquery("Question") if use_async else engine.query("Question")
    )

    expected_steps = 0 if early_stopping else num_steps
    assert query_engine.query.call_count == expected_steps
    assert len(response.metadata["sub_qa"]) == expected_steps
    assert str(response) == "Final answer"
    assert stop_fn.call_count == (1 if early_stopping and num_steps else 0)
