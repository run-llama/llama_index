"""
Regression tests for initial_state isolation across AGUIChatWorkflow instances.

The default workflow factory is meant to give each request its own isolated
workflow, but it used to hand the operator's ``initial_state`` dict to every
workflow it produced (stored by reference in ``__init__``), and the ``chat``
step derived per-request state with a shallow ``dict.copy()``. Both let state
leak across requests. These tests pin down the isolation contract.
"""

import asyncio
from typing import Any, AsyncGenerator

from llama_index.core.base.llms.types import ChatMessage, ChatResponse, LLMMetadata
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.protocols.ag_ui.router import get_default_workflow_factory


class _FakeLLM(FunctionCallingLLM):
    """
    Minimal function-calling LLM that streams one assistant message.

    Only the methods the ``chat`` step actually calls are implemented; the
    remaining abstract methods are never invoked in these tests.
    """

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(is_function_calling_model=True, model_name="fake")

    async def astream_chat_with_tools(
        self, *args: Any, **kwargs: Any
    ) -> AsyncGenerator[ChatResponse, None]:
        async def _gen() -> AsyncGenerator[ChatResponse, None]:
            yield ChatResponse(
                message=ChatMessage(role="assistant", content="ok"), delta="ok"
            )

        return _gen()

    def get_tool_calls_from_response(self, *args: Any, **kwargs: Any) -> list:
        return []

    def chat(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    async def achat(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def stream_chat(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    async def astream_chat(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def complete(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    async def acomplete(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def stream_complete(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    async def astream_complete(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def _prepare_chat_with_tools(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


def test_factory_isolates_initial_state_across_workflows() -> None:
    """Each workflow the factory produces must own a private initial_state."""
    operator_state = {"counter": 0, "items": [], "user": {"roles": []}}
    factory = get_default_workflow_factory(
        llm=_FakeLLM(), initial_state=operator_state, timeout=30
    )

    first = asyncio.run(factory())
    second = asyncio.run(factory())

    # Top-level dict must be a distinct object on each workflow and distinct
    # from the operator's config object.
    assert first.initial_state is not second.initial_state
    assert first.initial_state is not operator_state
    assert second.initial_state is not operator_state

    # A mutation on one workflow must not leak to the other or to the operator.
    first.initial_state["secret"] = "API-KEY-ALICE"
    assert "secret" not in second.initial_state
    assert "secret" not in operator_state


def test_factory_deep_copies_nested_state() -> None:
    """Nested mutable values must not be aliased across produced workflows."""
    operator_state = {"items": [], "user": {"roles": []}}
    factory = get_default_workflow_factory(
        llm=_FakeLLM(), initial_state=operator_state, timeout=30
    )

    first = asyncio.run(factory())
    second = asyncio.run(factory())

    assert first.initial_state["items"] is not second.initial_state["items"]
    assert first.initial_state["items"] is not operator_state["items"]
    assert first.initial_state["user"] is not operator_state["user"]

    first.initial_state["items"].append({"order_id": "ALICE-77"})
    first.initial_state["user"]["roles"].append("admin")
    assert second.initial_state["items"] == []
    assert second.initial_state["user"]["roles"] == []
    assert operator_state["items"] == []
    assert operator_state["user"]["roles"] == []


def test_chat_step_state_does_not_alias_initial_state() -> None:
    """
    When no request state is supplied, the per-request state derived in the
    ``chat`` step must be a deep copy of ``initial_state`` so that tool
    mutations during a run cannot bleed back into the template state.
    """
    from ag_ui.core import RunAgentInput, UserMessage

    async def _run() -> None:
        operator_state = {"counter": 0, "items": [], "user": {"roles": []}}
        factory = get_default_workflow_factory(
            llm=_FakeLLM(), initial_state=operator_state, timeout=30
        )
        workflow = await factory()

        run_input = RunAgentInput(
            thread_id="t1",
            run_id="r1",
            messages=[UserMessage(id="m1", role="user", content="hi")],
            tools=[],
            context=[],
            state=None,
            forwarded_props={},
        )

        handler = workflow.run(input_data=run_input)
        async for _ in handler.stream_events():
            pass
        await handler

        derived_state = await handler.ctx.store.get("state")
        assert derived_state["items"] is not workflow.initial_state["items"]
        assert derived_state["user"] is not workflow.initial_state["user"]

        # A mutation on the per-request state (as a tool would do via the
        # context store) must not bleed back into the template state.
        derived_state["items"].append({"order_id": "ALICE-77"})
        derived_state["user"]["roles"].append("admin")
        assert workflow.initial_state["items"] == []
        assert workflow.initial_state["user"]["roles"] == []

    asyncio.run(_run())
