import inspect
from unittest.mock import MagicMock, patch

import pytest
from starlette.exceptions import HTTPException

from llama_index.tools.aws_bedrock_agentcore.runtime.base import AgentCoreRuntime


class MockWorkflowHandler:
    """Mimics LlamaIndex WorkflowHandler: awaitable + stream_events()."""

    def __init__(self, result="done", events=None, raise_on_await=None):
        self._result = result
        self._events = events or []
        self._raise_on_await = raise_on_await

    def __await__(self):
        async def _resolve():
            if self._raise_on_await:
                raise self._raise_on_await
            return self._result

        return _resolve().__await__()

    async def stream_events(self):
        for event in self._events:
            yield event


class ErrorStreamWorkflowHandler(MockWorkflowHandler):
    """Raises mid-stream after yielding some events."""

    def __init__(self, events_before_error=None, error=None, **kwargs):
        super().__init__(**kwargs)
        self._events_before_error = events_before_error or []
        self._error = error or RuntimeError("stream failed")

    async def stream_events(self):
        for event in self._events_before_error:
            yield event
        raise self._error


class TestExtractPrompt:
    def test_prompt_field(self):
        assert AgentCoreRuntime._extract_prompt({"prompt": "hello"}) == "hello"

    def test_message_field(self):
        assert AgentCoreRuntime._extract_prompt({"message": "hello"}) == "hello"

    def test_input_field(self):
        assert AgentCoreRuntime._extract_prompt({"input": "hello"}) == "hello"

    def test_nested_prompt(self):
        assert (
            AgentCoreRuntime._extract_prompt({"input": {"prompt": "hello"}}) == "hello"
        )

    def test_priority_order(self):
        payload = {"prompt": "first", "message": "second", "input": "third"}
        assert AgentCoreRuntime._extract_prompt(payload) == "first"

    def test_missing_raises_400(self):
        with pytest.raises(HTTPException) as exc_info:
            AgentCoreRuntime._extract_prompt({"foo": "bar"})
        assert exc_info.value.status_code == 400

    def test_empty_raises_400(self):
        with pytest.raises(HTTPException) as exc_info:
            AgentCoreRuntime._extract_prompt({})
        assert exc_info.value.status_code == 400

    def test_empty_string_raises_400(self):
        with pytest.raises(HTTPException) as exc_info:
            AgentCoreRuntime._extract_prompt({"prompt": ""})
        assert exc_info.value.status_code == 400

    def test_whitespace_only_raises_400(self):
        with pytest.raises(HTTPException) as exc_info:
            AgentCoreRuntime._extract_prompt({"prompt": "   "})
        assert exc_info.value.status_code == 400

    def test_list_value_raises_400(self):
        with pytest.raises(HTTPException) as exc_info:
            AgentCoreRuntime._extract_prompt({"prompt": ["a", "b"]})
        assert exc_info.value.status_code == 400

    def test_numeric_value_coerced(self):
        assert AgentCoreRuntime._extract_prompt({"prompt": 42}) == "42"

    def test_strips_whitespace(self):
        assert AgentCoreRuntime._extract_prompt({"prompt": "  hello  "}) == "hello"


class TestNonStreamingHandler:
    @pytest.mark.asyncio
    async def test_returns_response_dict(self):
        mock_agent = MagicMock()
        mock_agent.run.return_value = MockWorkflowHandler(result="test response")

        with patch(
            "llama_index.tools.aws_bedrock_agentcore.runtime.base.BedrockAgentCoreApp"
        ):
            runtime = AgentCoreRuntime(agent=mock_agent, stream=False)

        mock_context = MagicMock()
        mock_context.session_id = None

        result = await runtime._non_streaming_handler({"prompt": "hello"}, mock_context)

        assert result == {"response": "test response"}
        mock_agent.run.assert_called_once_with(user_msg="hello", memory=None)


class TestStreamingHandler:
    @pytest.mark.asyncio
    async def test_yields_events(self):
        from llama_index.core.agent.workflow.workflow_events import (
            AgentOutput,
            AgentStream,
            ToolCall,
            ToolCallResult,
        )
        from llama_index.core.llms import ChatMessage
        from llama_index.core.tools.types import ToolOutput

        tool_output = ToolOutput(
            tool_name="my_tool",
            raw_input={"arg": "val"},
            raw_output="result",
        )
        tool_output.content = "result"

        events = [
            AgentStream(
                delta="hi",
                response="hi",
                current_agent_name="agent",
                thinking_delta=None,
            ),
            ToolCall(
                tool_name="my_tool",
                tool_kwargs={"arg": "val"},
                tool_id="t1",
            ),
            ToolCallResult(
                tool_name="my_tool",
                tool_kwargs={"arg": "val"},
                tool_id="t1",
                tool_output=tool_output,
                return_direct=False,
            ),
            AgentOutput(
                response=ChatMessage(role="assistant", content="done"),
                current_agent_name="agent",
                tool_calls=[],
                raw={},
            ),
        ]

        mock_agent = MagicMock()
        mock_agent.run.return_value = MockWorkflowHandler(events=events)

        with patch(
            "llama_index.tools.aws_bedrock_agentcore.runtime.base.BedrockAgentCoreApp"
        ):
            runtime = AgentCoreRuntime(agent=mock_agent, stream=True)

        mock_context = MagicMock()
        mock_context.session_id = None

        collected = []
        async for chunk in runtime._streaming_handler(
            {"message": "hello"}, mock_context
        ):
            collected.append(chunk)

        assert len(collected) == 4
        assert collected[0]["event"] == "agent_stream"
        assert collected[0]["delta"] == "hi"
        assert "thinking_delta" not in collected[0]
        assert collected[1]["event"] == "tool_call"
        assert collected[1]["tool_name"] == "my_tool"
        assert collected[2]["event"] == "tool_result"
        assert collected[2]["tool_output"] == "result"
        assert collected[3]["event"] == "done"

    @pytest.mark.asyncio
    async def test_thinking_delta_included(self):
        from llama_index.core.agent.workflow.workflow_events import (
            AgentOutput,
            AgentStream,
        )
        from llama_index.core.llms import ChatMessage

        events = [
            AgentStream(
                delta="",
                response="",
                current_agent_name="agent",
                thinking_delta="Let me think...",
            ),
            AgentOutput(
                response=ChatMessage(role="assistant", content="done"),
                current_agent_name="agent",
                tool_calls=[],
                raw={},
            ),
        ]

        mock_agent = MagicMock()
        mock_agent.run.return_value = MockWorkflowHandler(events=events)

        with patch(
            "llama_index.tools.aws_bedrock_agentcore.runtime.base.BedrockAgentCoreApp"
        ):
            runtime = AgentCoreRuntime(agent=mock_agent, stream=True)

        mock_context = MagicMock()
        mock_context.session_id = None

        collected = []
        async for chunk in runtime._streaming_handler(
            {"prompt": "think"}, mock_context
        ):
            collected.append(chunk)

        assert collected[0]["thinking_delta"] == "Let me think..."


class TestStreamingErrorHandling:
    @pytest.mark.asyncio
    async def test_error_mid_stream_yields_error_event(self):
        from llama_index.core.agent.workflow.workflow_events import AgentStream

        events_before = [
            AgentStream(
                delta="hi",
                response="hi",
                current_agent_name="agent",
                thinking_delta=None,
            ),
        ]

        mock_agent = MagicMock()
        mock_agent.run.return_value = ErrorStreamWorkflowHandler(
            events_before_error=events_before,
            error=RuntimeError("LLM connection lost"),
        )

        with patch(
            "llama_index.tools.aws_bedrock_agentcore.runtime.base.BedrockAgentCoreApp"
        ):
            runtime = AgentCoreRuntime(agent=mock_agent, stream=True)

        mock_context = MagicMock()
        mock_context.session_id = None

        collected = []
        async for chunk in runtime._streaming_handler(
            {"prompt": "hello"}, mock_context
        ):
            collected.append(chunk)

        assert len(collected) == 2
        assert collected[0]["event"] == "agent_stream"
        assert collected[1]["event"] == "error"
        assert "LLM connection lost" in collected[1]["message"]

    @pytest.mark.asyncio
    async def test_error_before_any_events(self):
        mock_agent = MagicMock()
        mock_agent.run.return_value = ErrorStreamWorkflowHandler(
            error=ValueError("bad input"),
        )

        with patch(
            "llama_index.tools.aws_bedrock_agentcore.runtime.base.BedrockAgentCoreApp"
        ):
            runtime = AgentCoreRuntime(agent=mock_agent, stream=True)

        mock_context = MagicMock()
        mock_context.session_id = None

        collected = []
        async for chunk in runtime._streaming_handler(
            {"prompt": "hello"}, mock_context
        ):
            collected.append(chunk)

        assert len(collected) == 1
        assert collected[0]["event"] == "error"
        assert "bad input" in collected[0]["message"]

    @pytest.mark.asyncio
    async def test_await_handler_error_does_not_crash(self):
        mock_agent = MagicMock()
        mock_agent.run.return_value = MockWorkflowHandler(
            events=[], raise_on_await=RuntimeError("flush failed")
        )

        with patch(
            "llama_index.tools.aws_bedrock_agentcore.runtime.base.BedrockAgentCoreApp"
        ):
            runtime = AgentCoreRuntime(agent=mock_agent, stream=True)

        mock_context = MagicMock()
        mock_context.session_id = None

        collected = []
        async for chunk in runtime._streaming_handler(
            {"prompt": "hello"}, mock_context
        ):
            collected.append(chunk)

        # Should complete without raising (error is logged, not propagated)
        assert len(collected) == 0


class TestSessionIdPropagation:
    def test_session_id_set_on_copy(self):
        mock_memory = MagicMock()
        mock_memory._context = MagicMock()
        mock_memory._context.session_id = "old-session"

        with patch(
            "llama_index.tools.aws_bedrock_agentcore.runtime.base.BedrockAgentCoreApp"
        ):
            runtime = AgentCoreRuntime(
                agent=MagicMock(), stream=False, memory=mock_memory
            )

        mock_context = MagicMock()
        mock_context.session_id = "new-session-123"

        result = runtime._get_memory(mock_context)

        # Should return a copy, not the original
        assert result is not mock_memory
        assert result._context.session_id == "new-session-123"
        # Original memory should be unchanged
        assert mock_memory._context.session_id == "old-session"

    def test_no_memory_returns_none(self):
        with patch(
            "llama_index.tools.aws_bedrock_agentcore.runtime.base.BedrockAgentCoreApp"
        ):
            runtime = AgentCoreRuntime(agent=MagicMock(), stream=False)

        assert runtime._get_memory(MagicMock()) is None

    def test_no_session_id_returns_original(self):
        mock_memory = MagicMock()
        mock_memory._context = MagicMock()
        mock_memory._context.session_id = "existing"

        with patch(
            "llama_index.tools.aws_bedrock_agentcore.runtime.base.BedrockAgentCoreApp"
        ):
            runtime = AgentCoreRuntime(
                agent=MagicMock(), stream=False, memory=mock_memory
            )

        mock_context = MagicMock()
        mock_context.session_id = None

        result = runtime._get_memory(mock_context)
        assert result is mock_memory


class TestServe:
    def test_serve_calls_run(self):
        with patch(
            "llama_index.tools.aws_bedrock_agentcore.runtime.base.BedrockAgentCoreApp"
        ):
            with patch.object(AgentCoreRuntime, "run") as mock_run:
                AgentCoreRuntime.serve(MagicMock(), port=9090, debug=True)
                mock_run.assert_called_once()


class TestEntrypointClosure:
    def test_streaming_entrypoint_is_async_gen_function(self):
        with patch(
            "llama_index.tools.aws_bedrock_agentcore.runtime.base.BedrockAgentCoreApp"
        ) as MockApp:
            mock_app = MagicMock()
            MockApp.return_value = mock_app

            AgentCoreRuntime(agent=MagicMock(), stream=True)

            registered_handler = mock_app.entrypoint.call_args[0][0]
            assert inspect.isasyncgenfunction(registered_handler)
            assert not inspect.ismethod(registered_handler)

    def test_non_streaming_entrypoint_is_coroutine_function(self):
        with patch(
            "llama_index.tools.aws_bedrock_agentcore.runtime.base.BedrockAgentCoreApp"
        ) as MockApp:
            mock_app = MagicMock()
            MockApp.return_value = mock_app

            AgentCoreRuntime(agent=MagicMock(), stream=False)

            registered_handler = mock_app.entrypoint.call_args[0][0]
            assert inspect.iscoroutinefunction(registered_handler)
            assert not inspect.ismethod(registered_handler)


class TestAppProperty:
    def test_exposes_app(self):
        with patch(
            "llama_index.tools.aws_bedrock_agentcore.runtime.base.BedrockAgentCoreApp"
        ) as MockApp:
            mock_app = MagicMock()
            MockApp.return_value = mock_app

            runtime = AgentCoreRuntime(agent=MagicMock())
            assert runtime.app is mock_app


class TestConstructorPassthrough:
    def test_lifespan_passed_to_app(self):
        async def my_lifespan(app):
            yield

        with patch(
            "llama_index.tools.aws_bedrock_agentcore.runtime.base.BedrockAgentCoreApp"
        ) as MockApp:
            mock_app = MagicMock()
            MockApp.return_value = mock_app

            AgentCoreRuntime(agent=MagicMock(), lifespan=my_lifespan)

            MockApp.assert_called_once_with(
                debug=False, lifespan=my_lifespan, middleware=None
            )

    def test_middleware_passed_to_app(self):
        from starlette.middleware import Middleware
        from starlette.middleware.gzip import GZipMiddleware

        mw = [Middleware(GZipMiddleware)]

        with patch(
            "llama_index.tools.aws_bedrock_agentcore.runtime.base.BedrockAgentCoreApp"
        ) as MockApp:
            mock_app = MagicMock()
            MockApp.return_value = mock_app

            AgentCoreRuntime(agent=MagicMock(), middleware=mw)

            MockApp.assert_called_once_with(debug=False, lifespan=None, middleware=mw)

    def test_all_app_params_passed(self):
        async def my_lifespan(app):
            yield

        from starlette.middleware import Middleware
        from starlette.middleware.gzip import GZipMiddleware

        mw = [Middleware(GZipMiddleware)]

        with patch(
            "llama_index.tools.aws_bedrock_agentcore.runtime.base.BedrockAgentCoreApp"
        ) as MockApp:
            mock_app = MagicMock()
            MockApp.return_value = mock_app

            AgentCoreRuntime(
                agent=MagicMock(),
                debug=True,
                lifespan=my_lifespan,
                middleware=mw,
            )

            MockApp.assert_called_once_with(
                debug=True, lifespan=my_lifespan, middleware=mw
            )
