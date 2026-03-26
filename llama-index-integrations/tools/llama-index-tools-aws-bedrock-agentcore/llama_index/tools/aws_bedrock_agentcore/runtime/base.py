import copy
import logging
from collections.abc import Sequence
from typing import Any, AsyncGenerator, Dict, Optional

from starlette.exceptions import HTTPException
from starlette.middleware import Middleware
from starlette.types import Lifespan

from bedrock_agentcore.runtime import BedrockAgentCoreApp, RequestContext
from llama_index.core.agent.workflow.workflow_events import (
    AgentOutput,
    AgentStream,
    ToolCall,
    ToolCallResult,
)

logger = logging.getLogger(__name__)


class AgentCoreRuntime:
    """Serves a LlamaIndex agent via BedrockAgentCoreApp (POST /invocations, GET /ping)."""

    def __init__(
        self,
        agent: Any,
        stream: bool = True,
        port: int = 8080,
        host: Optional[str] = None,
        debug: bool = False,
        memory: Optional[Any] = None,
        lifespan: Optional[Lifespan] = None,
        middleware: Optional[Sequence[Middleware]] = None,
    ):
        self._agent = agent
        self._stream = stream
        self._port = port
        self._host = host
        self._memory = memory
        self._app = BedrockAgentCoreApp(
            debug=debug, lifespan=lifespan, middleware=middleware
        )

        # Register entrypoint using closure wrappers (not bound methods).
        # entrypoint() sets func.run attr which fails on bound methods.
        # Closures also preserve isasyncgenfunction() detection for streaming.
        runtime = self
        if stream:

            async def streaming_entrypoint(
                payload: dict, context: RequestContext
            ) -> AsyncGenerator[dict, None]:
                # Validate eagerly (before the first yield) so that
                # HTTPException propagates before streaming begins.
                prompt = runtime._extract_prompt(payload)
                memory = runtime._get_memory(context)
                async for chunk in runtime._stream_events(prompt, memory):
                    yield chunk

            self._app.entrypoint(streaming_entrypoint)
        else:

            async def non_streaming_entrypoint(
                payload: dict, context: RequestContext
            ) -> dict:
                return await runtime._non_streaming_handler(payload, context)

            self._app.entrypoint(non_streaming_entrypoint)

    @classmethod
    def serve(cls, agent: Any, **kwargs: Any) -> None:
        """Create runtime and start server in one call."""
        runtime = cls(agent=agent, **kwargs)
        runtime.run()

    def run(self, **kwargs: Any) -> None:
        """Start uvicorn server."""
        self._app.run(port=self._port, host=self._host, **kwargs)

    @property
    def app(self) -> BedrockAgentCoreApp:
        """Expose for ASGI mounting or testing."""
        return self._app

    @staticmethod
    def _extract_prompt(payload: dict) -> str:
        """Normalize payload to user message string."""
        prompt = payload.get("prompt") or payload.get("message") or payload.get("input")
        if isinstance(prompt, dict):
            prompt = prompt.get("prompt")
        if isinstance(prompt, str):
            prompt = prompt.strip()
        if not prompt or not isinstance(prompt, (str, int, float)):
            raise HTTPException(
                status_code=400,
                detail="Request must include 'prompt', 'message', or 'input' field"
                " with a string value",
            )
        return str(prompt)

    def _get_memory(self, context: Optional[RequestContext] = None) -> Optional[Any]:
        """Return per-request memory copy with session_id from AgentCore context."""
        if self._memory is None:
            return None
        if context and context.session_id and hasattr(self._memory, "_context"):
            if hasattr(self._memory._context, "session_id"):
                memory = copy.copy(self._memory)
                memory._context = copy.copy(self._memory._context)
                memory._context.session_id = context.session_id
                return memory
        return self._memory

    async def _non_streaming_handler(
        self, payload: dict, context: RequestContext
    ) -> dict:
        """Handle non-streaming invocation. Returns JSON response."""
        prompt = self._extract_prompt(payload)
        memory = self._get_memory(context)
        handler = self._agent.run(user_msg=prompt, memory=memory)
        result = await handler
        return {"response": str(result)}

    async def _stream_events(
        self, prompt: str, memory: Optional[Any]
    ) -> AsyncGenerator[dict, None]:
        """
        Yield SSE event dicts for a validated prompt.

        Callers should validate the prompt (via _extract_prompt) before
        calling this method so that validation errors are raised eagerly.
        """
        handler = self._agent.run(user_msg=prompt, memory=memory)

        try:
            async for event in handler.stream_events():
                if isinstance(event, AgentStream):
                    ev: Dict[str, Any] = {
                        "event": "agent_stream",
                        "delta": event.delta,
                        "response": event.response,
                    }
                    if event.thinking_delta:
                        ev["thinking_delta"] = event.thinking_delta
                    yield ev
                elif isinstance(event, ToolCall):
                    yield {
                        "event": "tool_call",
                        "tool_name": event.tool_name,
                        "tool_kwargs": event.tool_kwargs,
                    }
                elif isinstance(event, ToolCallResult):
                    yield {
                        "event": "tool_result",
                        "tool_name": event.tool_name,
                        "tool_output": str(event.tool_output),
                    }
                elif isinstance(event, AgentOutput):
                    yield {"event": "done", "response": str(event.response)}
                else:
                    logger.debug(
                        "Ignoring unknown event type: %s", type(event).__name__
                    )
        except Exception as e:
            logger.exception("Error during streaming")
            yield {"event": "error", "message": str(e)}
            return

        # Await handler to ensure background tasks complete (memory flush, etc.)
        try:
            await handler
        except Exception:
            logger.exception("Error awaiting handler completion")
