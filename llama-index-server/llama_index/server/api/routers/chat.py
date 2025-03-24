import asyncio
import logging
from typing import Any, AsyncGenerator, Callable, Union

from fastapi import APIRouter

from llama_index.core.agent.workflow.workflow_events import AgentStream
from llama_index.core.workflow import StopEvent, Workflow
from llama_index.server.api.callbacks.stream_handler import StreamHandler
from llama_index.server.api.callbacks.suggest_next_questions import (
    SuggestNextQuestions,
)
from llama_index.server.api.models import ChatRequest
from llama_index.server.api.utils.vercel_stream import VercelStreamResponse


def chat_router(
    workflow_factory: Callable[[Any], Workflow],
    logger: logging.Logger,
):
    router = APIRouter(prefix="/chat")

    @router.post("")
    async def chat(request: ChatRequest):
        user_message = request.messages[-1].to_llamaindex_message()
        chat_history = [
            message.to_llamaindex_message() for message in request.messages[:-1]
        ]
        workflow = workflow_factory()
        workflow_handler = workflow.run(
            user_msg=user_message.content,
            chat_history=chat_history,
        )

        callbacks = []
        if request.config.next_question_suggestions:
            callbacks.append(SuggestNextQuestions(request))
        stream_handler = StreamHandler(
            workflow_handler=workflow_handler,
            callbacks=callbacks,
        )

        return VercelStreamResponse(
            content_generator=_stream_content(stream_handler, request, logger),
        )

    return router


async def _stream_content(handler, request: ChatRequest, logger: logging.Logger):
    async def _text_stream(event: Union[AgentStream, StopEvent]):
        if isinstance(event, AgentStream):
            if event.delta.strip():  # Only yield non-empty deltas
                yield event.delta
        elif isinstance(event, StopEvent):
            if isinstance(event.result, str):
                yield event.result
            elif isinstance(event.result, AsyncGenerator):
                async for chunk in event.result:
                    if isinstance(chunk, str):
                        yield chunk
                    elif (
                        hasattr(chunk, "delta") and chunk.delta.strip()
                    ):  # Only yield non-empty deltas
                        yield chunk.delta

    stream_started = False
    try:
        async for event in handler.stream_events():
            if not stream_started:
                # Start the stream with an empty message
                stream_started = True
                yield VercelStreamResponse.convert_text("")

            # Handle different types of events
            if isinstance(event, (AgentStream, StopEvent)):
                async for chunk in _text_stream(event):
                    handler.accumulate_text(chunk)
                    yield VercelStreamResponse.convert_text(chunk)
            elif isinstance(event, dict):
                yield VercelStreamResponse.convert_data(event)
            elif hasattr(event, "to_response"):
                event_response = event.to_response()
                yield VercelStreamResponse.convert_data(event_response)
            else:
                yield VercelStreamResponse.convert_data(event.model_dump())

    except asyncio.CancelledError:
        logger.warning("Client cancelled the request!")
        await handler.cancel_run()
    except Exception as e:
        logger.error(f"Error in stream response: {e}")
        yield VercelStreamResponse.convert_error(str(e))
        await handler.cancel_run()
