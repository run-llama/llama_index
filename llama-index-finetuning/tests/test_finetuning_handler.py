"""Tests for finetuning callback handlers."""

import json
import os
import tempfile
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from llama_index.core.callbacks.schema import CBEventType, EventPayload


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chat_message(role: str, content: str) -> Any:
    from llama_index.core.base.llms.types import ChatMessage, MessageRole

    role_map = {
        "system": MessageRole.SYSTEM,
        "user": MessageRole.USER,
        "assistant": MessageRole.ASSISTANT,
    }
    return ChatMessage(role=role_map[role], content=content)


def _make_llm_response(content: str) -> Any:
    """Return a mock LLM response object with a `.message` attribute."""
    msg = _make_chat_message("assistant", content)
    response = MagicMock()
    response.message = msg
    return response


# ---------------------------------------------------------------------------
# BaseFinetuningHandler / OpenAIFineTuningHandler – event recording
# ---------------------------------------------------------------------------


class TestOpenAIFineTuningHandler:
    def _get_handler(self):
        from llama_index.finetuning.callbacks.finetuning_handler import (
            OpenAIFineTuningHandler,
        )

        return OpenAIFineTuningHandler()

    def test_on_event_start_records_messages(self) -> None:
        handler = self._get_handler()
        messages = [
            _make_chat_message("user", "What is 2+2?"),
        ]
        handler.on_event_start(
            CBEventType.LLM,
            payload={EventPayload.MESSAGES: messages},
            event_id="evt1",
        )
        assert "evt1" in handler._finetuning_events
        assert len(handler._finetuning_events["evt1"]) == 1

    def test_on_event_start_prompt_creates_user_message(self) -> None:
        handler = self._get_handler()
        handler.on_event_start(
            CBEventType.LLM,
            payload={EventPayload.PROMPT: "Tell me a joke."},
            event_id="evt2",
        )
        assert "evt2" in handler._finetuning_events
        from llama_index.core.base.llms.types import MessageRole

        assert handler._finetuning_events["evt2"][0].role == MessageRole.USER

    def test_on_event_end_appends_response(self) -> None:
        handler = self._get_handler()
        messages = [_make_chat_message("user", "Hello")]
        handler.on_event_start(
            CBEventType.LLM,
            payload={EventPayload.MESSAGES: messages},
            event_id="evt3",
        )
        response_obj = _make_llm_response("Hi there!")
        handler.on_event_end(
            CBEventType.LLM,
            payload={EventPayload.RESPONSE: response_obj},
            event_id="evt3",
        )
        # Should have user message + assistant response
        assert len(handler._finetuning_events["evt3"]) == 2

    def test_on_event_end_string_response(self) -> None:
        handler = self._get_handler()
        messages = [_make_chat_message("user", "Hi")]
        handler.on_event_start(
            CBEventType.LLM,
            payload={EventPayload.MESSAGES: messages},
            event_id="evt4",
        )
        handler.on_event_end(
            CBEventType.LLM,
            payload={EventPayload.RESPONSE: "Hello!"},
            event_id="evt4",
        )
        from llama_index.core.base.llms.types import MessageRole

        last = handler._finetuning_events["evt4"][-1]
        assert last.role == MessageRole.ASSISTANT
        assert "Hello!" in str(last.content)

    def test_non_llm_events_are_ignored(self) -> None:
        handler = self._get_handler()
        handler.on_event_start(
            CBEventType.RETRIEVE,
            payload={},
            event_id="retrieve1",
        )
        assert "retrieve1" not in handler._finetuning_events

    def test_get_finetuning_events_structure(self) -> None:
        handler = self._get_handler()
        messages = [_make_chat_message("user", "Explain gravity.")]
        handler.on_event_start(
            CBEventType.LLM,
            payload={EventPayload.MESSAGES: messages},
            event_id="evtA",
        )
        response_obj = _make_llm_response("Gravity is a force.")
        handler.on_event_end(
            CBEventType.LLM,
            payload={EventPayload.RESPONSE: response_obj},
            event_id="evtA",
        )
        events = handler.get_finetuning_events()
        assert "evtA" in events
        assert "messages" in events["evtA"]
        assert "response" in events["evtA"]

    def test_save_finetuning_events_writes_jsonl(self) -> None:
        from llama_index.finetuning.callbacks.finetuning_handler import (
            OpenAIFineTuningHandler,
        )

        handler = OpenAIFineTuningHandler()
        messages = [
            _make_chat_message("system", "You are helpful."),
            _make_chat_message("user", "What is the capital of France?"),
        ]
        handler.on_event_start(
            CBEventType.LLM,
            payload={EventPayload.MESSAGES: messages},
            event_id="save_evt",
        )
        response_obj = _make_llm_response("Paris")
        handler.on_event_end(
            CBEventType.LLM,
            payload={EventPayload.RESPONSE: response_obj},
            event_id="save_evt",
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as tmp:
            tmp_path = tmp.name

        try:
            handler.save_finetuning_events(tmp_path)
            with open(tmp_path) as f:
                lines = [l for l in f.readlines() if l.strip()]
            assert len(lines) == 1
            record = json.loads(lines[0])
            assert "messages" in record
            # Should have system + user + assistant
            roles = [m["role"] for m in record["messages"]]
            assert "assistant" in roles
        finally:
            os.unlink(tmp_path)

    def test_multiple_events_saved(self) -> None:
        from llama_index.finetuning.callbacks.finetuning_handler import (
            OpenAIFineTuningHandler,
        )

        handler = OpenAIFineTuningHandler()

        for i in range(3):
            messages = [_make_chat_message("user", f"Question {i}")]
            handler.on_event_start(
                CBEventType.LLM,
                payload={EventPayload.MESSAGES: messages},
                event_id=f"evt_{i}",
            )
            response_obj = _make_llm_response(f"Answer {i}")
            handler.on_event_end(
                CBEventType.LLM,
                payload={EventPayload.RESPONSE: response_obj},
                event_id=f"evt_{i}",
            )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as tmp:
            tmp_path = tmp.name

        try:
            handler.save_finetuning_events(tmp_path)
            with open(tmp_path) as f:
                lines = [l for l in f.readlines() if l.strip()]
            assert len(lines) == 3
        finally:
            os.unlink(tmp_path)

    def test_start_trace_end_trace_are_noops(self) -> None:
        """start_trace and end_trace should not raise."""
        handler = self._get_handler()
        handler.start_trace(trace_id="trace1")
        handler.end_trace(trace_id="trace1", trace_map={})

    def test_function_calls_recorded(self) -> None:
        handler = self._get_handler()
        functions = [{"name": "get_weather", "description": "Get weather data"}]
        messages = [_make_chat_message("user", "What's the weather?")]
        handler.on_event_start(
            CBEventType.LLM,
            payload={
                EventPayload.MESSAGES: messages,
                EventPayload.ADDITIONAL_KWARGS: {"functions": functions},
            },
            event_id="fn_evt",
        )
        assert "fn_evt" in handler._function_calls
        assert handler._function_calls["fn_evt"] == functions
