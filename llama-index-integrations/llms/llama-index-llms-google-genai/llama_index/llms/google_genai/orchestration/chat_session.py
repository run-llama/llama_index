from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

import asyncio
import google.genai
import google.genai.types as types

from llama_index.core.base.llms.types import ChatMessage, ChatResponse
from llama_index.core.base.llms.types import MessageRole
from llama_index.core.base.llms.types import ChatResponseAsyncGen, ChatResponseGen

from llama_index.llms.google_genai.conversion.messages import MessageConverter
from llama_index.llms.google_genai.conversion.responses import (
    GeminiResponseParseState,
    ResponseConverter,
)
from llama_index.llms.google_genai.files import FileManager


class StreamDeltaAccumulator:
    """
    Accumulate response state and compute the text delta per chunk.

    Gemini streams deltas for text parts, so we can extract the text directly
    from the current chunk.
    """

    def __init__(self) -> None:
        self.state = GeminiResponseParseState()

        # Reuse the injected converter; do not recreate per chunk.
        self._converter = ResponseConverter()

    def update(
        self, response: types.GenerateContentResponse
    ) -> tuple[ChatResponse, str]:
        chat_response = self._converter.to_chat_response(response, state=self.state)

        # Google GenAI SDK streaming chunks are deltas for text.
        delta = self._content_text_delta(response) or ""
        return chat_response, delta

    @staticmethod
    def _content_text_delta(response: types.GenerateContentResponse) -> Optional[str]:
        if not response.candidates:
            return None
        top_candidate = response.candidates[0]
        if not top_candidate.content or not top_candidate.content.parts:
            return None

        texts = [
            part.text
            for part in top_candidate.content.parts
            if part.text is not None and not part.thought
        ]

        return "".join(texts) if texts else None


@dataclass(frozen=True)
class PreparedChat:
    """Result of preparing a chat call."""

    next_msg: Union[types.Content, str]
    chat_kwargs: Dict[str, Any]
    uploaded_file_names: List[str]


class ChatSessionRunner:
    """Run chat requests using the Gemini chats API."""

    @staticmethod
    def _is_tool_content(content: types.Content) -> bool:
        if not content.parts:
            return False
        return any(part.function_response is not None for part in content.parts)

    @staticmethod
    def _has_thought_signature(content: types.Content) -> bool:
        if not content.parts:
            return False
        return any(getattr(part, "thought_signature", None) for part in content.parts)

    @staticmethod
    def _has_function_call(content: types.Content) -> bool:
        if not content.parts:
            return False
        return any(part.function_call is not None for part in content.parts)

    @staticmethod
    def _has_function_response(content: types.Content) -> bool:
        if not content.parts:
            return False
        return any(part.function_response is not None for part in content.parts)

    @staticmethod
    def _is_empty_user_text_content(content: types.Content) -> bool:
        if content.role != "user" or not content.parts:
            return False
        if len(content.parts) != 1:
            return False
        part = content.parts[0]
        return part.text is not None and part.text == ""

    def __init__(
        self,
        *,
        client: google.genai.Client,
        model: str,
        file_manager: FileManager,
        message_converter: MessageConverter,
        response_converter: ResponseConverter,
    ) -> None:
        self._client = client
        self._model = model
        self._file_manager = file_manager
        self._message_converter = message_converter
        self._response_converter = response_converter

    async def prepare(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Optional[Union[types.Tool, List[types.Tool]]] = None,
        tool_config: Optional[types.ToolConfig] = None,
        **kwargs: Any,
    ) -> PreparedChat:
        """Prepare parameters for a chat call."""
        msgs = list(messages)
        system_message = self._extract_system_message(msgs)
        history_contents, uploaded_file_names = await self._build_history_contents(msgs)
        history = self._merge_tool_contents(history_contents)
        history = self._merge_adjacent_same_role_if_safe(history)

        # Prevent empty USER turns from breaking FC -> FR adjacency in Gemini 3.
        history = [c for c in history if not self._is_empty_user_text_content(c)]

        self._validate_function_calling_history(history)

        next_msg: Union[types.Content, str] = history.pop() if history else ""

        tools, tool_config = self._extract_tooling_kwargs(tools, tool_config, kwargs)

        generation_config: Union[types.GenerateContentConfig, Dict[str, Any]] = (
            kwargs.pop("generation_config", {})
        )
        if not isinstance(generation_config, dict):
            generation_config = generation_config.model_dump()

        if system_message:
            generation_config["system_instruction"] = system_message

        if tools:
            if not generation_config.get("automatic_function_calling"):
                generation_config["automatic_function_calling"] = (
                    types.AutomaticFunctionCallingConfig(
                        disable=True,
                        maximum_remote_calls=None,
                    )
                )

            if not generation_config.get("tool_config"):
                generation_config["tool_config"] = tool_config

            if not generation_config.get("tools"):
                generation_config["tools"] = tools

        config = types.GenerateContentConfig(**generation_config)

        chat_kwargs: Dict[str, Any] = {
            "model": self._model,
            "history": history,
            "config": config,
            **kwargs,
        }

        return PreparedChat(
            next_msg=next_msg,
            chat_kwargs=chat_kwargs,
            uploaded_file_names=uploaded_file_names,
        )

    @staticmethod
    def _extract_system_message(messages: List[ChatMessage]) -> Optional[str]:
        if messages and messages[0].role == MessageRole.SYSTEM:
            sys_msg = messages.pop(0)
            return sys_msg.content
        return None

    async def _build_history_contents(
        self, messages: Sequence[ChatMessage]
    ) -> tuple[List[types.Content], List[str]]:
        # Convert messages concurrently while preserving input order.
        contents_and_names = await asyncio.gather(
            *(self._message_converter.to_gemini_content(m) for m in list(messages))
        )
        contents = [it[0] for it in contents_and_names if it[0] is not None]
        uploaded_file_names = [name for it in contents_and_names for name in it[1]]
        return contents, uploaded_file_names

    def _merge_tool_contents(
        self, contents: List[types.Content]
    ) -> List[types.Content]:
        """
        Merge adjacent Gemini tool-response contents.

        Gemini tool outputs are represented as ``function_response`` parts.

        LlamaIndex tool execution often produces multiple TOOL-role ChatMessages
        back-to-back (one per tool result). After conversion to Gemini
        ``types.Content``, that becomes consecutive Contents like:

            user: [FR(file1)]
            user: [FR(file2)]

        For Gemini function calling validation, it is safer to send a single
        Content containing all functionResponse parts in order:

            user: [FR(file1), FR(file2)]

        This function performs that merge by concatenating ``parts``.
        """
        history: List[types.Content] = []
        for idx, msg in enumerate(contents):
            if idx < 1:
                history.append(msg)
                continue

            if not self._is_tool_content(msg):
                history.append(msg)
                continue

            last = history[-1]
            if not self._is_tool_content(last):
                history.append(msg)
                continue

            if last.role != msg.role:
                history.append(msg)
                continue

            last.parts = (last.parts or []) + (msg.parts or [])

        return history

    def _merge_adjacent_same_role_if_safe(
        self, contents: List[types.Content]
    ) -> List[types.Content]:
        """
        Gemini chat may reject consecutive contents with same role.

        We merge only when it is safe: no thought signatures and no tool/function parts.

        Example (safe to merge):

            user:  ["Hello"]
            user:  ["Please do X"]
            -> user: ["Hello", "Please do X"]

        Example (NOT safe):

            model: [FunctionCall(...)]
            model: ["..."]

        We avoid merging any content that contains thought signatures, function calls,
        or function responses because Gemini 3 validates part ordering/position.
        """
        merged: List[types.Content] = []
        for msg in contents:
            if not merged:
                merged.append(msg)
                continue

            last = merged[-1]
            if last.role != msg.role:
                merged.append(msg)
                continue

            if (
                self._has_thought_signature(last)
                or self._has_thought_signature(msg)
                or self._has_function_call(last)
                or self._has_function_call(msg)
                or self._has_function_response(last)
                or self._has_function_response(msg)
                or self._is_tool_content(last)
                or self._is_tool_content(msg)
            ):
                merged.append(msg)
                continue

            last.parts = (last.parts or []) + (msg.parts or [])

        return merged

    @staticmethod
    def _validate_function_calling_history(history: List[types.Content]) -> None:
        """
        Validate Gemini 3 function-calling ordering rules (doc-aligned).

        Per docs, sequential multistep tool use is valid across multiple Contents:
        User -> Model FC -> User FR -> Model FC -> User FR ...

        The strict requirement is that within a single step, functionCall parts
        must not be interleaved with functionResponse parts. In practice, the
        safest doc-aligned check we can do is to ensure a single Content does
        not mix FC and FR parts.
        """
        for content in history:
            if not content.parts:
                continue

            has_fc = any(part.function_call is not None for part in content.parts)
            has_fr = any(part.function_response is not None for part in content.parts)

            if has_fc and has_fr:
                raise ValueError(
                    "Invalid tool-calling content: found functionCall and functionResponse "
                    "parts mixed within the same message. Gemini requires all function calls "
                    "to be sent before all function responses (do not interleave within a step)."
                )

    @staticmethod
    def _extract_tooling_kwargs(
        tools: Optional[Union[types.Tool, List[types.Tool]]],
        tool_config: Optional[types.ToolConfig],
        kwargs: Dict[str, Any],
    ) -> tuple[Optional[List[types.Tool]], Optional[types.ToolConfig]]:
        overridden_tools: Optional[Union[types.Tool, List[types.Tool]]] = kwargs.pop(
            "tools", None
        )
        if overridden_tools is not None:
            tools = overridden_tools
        if tools is not None and not isinstance(tools, list):
            tools = [tools]

        overridden_tool_config: Optional[types.ToolConfig] = kwargs.pop(
            "tool_config", None
        )
        if overridden_tool_config is not None:
            tool_config = overridden_tool_config

        return tools, tool_config

    @staticmethod
    def _next_msg_parts(
        next_msg: Union[types.Content, str],
    ) -> Union[List[types.Part], str]:
        return next_msg.parts if isinstance(next_msg, types.Content) else next_msg

    def run(self, prepared: PreparedChat) -> ChatResponse:
        """Execute a prepared chat request (sync)."""
        chat = self._client.chats.create(**prepared.chat_kwargs)
        response = chat.send_message(
            prepared.next_msg.parts
            if isinstance(prepared.next_msg, types.Content)
            else prepared.next_msg
        )

        if self._file_manager.file_mode in ("fileapi", "hybrid"):
            self._file_manager.cleanup(prepared.uploaded_file_names)

        return self._response_converter.to_chat_response(
            response,
            state=GeminiResponseParseState(),
        )

    async def arun(self, prepared: PreparedChat) -> ChatResponse:
        """Execute a prepared chat request (async)."""
        chat = self._client.aio.chats.create(**prepared.chat_kwargs)
        response = await chat.send_message(
            prepared.next_msg.parts
            if isinstance(prepared.next_msg, types.Content)
            else prepared.next_msg
        )

        if self._file_manager.file_mode in ("fileapi", "hybrid"):
            await self._file_manager.acleanup(prepared.uploaded_file_names)

        return self._response_converter.to_chat_response(
            response,
            state=GeminiResponseParseState(),
        )

    def stream(self, prepared: PreparedChat) -> ChatResponseGen:
        """Stream a prepared chat request (sync)."""
        chat = self._client.chats.create(**prepared.chat_kwargs)
        response_stream = chat.send_message_stream(
            self._next_msg_parts(prepared.next_msg)
        )

        def gen() -> ChatResponseGen:
            accumulator = StreamDeltaAccumulator()
            try:
                for r in response_stream:
                    llama_resp, delta = accumulator.update(r)
                    llama_resp.delta = llama_resp.delta or delta
                    yield llama_resp
            finally:
                if self._file_manager.file_mode in ("fileapi", "hybrid"):
                    self._file_manager.cleanup(prepared.uploaded_file_names)

        return gen()

    async def astream(self, prepared: PreparedChat) -> ChatResponseAsyncGen:
        """Stream chat results (async)."""
        chat = self._client.aio.chats.create(**prepared.chat_kwargs)
        response_stream = await chat.send_message_stream(
            self._next_msg_parts(prepared.next_msg)
        )

        async def gen() -> ChatResponseAsyncGen:
            accumulator = StreamDeltaAccumulator()
            try:
                async for r in response_stream:
                    llama_resp, delta = accumulator.update(r)
                    llama_resp.delta = llama_resp.delta or delta
                    yield llama_resp
            finally:
                if self._file_manager.file_mode in ("fileapi", "hybrid"):
                    await self._file_manager.acleanup(prepared.uploaded_file_names)

        return gen()
