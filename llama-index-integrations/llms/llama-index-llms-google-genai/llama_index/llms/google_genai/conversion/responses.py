from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import google.genai.types as types
from llama_index.core.base.llms.types import ChatMessage, ChatResponse, MessageRole
from llama_index.core.base.llms.types import (
    ContentBlock,
    ImageBlock,
    TextBlock,
    ThinkingBlock,
    ToolCallBlock,
)


@dataclass
class GeminiResponseParseState:
    """
    Accumulate response blocks and raw Gemini parts across streaming chunks.

    Gemini thinking models can return a ``thought_signature`` on content parts.
    When present, signatures must be preserved and replayed back to Gemini
    exactly as received and in the same content-part position.

    The state maintained here is used to incrementally build up a single logical
    model message while streaming.
    """

    blocks: List[ContentBlock] = field(default_factory=list)
    thought_signatures: List[Optional[str]] = field(default_factory=list)


class GeminiResponseContentBuilder:
    """
    Accumulate LlamaIndex content blocks from Gemini response parts.

    Maintains a strict 1:1 alignment between blocks and thought signatures.
    This is important for Gemini function calling validation.
    """

    def __init__(
        self, blocks: List[ContentBlock], thought_signatures: List[Optional[str]]
    ) -> None:
        self.blocks = blocks
        self.thought_signatures = thought_signatures

    @classmethod
    def from_state(
        cls, state: GeminiResponseParseState
    ) -> "GeminiResponseContentBuilder":
        return cls(
            blocks=state.blocks,
            thought_signatures=state.thought_signatures,
        )

    def apply_part(self, part: types.Part) -> None:
        if part.function_call:
            self._append_tool_call(part)

        if part.inline_data:
            self._append_inline_image(part)

        if part.text is not None:
            if part.thought:
                self._append_thinking(part)
            else:
                self._append_text(part.text, part.thought_signature)

    def _append_block(
        self, block: ContentBlock, thought_signature: Optional[str]
    ) -> None:
        self.blocks.append(block)
        self.thought_signatures.append(thought_signature)

    def _append_text(self, text: str, thought_signature: Optional[str]) -> None:
        # preserve strict part boundaries: 1 Gemini Part -> 1 LlamaIndex block.
        # do not merge text blocks across parts (Gemini 3 validation).
        self._append_block(TextBlock(text=text), thought_signature)

    def _append_tool_call(self, part: types.Part) -> None:
        if not part.function_call:
            return

        # Gemini chat/function calling does not reliably provide a function_call.id
        # (unlike OpenAI's tool_call_id). For downstream compatibility we use the
        # function name as the ToolCallBlock id.
        tool_name = part.function_call.name or ""
        tool_call_id = tool_name

        self._append_block(
            ToolCallBlock(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                tool_kwargs=part.function_call.args or {},
            ),
            part.thought_signature,
        )

    def _append_inline_image(self, part: types.Part) -> None:
        if not part.inline_data:
            return

        self._append_block(
            ImageBlock(
                image=part.inline_data.data,
                image_mimetype=part.inline_data.mime_type,
            ),
            part.thought_signature,
        )

    def _append_thinking(self, part: types.Part) -> None:
        self._append_block(
            ThinkingBlock(
                content=part.text or "",
                additional_information=part.model_dump(exclude={"text"}),
            ),
            part.thought_signature,
        )


class ResponseConverter:
    """
    Convert Gemini responses to LlamaIndex responses.

    Some Gemini models return a ``thought_signature`` on response parts.

    Thought signatures must be preserved and replayed back to Gemini exactly as
    received, in the same content-part position, especially during function
    calling with thinking models.

    When streaming, this converter accumulates blocks and signatures in
    ``GeminiResponseParseState``.
    """

    @staticmethod
    def _extract_raw(response: types.GenerateContentResponse) -> Dict[str, Any]:
        raw: Dict[str, Any] = {}
        if response.candidates:
            top_candidate = response.candidates[0]
            raw.update(top_candidate.model_dump())

        response_feedback = (
            response.prompt_feedback.model_dump() if response.prompt_feedback else {}
        )
        raw.update(response_feedback)

        if response.usage_metadata:
            raw["usage_metadata"] = response.usage_metadata.model_dump()

        if (
            hasattr(response, "usage_metadata")
            and response.usage_metadata
            and response.usage_metadata.prompt_tokens_details
        ):
            raw["usage_metadata"]["prompt_tokens_details"] = [
                detail.model_dump()
                for detail in response.usage_metadata.prompt_tokens_details
            ]

        if hasattr(response, "cached_content") and response.cached_content:
            raw["cached_content"] = response.cached_content
        return raw

    @staticmethod
    def _error_if_finished_early(candidate: types.Candidate) -> None:
        if finish_reason := candidate.finish_reason:
            if finish_reason != types.FinishReason.STOP:
                reason = finish_reason.name
                if (
                    finish_reason == types.FinishReason.SAFETY
                    and candidate.safety_ratings
                ):
                    relevant_safety = list(
                        filter(
                            lambda sr: sr.probability
                            and sr.probability.value
                            > types.HarmProbability.NEGLIGIBLE.value,
                            candidate.safety_ratings,
                        )
                    )
                    reason += f" {relevant_safety}"
                raise RuntimeError(f"Response was terminated early: {reason}")

    @staticmethod
    def _from_gemini_role(role: str) -> MessageRole:
        if role == "model":
            return MessageRole.ASSISTANT
        if role == "function":
            return MessageRole.TOOL
        return MessageRole.USER

    def _role_from_candidate(self, candidate: types.Candidate) -> str:
        role = (
            candidate.content.role
            if candidate.content and candidate.content.role
            else "model"
        )
        mapped = self._from_gemini_role(role)
        return mapped.value

    def to_chat_response(
        self,
        response: types.GenerateContentResponse,
        *,
        state: GeminiResponseParseState,
    ) -> ChatResponse:
        """
        Convert a Gemini response into a LlamaIndex ``ChatResponse``.

        Some Gemini models return a ``thought_signature`` on response parts.
        When present, signatures must be preserved and replayed back to Gemini
        exactly as received, in the same content-part position.
        """
        raw = self._extract_raw(response)
        if response.candidates:
            top_candidate = response.candidates[0]
            self._error_if_finished_early(top_candidate)
            role = self._role_from_candidate(top_candidate)
            parts = (
                top_candidate.content.parts
                if top_candidate.content and top_candidate.content.parts
                else []
            )
        else:
            role = MessageRole.ASSISTANT.value
            parts = []

        builder = GeminiResponseContentBuilder.from_state(state)

        additional_kwargs: Dict[str, Any] = {
            "thought_signatures": state.thought_signatures
        }

        thought_tokens: Optional[int] = None
        if response.usage_metadata:
            additional_kwargs["prompt_tokens"] = (
                response.usage_metadata.prompt_token_count
            )
            additional_kwargs["completion_tokens"] = (
                response.usage_metadata.candidates_token_count
            )
            additional_kwargs["total_tokens"] = (
                response.usage_metadata.total_token_count
            )
            if response.usage_metadata.thoughts_token_count:
                thought_tokens = response.usage_metadata.thoughts_token_count

        for part in parts:
            if part.function_response:
                additional_kwargs["tool_call_id"] = part.function_response.id
                return ChatResponse(
                    message=ChatMessage(
                        role=role,
                        content=json.dumps(part.function_response.response),
                        additional_kwargs=additional_kwargs,
                    ),
                    raw=raw,
                    additional_kwargs=additional_kwargs,
                )

            builder.apply_part(part)

        if thought_tokens:
            thinking_blocks = [
                i
                for i, block in enumerate(state.blocks)
                if isinstance(block, ThinkingBlock)
            ]
            if len(thinking_blocks) == 1:
                state.blocks[thinking_blocks[0]].num_tokens = thought_tokens
            elif len(thinking_blocks) > 1:
                state.blocks[thinking_blocks[-1]].additional_information.update(
                    {"total_thinking_tokens": thought_tokens}
                )

        return ChatResponse(
            message=ChatMessage(
                role=role,
                blocks=state.blocks,
                additional_kwargs=additional_kwargs,
            ),
            raw=raw,
            additional_kwargs=additional_kwargs,
        )
