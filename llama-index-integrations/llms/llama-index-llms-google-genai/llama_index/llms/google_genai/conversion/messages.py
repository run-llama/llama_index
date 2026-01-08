from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Literal, Optional, Tuple

import google.genai.types as types
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.base.llms.types import (
    ContentBlock,
    DocumentBlock,
    ImageBlock,
    TextBlock,
    ThinkingBlock,
    ToolCallBlock,
    VideoBlock,
)

from llama_index.llms.google_genai.files import FileManager


GeminiRole = Literal["user", "model"]


GEMINI_ROLE_MODEL: GeminiRole = "model"
GEMINI_ROLE_USER: GeminiRole = "user"


class MessageConverter:
    """Convert LlamaIndex chat messages to Gemini content."""

    _ROLES_TO_GEMINI: ClassVar[Dict[MessageRole, GeminiRole]] = {
        MessageRole.USER: GEMINI_ROLE_USER,
        MessageRole.ASSISTANT: GEMINI_ROLE_MODEL,
        # Gemini chat history uses roles "user" and "model"; system messages are
        # handled via system_instruction.
        MessageRole.SYSTEM: GEMINI_ROLE_USER,
        MessageRole.MODEL: GEMINI_ROLE_MODEL,
        # Gemini chat mode accepts tool/function messages as "user" with
        # function_response parts.
        MessageRole.TOOL: GEMINI_ROLE_USER,
        MessageRole.FUNCTION: GEMINI_ROLE_USER,
    }

    def __init__(self, *, file_manager: FileManager) -> None:
        self._file_manager = file_manager

    def _to_gemini_role(self, role: MessageRole) -> GeminiRole:
        return self._ROLES_TO_GEMINI.get(role, GEMINI_ROLE_USER)

    def _gemini_roles_match(self, a: MessageRole, b: MessageRole) -> bool:
        """Return True when two LlamaIndex roles map to the same Gemini chat role."""
        return self._to_gemini_role(a) == self._to_gemini_role(b)

    async def to_gemini_content(
        self,
        message: ChatMessage,
    ) -> Tuple[Optional[types.Content], List[str]]:
        """Convert a LlamaIndex message into a Gemini Content payload."""
        gemini_role = self._to_gemini_role(message.role)

        # Tool responses are encoded as function_response parts.
        if message.additional_kwargs.get("tool_call_id"):
            content, names = self._to_function_response_content(
                message, role=gemini_role
            )
            return content, names

        parts: List[types.Part] = []
        uploaded_file_names: List[str] = []

        normalized_blocks = self._normalize_tool_calls_into_blocks(message)

        for block in normalized_blocks:
            part, file_api_name = await self._block_to_part(block, role=gemini_role)
            if file_api_name is not None:
                uploaded_file_names.append(file_api_name)
            if part is not None:
                parts.append(part)

        # Apply thought signatures to *model* role parts, by position.
        if gemini_role == GEMINI_ROLE_MODEL:
            self._apply_thought_signatures(parts, message)

        parts = self._filter_empty_parts(parts)

        if not parts:
            return None, uploaded_file_names

        return (
            types.Content(role=gemini_role, parts=parts),
            uploaded_file_names,
        )

    @staticmethod
    def _to_function_response_content(
        message: ChatMessage,
        *,
        role: str,
    ) -> Tuple[types.Content, List[str]]:
        tool_call_id = message.additional_kwargs.get("tool_call_id")
        if not isinstance(tool_call_id, str) or not tool_call_id:
            raise ValueError(
                "Tool response message is missing required 'tool_call_id' (Gemini function response id)."
            )

        function_response_part = types.Part.from_function_response(
            name=tool_call_id,
            response={"result": message.content},
        )
        return types.Content(role=role, parts=[function_response_part]), []

    async def _block_to_part(
        self,
        block: ContentBlock,
        role: str,
    ) -> Tuple[Optional[types.Part], Optional[str]]:
        if isinstance(block, TextBlock):
            return self._text_block_to_part(block, role), None
        if isinstance(block, ThinkingBlock):
            return self._thinking_block_to_part(block), None
        if isinstance(block, ToolCallBlock):
            return self._tool_call_block_to_part(block), None
        if isinstance(block, ImageBlock):
            return await self._image_block_to_part(block)
        if isinstance(block, VideoBlock):
            return await self._video_block_to_part(block)
        if isinstance(block, DocumentBlock):
            return await self._document_block_to_part(block)
        raise ValueError(f"Unsupported content block type: {type(block).__name__}")

    @staticmethod
    def _text_block_to_part(block: TextBlock, role: str) -> Optional[types.Part]:
        # MODEL role: preserve empty text (it may carry a thought_signature).
        # USER role: drop empty text (noise that can break FC->FR adjacency).
        if (block.text is None or block.text == "") and role != GEMINI_ROLE_MODEL:
            return None

        if block.text is None:
            return types.Part.from_text(text="")
        return types.Part.from_text(text=block.text)

    @staticmethod
    def _filter_empty_parts(parts: List[types.Part]) -> List[types.Part]:
        """
        Drop empty text parts that don't carry a thought signature.

        Gemini 3 requires strict adjacency between FunctionCall and FunctionResponse.
        Empty text parts (noise) can break this if they appear as separate turns.

        We preserve empty text ONLY if it carries a thought_signature (which is valid
        and sometimes required by Gemini 3).
        """
        filtered: List[types.Part] = []
        for part in parts:
            # Keep if not text (e.g. function_call, inline_data)
            # OR if text is not empty
            # OR if it has a thought_signature
            if (
                part.text is None
                or part.text != ""
                or getattr(part, "thought_signature", None)
            ):
                filtered.append(part)

        return filtered

    @staticmethod
    def _thinking_block_to_part(block: ThinkingBlock) -> Optional[types.Part]:
        part = types.Part.from_text(text=block.content or "")
        part.thought = True

        # Some thinking blocks include a thought_signature captured from Gemini.
        # Replay it exactly as received.
        signature = block.additional_information.get("thought_signature")
        if signature is not None:
            part.thought_signature = signature

        return part

    @staticmethod
    def _tool_call_block_to_part(block: ToolCallBlock) -> types.Part:
        return types.Part.from_function_call(
            name=block.tool_name,
            args=block.tool_kwargs,
        )

        # Gemini does not expose a stable function_call id (tool_call_id).
        # The official SDK examples use function name identity.

    async def _image_block_to_part(
        self, block: ImageBlock
    ) -> Tuple[Optional[types.Part], Optional[str]]:
        file_buffer = block.resolve_image(as_base64=False)
        mime_type = block.image_mimetype or "image/jpeg"
        return await self._file_manager.create_part(file_buffer, mime_type)

    async def _video_block_to_part(
        self, block: VideoBlock
    ) -> Tuple[Optional[types.Part], Optional[str]]:
        file_buffer = block.resolve_video(as_base64=False)
        mime_type = block.video_mimetype or "video/mp4"
        part, name = await self._file_manager.create_part(file_buffer, mime_type)
        if part is not None:
            part.video_metadata = types.VideoMetadata(fps=block.fps)
        return part, name

    async def _document_block_to_part(
        self, block: DocumentBlock
    ) -> Tuple[Optional[types.Part], Optional[str]]:
        file_buffer = block.resolve_document()
        mime_type = block.document_mimetype or "application/pdf"
        return await self._file_manager.create_part(file_buffer, mime_type)

    def _normalize_tool_calls_into_blocks(
        self, message: ChatMessage
    ) -> List[ContentBlock]:
        """
        Ensure tool calls live inside `blocks` so part ordering stays stable.

        We intentionally do not rely on SDK-specific raw parts; we only use the
        normalized LlamaIndex representation.
        """
        blocks: List[ContentBlock] = list(message.blocks)

        tool_calls = list(message.additional_kwargs.get("tool_calls", []) or [])
        if not tool_calls:
            return blocks

        existing: set[str] = set()
        for block in blocks:
            if isinstance(block, ToolCallBlock):
                existing.add(f"{block.tool_name}|{block.tool_kwargs!s}")

        for tool_call in tool_calls:
            key = self._tool_call_dedupe_key(tool_call)
            if key is None or key in existing:
                continue
            tool_id, tool_name, args, _signature = self._parse_tool_call(tool_call)
            blocks.append(
                ToolCallBlock(
                    tool_call_id=tool_id, tool_name=tool_name, tool_kwargs=args
                )
            )
            existing.add(key)

        return blocks

    @staticmethod
    def _tool_call_dedupe_key(tool_call: Any) -> Optional[str]:
        if isinstance(tool_call, dict):
            name = tool_call.get("name")
            args = tool_call.get("args")
        else:
            name = getattr(tool_call, "name", None)
            args = getattr(tool_call, "args", None)

        if not isinstance(name, str) or not name:
            return None
        # Allow None args -> {}
        if args is None:
            args = {}
        if not isinstance(args, dict):
            return None
        return f"{name}|{args!s}"

    @staticmethod
    def _parse_tool_call(
        tool_call: Any,
    ) -> Tuple[str, str, Dict[str, Any], Optional[str]]:
        if isinstance(tool_call, dict):
            tool_id = tool_call.get("id")
            name = tool_call.get("name")
            args = tool_call.get("args")
            signature = tool_call.get("thought_signature")
        else:
            tool_id = getattr(tool_call, "id", None)
            name = getattr(tool_call, "name", None)
            args = getattr(tool_call, "args", None)
            signature = getattr(tool_call, "thought_signature", None)

        # Gemini does not provide a tool-call id; fall back to tool name.
        if tool_id is None:
            tool_id = name
        if not isinstance(tool_id, str) or not tool_id:
            tool_id = name

        if not isinstance(name, str) or not name:
            raise ValueError("tool_calls entry is missing a valid 'name'")

        if args is None:
            args = {}
        if not isinstance(args, dict):
            raise ValueError("tool_calls entry 'args' must be a dict")

        if signature is not None and not isinstance(signature, str):
            raise ValueError("tool_calls entry 'thought_signature' must be a string")

        return tool_id, name, args, signature

    @staticmethod
    def _apply_thought_signatures(
        parts: List[types.Part], message: ChatMessage
    ) -> None:
        thought_signatures = list(
            message.additional_kwargs.get("thought_signatures", [])
        )
        if not thought_signatures:
            return

        for idx, part in enumerate(parts):
            if idx >= len(thought_signatures):
                break
            signature = thought_signatures[idx]
            if signature is not None:
                part.thought_signature = signature
