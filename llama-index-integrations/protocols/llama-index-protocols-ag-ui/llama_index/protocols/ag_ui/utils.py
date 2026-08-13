import base64
import datetime
import logging
import re
import uuid
from ag_ui.core import (
    Message,
    SystemMessage,
    UserMessage,
    AssistantMessage,
    ToolMessage,
    DeveloperMessage,
    ToolCall,
    FunctionCall,
)
from ag_ui.core.types import (
    AudioInputContent,
    BinaryInputContent,
    DocumentInputContent,
    ImageInputContent,
    InputContentDataSource,
    InputContentUrlSource,
    TextInputContent,
    VideoInputContent,
)
from ag_ui.encoder import EventEncoder
from typing import Any, List, Optional, Union, Callable

from llama_index.core.llms import ChatMessage
from llama_index.core.base.llms.types import (
    AudioBlock,
    ContentBlock,
    DocumentBlock,
    ImageBlock,
    TextBlock,
)

try:  # VideoBlock landed after the 0.13 floor of this package's core range
    from llama_index.core.base.llms.types import VideoBlock
except ImportError:  # pragma: no cover
    VideoBlock = None  # type: ignore[assignment,misc]
from llama_index.core.tools import BaseTool, FunctionTool
from llama_index.core.workflow import Event

logger = logging.getLogger(__name__)


def _source_kwargs(source: Union[InputContentDataSource, InputContentUrlSource]) -> dict:
    """Project an AG-UI input-content source onto block constructor kwargs.

    A ``data`` source carries base64 in ``value``; blocks take raw bytes. A
    ``url`` source passes through. Returns ``{}`` for a source this converter
    cannot read, so the caller can skip the part with a warning.
    """
    if isinstance(source, InputContentDataSource):
        return {"data": base64.b64decode(source.value), "mime_type": source.mime_type}
    if isinstance(source, InputContentUrlSource):
        return {"url": source.value, "mime_type": source.mime_type}
    return {}


def _binary_part_to_block(part: BinaryInputContent) -> Optional[ContentBlock]:
    """Convert the deprecated flat ``binary`` part, routed by MIME prefix."""
    mime = part.mime_type or ""
    data = base64.b64decode(part.data) if part.data else None
    if not part.url and data is None:
        return None
    if mime.startswith("image/"):
        return ImageBlock(image=data, url=part.url, image_mimetype=mime or None)
    if mime.startswith("audio/"):
        return AudioBlock(audio=data, url=part.url)
    if mime.startswith("video/") and VideoBlock is not None:
        return VideoBlock(video=data, url=part.url, video_mimetype=mime or None)
    return DocumentBlock(
        data=data, url=part.url, document_mimetype=mime or None, title=part.filename
    )


def agui_content_to_blocks(content: Union[str, List[Any]]) -> List[ContentBlock]:
    """Convert AG-UI user-message content onto llama-index content blocks.

    String content becomes a single :class:`TextBlock`. List content maps each
    typed part (text/image/audio/video/document, plus the deprecated flat
    ``binary`` shape) onto the corresponding block; a part this converter
    cannot represent is skipped with a warning rather than silently stringified.
    """
    if isinstance(content, str):
        return [TextBlock(text=content)]

    blocks: List[ContentBlock] = []
    for part in content:
        if isinstance(part, TextInputContent):
            blocks.append(TextBlock(text=part.text))
        elif isinstance(part, ImageInputContent):
            kwargs = _source_kwargs(part.source)
            if not kwargs:
                logger.warning("Skipping image part with unreadable source")
                continue
            blocks.append(
                ImageBlock(
                    image=kwargs.get("data"),
                    url=kwargs.get("url"),
                    image_mimetype=kwargs.get("mime_type"),
                )
            )
        elif isinstance(part, AudioInputContent):
            kwargs = _source_kwargs(part.source)
            if not kwargs:
                logger.warning("Skipping audio part with unreadable source")
                continue
            blocks.append(AudioBlock(audio=kwargs.get("data"), url=kwargs.get("url")))
        elif isinstance(part, VideoInputContent):
            if VideoBlock is None:
                logger.warning(
                    "Skipping video part: this llama-index-core has no VideoBlock"
                )
                continue
            kwargs = _source_kwargs(part.source)
            if not kwargs:
                logger.warning("Skipping video part with unreadable source")
                continue
            blocks.append(
                VideoBlock(
                    video=kwargs.get("data"),
                    url=kwargs.get("url"),
                    video_mimetype=kwargs.get("mime_type"),
                )
            )
        elif isinstance(part, DocumentInputContent):
            kwargs = _source_kwargs(part.source)
            if not kwargs:
                logger.warning("Skipping document part with unreadable source")
                continue
            blocks.append(
                DocumentBlock(
                    data=kwargs.get("data"),
                    url=kwargs.get("url"),
                    document_mimetype=kwargs.get("mime_type"),
                )
            )
        elif isinstance(part, BinaryInputContent):
            block = _binary_part_to_block(part)
            if block is None:
                logger.warning("Skipping binary part with no url or data")
                continue
            blocks.append(block)
        else:
            logger.warning(
                "Skipping unrecognized content part type: %r", type(part).__name__
            )
    return blocks


def _block_to_agui_part(block: ContentBlock) -> Optional[Any]:
    """Convert one llama-index content block back onto an AG-UI input part.

    The inverse of :func:`agui_content_to_blocks` for the block types it emits.
    A block with neither bytes nor a URL (e.g. path-based) has no AG-UI wire
    shape and returns ``None``.
    """

    def _source(
        data: Optional[bytes], url: Optional[Any], mime: Optional[str]
    ) -> Optional[Union[InputContentDataSource, InputContentUrlSource]]:
        if url:
            return InputContentUrlSource(type="url", value=str(url), mime_type=mime)
        if data is not None:
            return InputContentDataSource(
                type="data",
                value=base64.b64encode(data).decode("ascii"),
                mime_type=mime or "application/octet-stream",
            )
        return None

    def _bytes(block: ContentBlock, field: Optional[bytes], resolve: Callable) -> Optional[bytes]:
        # The block classes normalize stored bytes through a base64 heuristic,
        # so the field value is not reliably raw; resolve_*() is. Only resolved
        # when the block has no URL — resolving a URL-backed block would fetch
        # it over the network, and the URL passes through as-is instead.
        if block.url or field is None:
            return None
        return resolve().read()

    if isinstance(block, TextBlock):
        return TextInputContent(type="text", text=block.text)
    if isinstance(block, ImageBlock):
        data = _bytes(block, block.image, block.resolve_image)
        source = _source(data, block.url, block.image_mimetype)
        return ImageInputContent(type="image", source=source) if source else None
    if isinstance(block, AudioBlock):
        mime = f"audio/{block.format}" if block.format else None
        data = _bytes(block, block.audio, block.resolve_audio)
        source = _source(data, block.url, mime)
        return AudioInputContent(type="audio", source=source) if source else None
    if VideoBlock is not None and isinstance(block, VideoBlock):
        data = _bytes(block, block.video, block.resolve_video)
        source = _source(data, block.url, block.video_mimetype)
        return VideoInputContent(type="video", source=source) if source else None
    if isinstance(block, DocumentBlock):
        data = _bytes(block, block.data, block.resolve_document)
        source = _source(data, block.url, block.document_mimetype)
        return DocumentInputContent(type="document", source=source) if source else None
    return None


_STATE_TAG_RE = re.compile(r"<state>[\s\S]*?</state>")


def llama_index_message_to_ag_ui_message(
    message: ChatMessage,
) -> Message:
    msg_id = message.additional_kwargs.get("id", str(uuid.uuid4()))
    if message.role.value == "system":
        return SystemMessage(id=msg_id, content=message.content, role="system")
    elif (
        message.role.value == "user" and "tool_call_id" not in message.additional_kwargs
    ):
        non_text = [b for b in message.blocks if not isinstance(b, TextBlock)]
        if non_text:
            # Multimodal user message: emit typed AG-UI parts, applying the
            # state-tag cleanup to each text part.
            parts = []
            for block in message.blocks:
                if isinstance(block, TextBlock):
                    text = _STATE_TAG_RE.sub("", block.text).strip()
                    if text:
                        parts.append(TextInputContent(type="text", text=text))
                    continue
                part = _block_to_agui_part(block)
                if part is not None:
                    parts.append(part)
            return UserMessage(id=msg_id, content=parts, role="user")
        content = message.content or ""
        content = _STATE_TAG_RE.sub("", content).strip()
        return UserMessage(id=msg_id, content=content, role="user")
    elif message.role.value == "assistant":
        # Remove tool calls from the message
        content = message.content
        if content:
            content = re.sub(r"<tool_call>[\s\S]*?</tool_call>", "", content).strip()

        # Fetch tool calls from the message
        if message.additional_kwargs.get("ag_ui_tool_calls", None):
            tool_calls = [
                ToolCall(
                    type="function",
                    id=tool_call["id"],
                    function=FunctionCall(
                        name=tool_call["name"],
                        arguments=tool_call["arguments"],
                    ),
                )
                for tool_call in message.additional_kwargs["ag_ui_tool_calls"]
            ]
        else:
            tool_calls = None

        return AssistantMessage(
            id=msg_id,
            content=content or None,
            role="assistant",
            tool_calls=tool_calls,
        )
    elif message.role.value == "tool" or "tool_call_id" in message.additional_kwargs:
        tool_call_id = message.additional_kwargs.get("tool_call_id")
        if tool_call_id is None:
            raise ValueError(
                f"Cannot convert tool message to AG-UI ToolMessage: "
                f"'tool_call_id' is missing from additional_kwargs. "
                f"Message id={msg_id!r}"
            )
        return ToolMessage(
            id=msg_id,
            content=message.content or "",
            role="tool",
            tool_call_id=tool_call_id,
        )
    else:
        raise ValueError(f"Unknown message role: {message.role}")


def ag_ui_message_to_llama_index_message(message: Message) -> ChatMessage:
    if isinstance(message, SystemMessage):
        return ChatMessage(
            role="system", content=message.content, additional_kwargs={"id": message.id}
        )
    elif isinstance(message, UserMessage):
        return ChatMessage(
            role="user",
            blocks=agui_content_to_blocks(message.content or ""),
            additional_kwargs={"id": message.id},
        )
    elif isinstance(message, AssistantMessage):
        # TODO: llama-index-core needs to support tool calls on messages in a more official way
        # For now, we'll just convert the tool call into an assistant message
        # This is a bit of an opinionated hack for now to support the ag-ui tool calls
        tool_calls = message.tool_calls
        content = message.content or ""
        if tool_calls:
            tool_calls_str = "\n".join(
                [
                    f"<tool_call><name>{tool_call.function.name}</name><arguments>{tool_call.function.arguments}</arguments></tool_call>"
                    for tool_call in tool_calls
                ]
            )
            content = f"{content}\n\n{tool_calls_str}".strip()
            ag_ui_tool_calls = [
                {
                    "id": tool_call.id,
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                }
                for tool_call in tool_calls
            ]
        else:
            ag_ui_tool_calls = None

        return ChatMessage(
            role="assistant",
            content=content,
            additional_kwargs={
                "id": message.id,
                "ag_ui_tool_calls": ag_ui_tool_calls,
            },
        )
    elif isinstance(message, ToolMessage):
        # TODO: llama-index-core needs to support tool calls on messages in a more official way
        # tool call results into a user message
        # This is a bit of an opinionated hack for now to support the ag-ui tool calls
        return ChatMessage(
            role="user",
            content=message.content,
            additional_kwargs={"id": message.id, "tool_call_id": message.tool_call_id},
        )
    elif isinstance(message, DeveloperMessage):
        return ChatMessage(
            role="system", content=message.content, additional_kwargs={"id": message.id}
        )
    else:
        raise ValueError(f"Unknown message type: {type(message)}")


def workflow_event_to_sse(event: Event) -> str:
    return EventEncoder().encode(event)


def timestamp() -> int:
    return int(datetime.datetime.now().timestamp())


def validate_tool(tool: Union[BaseTool, Callable]) -> BaseTool:
    if isinstance(tool, BaseTool):
        return tool
    elif callable(tool):
        return FunctionTool.from_defaults(tool)
    else:
        raise ValueError(f"Invalid tool type: {type(tool)}")
