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
    VideoBlock,
)
from llama_index.core.tools import BaseTool, FunctionTool
from llama_index.core.workflow import Event

logger = logging.getLogger(__name__)

#: MIME type <-> ``AudioBlock.format`` mapping for the cases where the format is
#: not simply the MIME subtype.
_AUDIO_MIME_TO_FORMAT = {"audio/mpeg": "mp3", "audio/x-wav": "wav"}
_AUDIO_FORMAT_TO_MIME = {"mp3": "audio/mpeg", "wav": "audio/x-wav"}


def _audio_format_from_mime(mime: Optional[str]) -> Optional[str]:
    if not mime:
        return None
    # Drop MIME parameters ("audio/wav; codecs=1" -> "audio/wav") and lowercase
    # (MIME types are case-insensitive per RFC 2045): providers take bare
    # lowercase formats like "wav", and llama-index forwards this field to them.
    base = mime.split(";")[0].strip().lower()
    return _AUDIO_MIME_TO_FORMAT.get(base, base.split("/")[-1])


def _audio_mime_from_format(fmt: Optional[str]) -> Optional[str]:
    if not fmt:
        return None
    return _AUDIO_FORMAT_TO_MIME.get(fmt, f"audio/{fmt}")


def _drop_none(**kwargs: Any) -> dict:
    """
    Block constructor kwargs without ``None`` values.

    Field strictness varies across the supported llama-index-core range (0.14.1
    rejects an explicit ``url=None`` on ``AudioBlock``), so absent values are
    omitted rather than passed as ``None``.
    """
    return {key: value for key, value in kwargs.items() if value is not None}


def _source_kwargs(
    source: Union[InputContentDataSource, InputContentUrlSource],
) -> dict:
    """
    Project an AG-UI input-content source onto block constructor kwargs.

    A ``data`` source's base64 payload is passed through as ASCII bytes rather
    than decoded: the llama-index block normalizer keeps bytes that already are
    valid base64 and encodes any other bytes, so handing it raw decoded bytes
    corrupts every payload whose raw form happens to decode as base64 (the
    block's ``resolve_*`` would then decode it a second time). Base64 input is
    always kept verbatim and decoded exactly once. A ``url`` source passes
    through. Returns ``{}`` for a source this converter cannot read, so the
    caller can skip the part with a warning.
    """
    if isinstance(source, InputContentDataSource):
        return {"data": source.value.encode("ascii"), "mime_type": source.mime_type}
    if isinstance(source, InputContentUrlSource):
        return {"url": source.value, "mime_type": source.mime_type}
    return {}


def _binary_part_to_block(part: BinaryInputContent) -> Optional[ContentBlock]:
    """Convert the deprecated flat ``binary`` part, routed by MIME prefix."""
    mime = part.mime_type or ""
    # MIME types are case-insensitive (RFC 2045); route on the lowercased form
    # but keep the original casing on the block's mimetype field.
    mime_lower = mime.lower()
    # Base64 passes through verbatim; see _source_kwargs for why it is not decoded.
    data = part.data.encode("ascii") if part.data else None
    if not part.url and data is None:
        return None
    if mime_lower.startswith("image/"):
        return ImageBlock(
            **_drop_none(image=data, url=part.url, image_mimetype=mime or None)
        )
    if mime_lower.startswith("audio/"):
        return AudioBlock(
            **_drop_none(audio=data, url=part.url, format=_audio_format_from_mime(mime))
        )
    if mime_lower.startswith("video/"):
        return VideoBlock(
            **_drop_none(video=data, url=part.url, video_mimetype=mime or None)
        )
    return DocumentBlock(
        **_drop_none(
            data=data,
            url=part.url,
            document_mimetype=mime or None,
            title=part.filename,
        )
    )


#: additional_kwargs key marking a user message whose AG-UI wire content was a
#: parts LIST (as opposed to the legacy string). The reverse conversion re-emits
#: the same shape instead of inferring it from the block structure, which is
#: lossy for single-part and empty arrays.
AG_UI_PARTS_KEY = "ag_ui_parts"

#: additional_kwargs key marking that the state rewrite prepended a state text
#: block at index 0. The reverse conversion drops exactly that block — never a
#: user-authored block that happens to contain ``<state>`` markup.
AG_UI_STATE_BLOCK_KEY = "ag_ui_injected_state_block"


def agui_content_to_blocks(content: Union[str, List[Any]]) -> List[ContentBlock]:
    """
    Convert AG-UI user-message content onto llama-index content blocks.

    String content becomes a single :class:`TextBlock`. List content maps each
    typed part (text/image/audio/video/document, plus the deprecated flat
    ``binary`` shape) onto the corresponding block; a part this converter
    cannot represent is skipped with a warning rather than silently stringified.
    Part ``metadata`` is deliberately not carried onto blocks: llama-index
    blocks have no field for it, and providers reject unknown content keys.
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
                    **_drop_none(
                        image=kwargs.get("data"),
                        url=kwargs.get("url"),
                        image_mimetype=kwargs.get("mime_type"),
                    )
                )
            )
        elif isinstance(part, AudioInputContent):
            kwargs = _source_kwargs(part.source)
            if not kwargs:
                logger.warning("Skipping audio part with unreadable source")
                continue
            blocks.append(
                AudioBlock(
                    **_drop_none(
                        audio=kwargs.get("data"),
                        url=kwargs.get("url"),
                        format=_audio_format_from_mime(kwargs.get("mime_type")),
                    )
                )
            )
        elif isinstance(part, VideoInputContent):
            kwargs = _source_kwargs(part.source)
            if not kwargs:
                logger.warning("Skipping video part with unreadable source")
                continue
            blocks.append(
                VideoBlock(
                    **_drop_none(
                        video=kwargs.get("data"),
                        url=kwargs.get("url"),
                        video_mimetype=kwargs.get("mime_type"),
                    )
                )
            )
        elif isinstance(part, DocumentInputContent):
            kwargs = _source_kwargs(part.source)
            if not kwargs:
                logger.warning("Skipping document part with unreadable source")
                continue
            blocks.append(
                DocumentBlock(
                    **_drop_none(
                        data=kwargs.get("data"),
                        url=kwargs.get("url"),
                        document_mimetype=kwargs.get("mime_type"),
                    )
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
    """
    Convert one llama-index content block back onto an AG-UI input part.

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

    def _bytes(
        block: ContentBlock, field: Optional[bytes], resolve: Callable
    ) -> Optional[bytes]:
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
        data = _bytes(block, block.audio, block.resolve_audio)
        source = _source(data, block.url, _audio_mime_from_format(block.format))
        return AudioInputContent(type="audio", source=source) if source else None
    if isinstance(block, VideoBlock):
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
        is_parts = bool(message.additional_kwargs.get(AG_UI_PARTS_KEY)) or not (
            len(message.blocks) == 1 and isinstance(message.blocks[0], TextBlock)
        )
        if not is_parts:
            # Legacy string shape: keep emitting a plain string, including the
            # long-standing state-tag cleanup this path always applied.
            content = message.content or ""
            content = _STATE_TAG_RE.sub("", content).strip()
            return UserMessage(id=msg_id, content=content, role="user")
        # Parts shape: emit typed AG-UI parts, preserving the fragment
        # boundaries, ordering, and exact text the client sent (stripping
        # would corrupt significant whitespace). Only the block the state
        # rewrite recorded as injected is dropped — user-authored text that
        # merely contains <state> markup passes through verbatim.
        blocks = list(message.blocks)
        if message.additional_kwargs.get(AG_UI_STATE_BLOCK_KEY) and blocks:
            blocks = blocks[1:]
        parts = []
        for block in blocks:
            if isinstance(block, TextBlock):
                parts.append(TextInputContent(type="text", text=block.text))
                continue
            part = _block_to_agui_part(block)
            if part is not None:
                parts.append(part)
        return UserMessage(id=msg_id, content=parts, role="user")
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
        kwargs = {"id": message.id}
        if isinstance(message.content, list):
            kwargs[AG_UI_PARTS_KEY] = True
        # `or ""` would fold an empty parts array into a phantom empty text
        # block; only None means "no content".
        content = message.content if message.content is not None else ""
        return ChatMessage(
            role="user",
            blocks=agui_content_to_blocks(content),
            additional_kwargs=kwargs,
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
