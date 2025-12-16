from typing import (
    Any,
    Optional,
    Sequence,
    List,
    Literal,
    Generator,
    AsyncGenerator,
    Callable,
    Optional,
    Union,
    TYPE_CHECKING,
)
from typing_extensions import TypeAlias
from base64 import b64decode
from json import JSONDecodeError
from llama_index.core.base.llms.types import (
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseGen,
    CompletionResponseAsyncGen,
    LLMMetadata,
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    TextBlock,
    DocumentBlock,
    ImageBlock,
    AudioBlock,
    ToolCallBlock,
    ThinkingBlock,
    CitableBlock,
    CitationBlock,
    VideoBlock,
    ContentBlock,
    MessageRole,
)
from llama_index.core.callbacks import CallbackManager
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from llama_index.core.llms.custom import CustomLLM
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.llms.llm import (
    MessagesToPromptType,
    CompletionToPromptType,
    ToolSelection,
)
from llama_index.core.llms.utils import parse_partial_json
from llama_index.core.types import PydanticProgramMode

from pydantic import Field, PrivateAttr
import copy

if TYPE_CHECKING:
    from llama_index.core.tools import BaseTool


class MockLLM(CustomLLM):
    max_tokens: Optional[int]

    def __init__(
        self,
        max_tokens: Optional[int] = None,
        callback_manager: Optional[CallbackManager] = None,
        system_prompt: Optional[str] = None,
        messages_to_prompt: Optional[MessagesToPromptType] = None,
        completion_to_prompt: Optional[CompletionToPromptType] = None,
        pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
    ) -> None:
        super().__init__(
            max_tokens=max_tokens,
            callback_manager=callback_manager or CallbackManager([]),
            system_prompt=system_prompt,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            pydantic_program_mode=pydantic_program_mode,
        )

    @classmethod
    def class_name(cls) -> str:
        return "MockLLM"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(num_output=self.max_tokens or -1)

    def _generate_text(self, length: int) -> str:
        return " ".join(["text" for _ in range(length)])

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        response_text = (
            self._generate_text(self.max_tokens) if self.max_tokens else prompt
        )

        return CompletionResponse(
            text=response_text,
        )

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        def gen_prompt() -> CompletionResponseGen:
            if not prompt:
                yield CompletionResponse(text="", delta="")
                return

            for ch in prompt:
                yield CompletionResponse(
                    text=prompt,
                    delta=ch,
                )

        def gen_response(max_tokens: int) -> CompletionResponseGen:
            for i in range(max_tokens):
                response_text = self._generate_text(i)
                yield CompletionResponse(
                    text=response_text,
                    delta="text ",
                )

        return gen_response(self.max_tokens) if self.max_tokens else gen_prompt()


class MockLLMWithNonyieldingChatStream(MockLLM):
    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        yield from []


class MockLLMWithChatMemoryOfLastCall(MockLLM):
    """
    Mock LLM that keeps track of chat messages of function calls.

    The idea behind this is to be able to easily checks whether the right messages would have been passed to an actual LLM.
    """

    last_chat_messages: Optional[Sequence[ChatMessage]] = Field(
        default=None, exclude=True
    )
    last_called_chat_function: List[str] = Field(default=[], exclude=True)

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        r = super().chat(copy.deepcopy(messages), **kwargs)
        self.last_chat_messages = messages
        self.last_called_chat_function.append("chat")
        return r

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        r = super().stream_chat(copy.deepcopy(messages), **kwargs)
        self.last_chat_messages = messages
        self.last_called_chat_function.append("stream_chat")
        return r

    @llm_chat_callback()
    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        r = await super().achat(copy.deepcopy(messages), **kwargs)
        self.last_chat_messages = messages
        self.last_called_chat_function.append("achat")
        return r

    @llm_chat_callback()
    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        r = await super().astream_chat(copy.deepcopy(messages), **kwargs)
        self.last_chat_messages = messages
        self.last_called_chat_function.append("astream_chat")
        return r

    def reset_memory(self) -> None:
        self.last_chat_messages = None
        self.last_called_chat_function = []

    @classmethod
    def class_name(cls) -> str:
        return "MockLLMWithChatMemoryOfLastCall"


def _data_from_binary(
    data: bytes, block_type: Literal["audio", "document", "image", "video"]
) -> str:
    try:
        b64data = b64decode(data).decode("utf-8")
    except Exception:
        b64data = ""
    return f"<{block_type}>{b64data}</{block_type}>"


def _default_blocks_to_content_callback(
    blocks: list[ContentBlock], tool_calls: Optional[list[ToolCallBlock]] = None
) -> str:
    content_parts: list[str] = []
    for block in blocks:
        if isinstance(block, DocumentBlock):
            content_parts.append(
                _data_from_binary(
                    data=block.data if isinstance(block.data, bytes) else b"",
                    block_type=block.block_type,
                )
            )
        elif isinstance(block, ImageBlock):
            content_parts.append(
                _data_from_binary(
                    data=block.image if isinstance(block.image, bytes) else b"",
                    block_type=block.block_type,
                )
            )
        elif isinstance(block, VideoBlock):
            content_parts.append(
                _data_from_binary(
                    data=block.video if isinstance(block.video, bytes) else b"",
                    block_type=block.block_type,
                )
            )
        elif isinstance(block, AudioBlock):
            content_parts.append(
                _data_from_binary(
                    data=block.audio if isinstance(block.audio, bytes) else b"",
                    block_type=block.block_type,
                )
            )
        elif isinstance(block, TextBlock):
            content_parts.append(block.text)
        elif isinstance(block, ThinkingBlock):
            content_parts.append(block.content or "")
        elif isinstance(block, CitableBlock):
            for c in block.content:
                if isinstance(c, TextBlock):
                    content_parts.append(c.text)
                elif isinstance(c, ImageBlock):
                    content_parts.append(
                        _data_from_binary(
                            c.image if isinstance(c.image, bytes) else b"",
                            block_type=c.block_type,
                        )
                    )
                else:
                    content_parts.append(
                        _data_from_binary(
                            c.data if isinstance(c.data, bytes) else b"",
                            block_type=c.block_type,
                        )
                    )
        elif isinstance(block, CitationBlock):
            if isinstance(block.cited_content, TextBlock):
                content_parts.append(block.cited_content.text)
            else:
                content_parts.append(
                    _data_from_binary(
                        data=block.cited_content.image
                        if isinstance(block.cited_content.image, bytes)
                        else b"",
                        block_type=block.cited_content.block_type,
                    )
                )
        elif isinstance(block, ToolCallBlock):
            if tool_calls is not None:
                tool_calls.append(block)
        else:
            pass
    return "".join(content_parts)


BlockToContentCallback: TypeAlias = Callable[
    [list[ContentBlock], Optional[list[ToolCallBlock]]], str
]

ResponseGenerator: TypeAlias = Callable[[Sequence[ChatMessage]], ChatMessage]


def _default_response_generator(messages: Sequence[ChatMessage]) -> ChatMessage:
    """Default response generator that echoes the last message's content."""
    if not messages:
        return ChatMessage(role="assistant", content="<empty>")

    tool_calls: List[ToolCallBlock] = []
    content = _default_blocks_to_content_callback(messages[-1].blocks, tool_calls)
    if not content:
        content = "<empty>"
    return ChatMessage(role="assistant", content=content)


class MockFunctionCallingLLM(FunctionCallingLLM):
    tool_calls: List[ToolCallBlock] = Field(default_factory=list)
    blocks_to_content_callback: BlockToContentCallback = Field(
        default=_default_blocks_to_content_callback
    )

    _response_generator: Optional[ResponseGenerator] = PrivateAttr(default=None)

    def __init__(
        self,
        callback_manager: Optional[CallbackManager] = None,
        system_prompt: Optional[str] = None,
        messages_to_prompt: Optional[MessagesToPromptType] = None,
        completion_to_prompt: Optional[CompletionToPromptType] = None,
        pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
        blocks_to_content_callback: Optional[BlockToContentCallback] = None,
        response_generator: Optional[ResponseGenerator] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            callback_manager=callback_manager or CallbackManager([]),
            system_prompt=system_prompt,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            pydantic_program_mode=pydantic_program_mode,
            **kwargs,
        )

        if blocks_to_content_callback is not None:
            self.blocks_to_content_callback = blocks_to_content_callback

        if response_generator is not None:
            self._response_generator = response_generator
        # else: leave as None, will use _get_response_generator() method

    @classmethod
    def class_name(cls) -> str:
        return "MockFunctionCallingLLM"

    def _get_response_generator(self) -> ResponseGenerator:
        """Get the active response generator, using default if none set."""
        if self._response_generator is not None:
            return self._response_generator

        # Return a default generator that uses instance's blocks_to_content_callback
        def default_generator(messages: Sequence[ChatMessage]) -> ChatMessage:
            if not messages:
                return ChatMessage(role="assistant", content="<empty>")

            # Pass self.tool_calls to accumulate tool call blocks
            content = self.blocks_to_content_callback(
                messages[-1].blocks, self.tool_calls
            )
            if not content:
                content = "<empty>"
            return ChatMessage(role="assistant", content=content)

        return default_generator

    @property
    def response_generator(self) -> ResponseGenerator:
        """Get the response generator function."""
        return self._get_response_generator()

    @response_generator.setter
    def response_generator(self, value: ResponseGenerator) -> None:
        """Set the response generator function."""
        self._response_generator = value

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(is_function_calling_model=True)

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        return CompletionResponse(text=prompt)

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        def gen() -> CompletionResponseGen:
            yield CompletionResponse(text=prompt, delta=prompt)

        return gen()

    @llm_completion_callback()
    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        return CompletionResponse(text=prompt)

    @llm_completion_callback()
    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        async def gen() -> AsyncGenerator[CompletionResponse, None]:
            yield CompletionResponse(text=prompt, delta=prompt)

        return gen()

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        response_msg = self._get_response_generator()(messages)
        content = response_msg.content or ""

        def _gen() -> Generator[ChatResponse, None, None]:
            yield ChatResponse(
                message=response_msg,
                delta=content,
                raw={"content": content},
            )

        return _gen()

    @llm_chat_callback()
    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        response_msg = self._get_response_generator()(messages)
        content = response_msg.content or ""

        async def _gen() -> AsyncGenerator[ChatResponse, None]:
            yield ChatResponse(
                message=response_msg,
                delta=content,
                raw={"content": content},
            )

        return _gen()

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        response_msg = self._get_response_generator()(messages)
        content = response_msg.content or ""
        return ChatResponse(
            message=response_msg,
            delta=content,
            raw={"content": content},
        )

    @llm_chat_callback()
    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        return self.chat(messages=messages)

    def _prepare_chat_with_tools(
        self,
        tools: Sequence["BaseTool"],
        user_msg: Optional[Union[str, ChatMessage]] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        tool_required: bool = False,
        **kwargs: Any,
    ) -> dict:
        """Prepare the arguments needed to let the LLM chat with tools."""
        # Build the complete message list
        messages = list(chat_history or [])

        # Add user message if provided
        if user_msg:
            if isinstance(user_msg, str):
                messages.append(ChatMessage(role=MessageRole.USER, content=user_msg))
            else:
                messages.append(user_msg)

        # For the mock implementation, we return the messages and tools
        # The actual tools are not used since this is a mock
        return {
            "messages": messages,
            "tools": tools,
        }

    def get_tool_calls_from_response(
        self, response: ChatResponse, error_on_no_tool_call: bool = False, **kwargs: Any
    ) -> List[ToolSelection]:
        # First, check if tool calls are in additional_kwargs (for compatibility with test mocks)
        if "tool_calls" in response.message.additional_kwargs:
            return response.message.additional_kwargs.get("tool_calls", [])

        # Otherwise, extract from blocks
        tool_calls = [
            block
            for block in response.message.blocks
            if isinstance(block, ToolCallBlock)
        ]
        tool_selections = []
        for tool_call in tool_calls:
            # this should handle both complete and partial jsons
            try:
                if isinstance(tool_call.tool_kwargs, str):
                    argument_dict = parse_partial_json(tool_call.tool_kwargs)
                else:
                    argument_dict = tool_call.tool_kwargs
            except (ValueError, TypeError, JSONDecodeError):
                argument_dict = {}
            tool_selections.append(
                ToolSelection(
                    tool_id=tool_call.tool_call_id or "",
                    tool_name=tool_call.tool_name,
                    tool_kwargs=argument_dict,
                )
            )
        return tool_selections
