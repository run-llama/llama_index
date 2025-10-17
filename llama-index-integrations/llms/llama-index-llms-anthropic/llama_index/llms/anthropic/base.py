import json
import logging
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    CompletionResponse,
    LLMMetadata,
    MessageRole,
    ContentBlock,
)
from llama_index.core.base.llms.types import TextBlock as LITextBlock
from llama_index.core.base.llms.types import CitationBlock as LICitationBlock
from llama_index.core.base.llms.types import ThinkingBlock as LIThinkingBlock
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CallbackManager
from llama_index.core.constants import DEFAULT_TEMPERATURE
from llama_index.core.llms.callbacks import (
    llm_chat_callback,
    llm_completion_callback,
)
from llama_index.core.llms.function_calling import FunctionCallingLLM, ToolSelection
from llama_index.core.llms.utils import parse_partial_json
from llama_index.core.types import BaseOutputParser, PydanticProgramMode
from llama_index.core.utils import Tokenizer
from llama_index.llms.anthropic.utils import (
    anthropic_modelname_to_contextsize,
    force_single_tool_call,
    is_anthropic_prompt_caching_supported_model,
    is_function_calling_model,
    messages_to_anthropic_messages,
)

import anthropic
from anthropic.types import (
    CitationsDelta,
    ContentBlockDeltaEvent,
    ContentBlockStartEvent,
    ContentBlockStopEvent,
    CitationsSearchResultLocation,
    InputJSONDelta,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    TextBlock,
    TextDelta,
    ThinkingBlock,
    ThinkingDelta,
    ToolUseBlock,
    TextCitation,
    SignatureDelta,
)

if TYPE_CHECKING:
    from llama_index.core.tools.types import BaseTool


logger = logging.getLogger(__name__)

DEFAULT_ANTHROPIC_MODEL = "claude-2.1"
DEFAULT_ANTHROPIC_MAX_TOKENS = 512


class AnthropicTokenizer:
    def __init__(self, client, model) -> None:
        self._client = client
        self.model = model

    def encode(self, text: str, *args: Any, **kwargs: Any) -> List[int]:
        count = self._client.beta.messages.count_tokens(
            messages=[{"role": "user", "content": text}],
            model=self.model,
        ).input_tokens
        return [1] * count


class AnthropicChatResponse(ChatResponse):
    """Extended ChatResponse for Anthropic with citation support."""

    citations: List[Dict[str, Any]] = Field(default_factory=list)


class AnthropicCompletionResponse(CompletionResponse):
    """Extended CompletionResponse for Anthropic with citation support."""

    citations: List[Dict[str, Any]] = Field(default_factory=list)


class Anthropic(FunctionCallingLLM):
    """
    Anthropic LLM.

    Examples:
        `pip install llama-index-llms-anthropic`

        ```python
        from llama_index.llms.anthropic import Anthropic

        llm = Anthropic(model="claude-instant-1")
        resp = llm.stream_complete("Paul Graham is ")
        for r in resp:
            print(r.delta, end="")
        ```

    """

    model: str = Field(
        default=DEFAULT_ANTHROPIC_MODEL, description="The anthropic model to use."
    )
    temperature: float = Field(
        default=DEFAULT_TEMPERATURE,
        description="The temperature to use for sampling.",
        ge=0.0,
        le=1.0,
    )
    max_tokens: int = Field(
        default=DEFAULT_ANTHROPIC_MAX_TOKENS,
        description="The maximum number of tokens to generate.",
        gt=0,
    )

    base_url: Optional[str] = Field(default=None, description="The base URL to use.")
    timeout: Optional[float] = Field(
        default=None, description="The timeout to use in seconds.", ge=0
    )
    max_retries: int = Field(
        default=10, description="The maximum number of API retries.", ge=0
    )
    additional_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Additional kwargs for the anthropic API."
    )
    cache_idx: Optional[int] = Field(
        default=None,
        description=(
            "Set the cache_control for every message up to and including this index. "
            "Set to -1 to cache all messages. "
            "Set to None to disable caching."
        ),
    )
    thinking_dict: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Configure thinking controls for the LLM. See the Anthropic API docs for more details. "
            "For example: thinking_dict={'type': 'enabled', 'budget_tokens': 16000}"
        ),
    )
    tools: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description=(
            "List of tools to provide to the model. "
            "For example: tools=[{'type': 'web_search_20250305', 'name': 'web_search', 'max_uses': 3}]"
        ),
    )
    mcp_servers: Optional[List[dict]] = Field(
        default=None,
        description=(
            "List of MCP servers to use for the model. "
            "For example: mcp_servers=[{'type': 'url', 'url': 'https://mcp.example.com/sse', 'name': 'example-mcp', 'authorization_token': 'YOUR_TOKEN'}]"
        ),
    )

    _client: Union[
        anthropic.Anthropic, anthropic.AnthropicVertex, anthropic.AnthropicBedrock
    ] = PrivateAttr()
    _aclient: Union[
        anthropic.AsyncAnthropic,
        anthropic.AsyncAnthropicVertex,
        anthropic.AsyncAnthropicBedrock,
    ] = PrivateAttr()

    def __init__(
        self,
        model: str = DEFAULT_ANTHROPIC_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_ANTHROPIC_MAX_TOKENS,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: int = 10,
        api_key: Optional[str] = None,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        callback_manager: Optional[CallbackManager] = None,
        default_headers: Optional[Dict[str, str]] = None,
        system_prompt: Optional[str] = None,
        messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]] = None,
        completion_to_prompt: Optional[Callable[[str], str]] = None,
        pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
        output_parser: Optional[BaseOutputParser] = None,
        region: Optional[str] = None,
        project_id: Optional[str] = None,
        aws_region: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        cache_idx: Optional[int] = None,
        thinking_dict: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        mcp_servers: Optional[List[dict]] = None,
    ) -> None:
        additional_kwargs = additional_kwargs or {}
        callback_manager = callback_manager or CallbackManager([])
        # set the temperature to 1 when thinking is enabled, as per: https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking#important-considerations-when-using-extended-thinking
        if thinking_dict and thinking_dict.get("type") == "enabled":
            temperature = 1

        super().__init__(
            temperature=temperature,
            max_tokens=max_tokens,
            additional_kwargs=additional_kwargs,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            model=model,
            callback_manager=callback_manager,
            system_prompt=system_prompt,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            pydantic_program_mode=pydantic_program_mode,
            output_parser=output_parser,
            cache_idx=cache_idx,
            thinking_dict=thinking_dict,
            tools=tools,
            mcp_servers=mcp_servers,
        )

        if region and project_id and not aws_region:
            self._client = anthropic.AnthropicVertex(
                region=region,
                project_id=project_id,
                timeout=timeout,
                max_retries=max_retries,
                default_headers=default_headers,
            )

            self._aclient = anthropic.AsyncAnthropicVertex(
                region=region,
                project_id=project_id,
                timeout=timeout,
                max_retries=max_retries,
                default_headers=default_headers,
            )
        elif aws_region:
            self._client = anthropic.AnthropicBedrock(
                aws_region=aws_region,
                aws_access_key=aws_access_key_id,
                aws_secret_key=aws_secret_access_key,
                max_retries=max_retries,
                default_headers=default_headers,
                timeout=timeout,
            )
            self._aclient = anthropic.AsyncAnthropicBedrock(
                aws_region=aws_region,
                aws_access_key=aws_access_key_id,
                aws_secret_key=aws_secret_access_key,
                max_retries=max_retries,
                default_headers=default_headers,
                timeout=timeout,
            )
        else:
            self._client = anthropic.Anthropic(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
                max_retries=max_retries,
                default_headers=default_headers,
            )
            self._aclient = anthropic.AsyncAnthropic(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
                max_retries=max_retries,
                default_headers=default_headers,
            )

    @classmethod
    def class_name(cls) -> str:
        return "Anthropic_LLM"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=anthropic_modelname_to_contextsize(self.model),
            num_output=self.max_tokens,
            is_chat_model=True,
            model_name=self.model,
            is_function_calling_model=is_function_calling_model(self.model),
        )

    @property
    def tokenizer(self) -> Tokenizer:
        return AnthropicTokenizer(self._client, self.model)

    @property
    def _model_kwargs(self) -> Dict[str, Any]:
        base_kwargs = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        return {
            **base_kwargs,
            **self.additional_kwargs,
        }

    def _get_all_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
        kwargs = {
            **self._model_kwargs,
            **kwargs,
        }

        if self.thinking_dict and "thinking" not in kwargs:
            kwargs["thinking"] = self.thinking_dict

        if self.tools and "tools" not in kwargs:
            kwargs["tools"] = self.tools
        elif self.tools and "tools" in kwargs:
            kwargs["tools"] = [*self.tools, *kwargs["tools"]]

        if self.mcp_servers and "mcp_servers" not in kwargs:
            kwargs["mcp_servers"] = self.mcp_servers
            kwargs["betas"] = ["mcp-client-2025-04-04"]
        elif self.mcp_servers and "mcp_servers" in kwargs:
            kwargs["mcp_servers"] = [*self.mcp_servers, *kwargs["mcp_servers"]]
            kwargs["betas"] = ["mcp-client-2025-04-04"]

        return kwargs

    def _completion_response_from_chat_response(
        self, chat_response: AnthropicChatResponse
    ) -> AnthropicCompletionResponse:
        return AnthropicCompletionResponse(
            text=chat_response.message.content,
            delta=chat_response.delta,
            additional_kwargs=chat_response.additional_kwargs,
            raw=chat_response.raw,
            citations=chat_response.citations,
        )

    def _get_blocks_and_tool_calls_and_thinking(
        self, response: Any
    ) -> Tuple[List[ContentBlock], List[Dict[str, Any]], List[Dict[str, Any]]]:
        tool_calls = []
        blocks: List[ContentBlock] = []
        citations: List[TextCitation] = []
        tracked_citations: Set[str] = set()

        for content_block in response.content:
            if isinstance(content_block, TextBlock):
                blocks.append(LITextBlock(text=content_block.text))
                # Check for citations in this text block
                if hasattr(content_block, "citations") and content_block.citations:
                    for citation in content_block.citations:
                        if (
                            isinstance(citation, CitationsSearchResultLocation)
                            and str(citation) not in tracked_citations
                        ):
                            tracked_citations.add(str(citation))
                            blocks.append(
                                LICitationBlock(
                                    cited_content=LITextBlock(text=citation.cited_text),
                                    source=citation.source,
                                    title=citation.title,
                                    additional_location_info={
                                        "start_block_index": citation.start_block_index,
                                        "end_block_index": citation.end_block_index,
                                        "search_result_index": citation.search_result_index,
                                    },
                                )
                            )
                    citations.extend(content_block.citations)
            # this assumes a single thinking block, which as of 2025-03-06, is always true
            elif isinstance(content_block, ThinkingBlock):
                blocks.append(
                    LIThinkingBlock(
                        content=content_block.thinking,
                        additional_information=content_block.model_dump(
                            exclude={"thinking"}
                        ),
                    )
                )
            elif isinstance(content_block, ToolUseBlock):
                tool_calls.append(content_block.model_dump())

        return blocks, tool_calls, [x.model_dump() for x in citations]

    @llm_chat_callback()
    def chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> AnthropicChatResponse:
        anthropic_messages, system_prompt = messages_to_anthropic_messages(
            messages, self.cache_idx, self.model
        )
        all_kwargs = self._get_all_kwargs(**kwargs)

        response = self._client.messages.create(
            messages=anthropic_messages,
            stream=False,
            system=system_prompt,
            **all_kwargs,
        )

        blocks, tool_calls, citations = self._get_blocks_and_tool_calls_and_thinking(
            response
        )

        return AnthropicChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT,
                blocks=blocks,
                additional_kwargs={
                    "tool_calls": tool_calls,
                },
            ),
            citations=citations,
            raw=dict(response),
        )

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> AnthropicCompletionResponse:
        chat_message = ChatMessage(role=MessageRole.USER, content=prompt)
        chat_response = self.chat([chat_message], **kwargs)
        return self._completion_response_from_chat_response(chat_response)

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> Generator[AnthropicChatResponse, None, None]:
        anthropic_messages, system_prompt = messages_to_anthropic_messages(
            messages, self.cache_idx, self.model
        )
        all_kwargs = self._get_all_kwargs(**kwargs)

        response = self._client.messages.create(
            messages=anthropic_messages, system=system_prompt, stream=True, **all_kwargs
        )

        def gen() -> Generator[AnthropicChatResponse, None, None]:
            content = []
            cur_block = None
            content_delta = ""
            thinking = None
            cur_tool_calls: List[ToolUseBlock] = []
            cur_tool_call: Optional[ToolUseBlock] = None
            cur_tool_json: str = ""
            cur_citations: List[Dict[str, Any]] = []
            tracked_citations: Set[str] = set()
            role = MessageRole.ASSISTANT
            for r in response:
                if isinstance(r, (ContentBlockDeltaEvent, RawContentBlockDeltaEvent)):
                    if isinstance(r.delta, TextDelta):
                        content_delta = r.delta.text or ""
                        if not isinstance(cur_block, LITextBlock):
                            cur_block = LITextBlock(text=content_delta)
                        else:
                            cur_block.text += content_delta

                    elif isinstance(r.delta, CitationsDelta) and isinstance(
                        r.delta.citation, CitationsSearchResultLocation
                    ):
                        citation = r.delta.citation
                        if str(citation) not in tracked_citations:
                            tracked_citations.add(str(citation))
                            content.append(
                                LICitationBlock(
                                    cited_content=LITextBlock(text=citation.cited_text),
                                    source=citation.source,
                                    title=citation.title,
                                    additional_location_info={
                                        "start_block_index": citation.start_block_index,
                                        "end_block_index": citation.end_block_index,
                                        "search_result_index": citation.search_result_index,
                                    },
                                )
                            )
                    elif isinstance(r.delta, SignatureDelta):
                        if not isinstance(cur_block, LIThinkingBlock):
                            cur_block = LIThinkingBlock(
                                content="",
                                additional_information={"signature": r.delta.signature},
                            )
                        else:
                            cur_block.additional_information["signature"] += (
                                r.delta.signature
                            )
                    elif isinstance(r.delta, ThinkingDelta):
                        if cur_block is None:
                            cur_block = LIThinkingBlock(
                                content=r.delta.thinking or "",
                                additional_information={"signature": ""},
                            )
                        else:
                            cur_block.content += r.delta.thinking
                    elif isinstance(r.delta, CitationsDelta):
                        # TODO: handle citation deltas
                        cur_citations.append(r.delta.citation.model_dump())
                    elif isinstance(r.delta, InputJSONDelta) and not isinstance(
                        cur_tool_call, ToolUseBlock
                    ):
                        # TODO: handle server-side tool calls
                        pass
                    else:
                        if not isinstance(cur_tool_call, ToolUseBlock):
                            raise ValueError(
                                "Tool call not started, but got block type "
                                + str(type(r.delta))
                            )
                        content_delta = r.delta.partial_json
                        cur_tool_json += content_delta
                        try:
                            argument_dict = parse_partial_json(cur_tool_json)
                            cur_tool_call.input = argument_dict
                        except ValueError:
                            pass

                    if cur_tool_call is not None:
                        tool_calls_to_send = [*cur_tool_calls, cur_tool_call]
                    else:
                        tool_calls_to_send = cur_tool_calls

                    yield AnthropicChatResponse(
                        message=ChatMessage(
                            role=role,
                            blocks=content,
                            additional_kwargs={
                                "tool_calls": [t.dict() for t in tool_calls_to_send]
                            },
                        ),
                        citations=cur_citations,
                        delta=content_delta,
                        raw=dict(r),
                    )
                elif isinstance(r, (ContentBlockStartEvent, RawContentBlockStartEvent)):
                    if isinstance(r.content_block, ToolUseBlock):
                        cur_tool_call = r.content_block
                        cur_tool_json = ""
                elif isinstance(r, (ContentBlockStopEvent, RawContentBlockStopEvent)):
                    if isinstance(cur_tool_call, ToolUseBlock):
                        cur_tool_calls.append(cur_tool_call)

                    if cur_block is not None:
                        content.append(cur_block)
                        cur_block = None

                    yield AnthropicChatResponse(
                        message=ChatMessage(
                            role=role,
                            blocks=content,
                            additional_kwargs={
                                "tool_calls": [t.dict() for t in tool_calls_to_send]
                            },
                        ),
                        citations=cur_citations,
                        delta="",
                        raw=dict(r),
                    )

        return gen()

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> Generator[AnthropicCompletionResponse, None, None]:
        chat_message = ChatMessage(role=MessageRole.USER, content=prompt)
        chat_response = self.stream_chat([chat_message], **kwargs)

        def gen() -> Generator[AnthropicCompletionResponse, None, None]:
            for r in chat_response:
                yield self._completion_response_from_chat_response(r)

        return gen()

    @llm_chat_callback()
    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> AnthropicChatResponse:
        anthropic_messages, system_prompt = messages_to_anthropic_messages(
            messages, self.cache_idx, self.model
        )
        all_kwargs = self._get_all_kwargs(**kwargs)

        response = await self._aclient.messages.create(
            messages=anthropic_messages,
            system=system_prompt,
            stream=False,
            **all_kwargs,
        )

        blocks, tool_calls, citations = self._get_blocks_and_tool_calls_and_thinking(
            response
        )

        return AnthropicChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT,
                blocks=blocks,
                additional_kwargs={
                    "tool_calls": tool_calls,
                },
            ),
            citations=citations,
            raw=dict(response),
        )

    @llm_completion_callback()
    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> AnthropicCompletionResponse:
        chat_message = ChatMessage(role=MessageRole.USER, content=prompt)
        chat_response = await self.achat([chat_message], **kwargs)
        return self._completion_response_from_chat_response(chat_response)

    @llm_chat_callback()
    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> AsyncGenerator[AnthropicChatResponse, None]:
        anthropic_messages, system_prompt = messages_to_anthropic_messages(
            messages, self.cache_idx, self.model
        )
        all_kwargs = self._get_all_kwargs(**kwargs)

        response = await self._aclient.messages.create(
            messages=anthropic_messages, system=system_prompt, stream=True, **all_kwargs
        )

        async def gen() -> ChatResponseAsyncGen:
            content = []
            cur_block = None
            content_delta = ""
            thinking = None
            cur_tool_calls: List[ToolUseBlock] = []
            cur_tool_call: Optional[ToolUseBlock] = None
            cur_tool_json: str = ""
            cur_citations: List[Dict[str, Any]] = []
            tracked_citations: Set[str] = set()
            role = MessageRole.ASSISTANT
            async for r in response:
                if isinstance(r, (ContentBlockDeltaEvent, RawContentBlockDeltaEvent)):
                    if isinstance(r.delta, TextDelta):
                        content_delta = r.delta.text or ""
                        if not isinstance(cur_block, LITextBlock):
                            cur_block = LITextBlock(text=content_delta)
                        else:
                            cur_block.text += content_delta

                    elif isinstance(r.delta, CitationsDelta) and isinstance(
                        r.delta.citation, CitationsSearchResultLocation
                    ):
                        citation = r.delta.citation
                        if str(citation) not in tracked_citations:
                            tracked_citations.add(str(citation))
                            content.append(
                                LICitationBlock(
                                    cited_content=LITextBlock(text=citation.cited_text),
                                    source=citation.source,
                                    title=citation.title,
                                    additional_location_info={
                                        "start_block_index": citation.start_block_index,
                                        "end_block_index": citation.end_block_index,
                                        "search_result_index": citation.search_result_index,
                                    },
                                )
                            )
                    elif isinstance(r.delta, SignatureDelta):
                        if not isinstance(cur_block, LIThinkingBlock):
                            cur_block = LIThinkingBlock(
                                content="",
                                additional_information={"signature": r.delta.signature},
                            )
                        else:
                            cur_block.additional_information["signature"] += (
                                r.delta.signature
                            )
                    elif isinstance(r.delta, ThinkingDelta):
                        if cur_block is None:
                            cur_block = LIThinkingBlock(
                                content=r.delta.thinking or "",
                                additional_information={"signature": ""},
                            )
                        else:
                            cur_block.content += r.delta.thinking
                    elif isinstance(r.delta, CitationsDelta):
                        # TODO: handle citation deltas
                        cur_citations.append(r.delta.citation.model_dump())
                    elif isinstance(r.delta, InputJSONDelta) and not isinstance(
                        cur_tool_call, ToolUseBlock
                    ):
                        # TODO: handle server-side tool calls
                        pass
                    else:
                        if not isinstance(cur_tool_call, ToolUseBlock):
                            raise ValueError(
                                "Tool call not started, but got block type "
                                + str(type(r.delta))
                            )
                        content_delta = r.delta.partial_json
                        cur_tool_json += content_delta
                        try:
                            argument_dict = parse_partial_json(cur_tool_json)
                            cur_tool_call.input = argument_dict
                        except ValueError:
                            pass

                    if cur_tool_call is not None:
                        tool_calls_to_send = [*cur_tool_calls, cur_tool_call]
                    else:
                        tool_calls_to_send = cur_tool_calls

                    yield AnthropicChatResponse(
                        message=ChatMessage(
                            role=role,
                            blocks=content,
                            additional_kwargs={
                                "tool_calls": [t.dict() for t in tool_calls_to_send]
                            },
                        ),
                        citations=cur_citations,
                        delta=content_delta,
                        raw=dict(r),
                    )
                elif isinstance(r, (ContentBlockStartEvent, RawContentBlockStartEvent)):
                    if isinstance(r.content_block, ToolUseBlock):
                        cur_tool_call = r.content_block
                        cur_tool_json = ""
                elif isinstance(r, (ContentBlockStopEvent, RawContentBlockStopEvent)):
                    if isinstance(cur_tool_call, ToolUseBlock):
                        cur_tool_calls.append(cur_tool_call)

                    if cur_block is not None:
                        content.append(cur_block)
                        cur_block = None

                    yield AnthropicChatResponse(
                        message=ChatMessage(
                            role=role,
                            blocks=content,
                            additional_kwargs={
                                "tool_calls": [t.dict() for t in tool_calls_to_send]
                            },
                        ),
                        citations=cur_citations,
                        delta="",
                        raw=dict(r),
                    )

        return gen()

    @llm_completion_callback()
    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> AsyncGenerator[AnthropicCompletionResponse, None]:
        chat_message = ChatMessage(role=MessageRole.USER, content=prompt)
        chat_response_gen = await self.astream_chat([chat_message], **kwargs)

        async def gen() -> AsyncGenerator[AnthropicCompletionResponse, None]:
            async for r in chat_response_gen:
                yield self._completion_response_from_chat_response(r)

        return gen()

    def _map_tool_choice_to_anthropic(
        self, tool_required: bool, allow_parallel_tool_calls: bool
    ) -> dict:
        is_thinking_enabled = (
            self.thinking_dict and self.thinking_dict.get("type") == "enabled"
        )
        return {
            "disable_parallel_tool_use": not allow_parallel_tool_calls,
            "type": "any" if tool_required and not is_thinking_enabled else "auto",
        }

    def _prepare_chat_with_tools(
        self,
        tools: List["BaseTool"],
        user_msg: Optional[Union[str, ChatMessage]] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        tool_required: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Prepare the chat with tools."""
        chat_history = chat_history or []

        if isinstance(user_msg, str):
            user_msg = ChatMessage(role=MessageRole.USER, content=user_msg)
            chat_history.append(user_msg)

        tool_dicts = []
        if tools:
            for tool in tools:
                tool_dicts.append(
                    {
                        "name": tool.metadata.name,
                        "description": tool.metadata.description,
                        "input_schema": tool.metadata.get_parameters_dict(),
                    }
                )
            if "prompt-caching" in kwargs.get("extra_headers", {}).get(
                "anthropic-beta", ""
            ):
                if is_anthropic_prompt_caching_supported_model(self.model):
                    tool_dicts[-1]["cache_control"] = {"type": "ephemeral"}
                else:
                    logger.warning(
                        f"Model '{self.model}' does not support prompt caching. "
                        "Cache control will be ignored. "
                        "See: https://docs.claude.com/en/docs/build-with-claude/prompt-caching"
                    )

        # anthropic doesn't like you specifying a tool choice if you don't have any tools
        tool_choice_dict = (
            {}
            if not tools and not tool_required
            else {
                "tool_choice": self._map_tool_choice_to_anthropic(
                    tool_required, allow_parallel_tool_calls
                )
            }
        )

        return {
            "messages": chat_history,
            "tools": tool_dicts,
            **tool_choice_dict,
            **kwargs,
        }

    def _validate_chat_with_tools_response(
        self,
        response: ChatResponse,
        tools: List["BaseTool"],
        allow_parallel_tool_calls: bool = False,
        **kwargs: Any,
    ) -> ChatResponse:
        """Validate the response from chat_with_tools."""
        if not allow_parallel_tool_calls:
            force_single_tool_call(response)
        return response

    def get_tool_calls_from_response(
        self,
        response: "ChatResponse",
        error_on_no_tool_call: bool = True,
        **kwargs: Any,
    ) -> List[ToolSelection]:
        """Predict and call the tool."""
        tool_calls = response.message.additional_kwargs.get("tool_calls", [])

        if len(tool_calls) < 1:
            if error_on_no_tool_call:
                raise ValueError(
                    f"Expected at least one tool call, but got {len(tool_calls)} tool calls."
                )
            else:
                return []

        tool_selections = []
        for tool_call in tool_calls:
            if (
                "input" not in tool_call
                or "id" not in tool_call
                or "name" not in tool_call
            ):
                raise ValueError("Invalid tool call.")
            if tool_call["type"] != "tool_use":
                raise ValueError("Invalid tool type. Unsupported by Anthropic")
            argument_dict = (
                json.loads(tool_call["input"])
                if isinstance(tool_call["input"], str)
                else tool_call["input"]
            )

            tool_selections.append(
                ToolSelection(
                    tool_id=tool_call["id"],
                    tool_name=tool_call["name"],
                    tool_kwargs=argument_dict,
                )
            )

        return tool_selections
