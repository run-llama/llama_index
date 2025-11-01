import json
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    TYPE_CHECKING,
)

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    ContentBlock,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    LLMMetadata,
    MessageRole,
    TextBlock,
    ImageBlock,
    ThinkingBlock,
    ToolCallBlock,
)
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CallbackManager
from llama_index.core.constants import DEFAULT_TEMPERATURE
from llama_index.core.llms.callbacks import (
    llm_chat_callback,
    llm_completion_callback,
)
from llama_index.core.base.llms.generic_utils import (
    achat_to_completion_decorator,
    astream_chat_to_completion_decorator,
    chat_to_completion_decorator,
    get_from_param_or_env,
    stream_chat_to_completion_decorator,
)
from llama_index.core.llms.llm import ToolSelection
from llama_index.core.types import BaseOutputParser, PydanticProgramMode
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.llms.mistralai.utils import (
    is_mistralai_function_calling_model,
    is_mistralai_code_model,
    mistralai_modelname_to_contextsize,
    MISTRAL_AI_REASONING_MODELS,
    THINKING_REGEX,
    THINKING_START_REGEX,
)

from mistralai import Mistral
from mistralai.models import ToolCall, FunctionCall
from mistralai.models import (
    Messages,
    AssistantMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
    TextChunk,
    ImageURLChunk,
    ContentChunk,
    ThinkChunk,
)

if TYPE_CHECKING:
    from llama_index.core.tools.types import BaseTool

DEFAULT_MISTRALAI_MODEL = "mistral-large-latest"
DEFAULT_MISTRALAI_ENDPOINT = "https://api.mistral.ai"
DEFAULT_MISTRALAI_MAX_TOKENS = 512


def to_mistral_chunks(content_blocks: Sequence[ContentBlock]) -> Sequence[ContentChunk]:
    content_chunks = []
    for content_block in content_blocks:
        if isinstance(content_block, TextBlock):
            content_chunks.append(TextChunk(text=content_block.text))
        elif isinstance(content_block, ThinkingBlock):
            if content_block.content:
                content_chunks.append(
                    ThinkChunk(thinking=[TextChunk(text=content_block.content)])
                )
        elif isinstance(content_block, ImageBlock):
            if content_block.url:
                content_chunks.append(ImageURLChunk(image_url=str(content_block.url)))
            else:
                base_64_str = (
                    content_block.resolve_image(as_base64=True).read().decode("utf-8")
                )
                image_mimetype = content_block.image_mimetype
                if not image_mimetype:
                    raise ValueError(
                        "Image mimetype not found in chat message image block"
                    )

                content_chunks.append(
                    ImageURLChunk(
                        image_url=f"data:{image_mimetype};base64,{base_64_str}"
                    )
                )
        elif isinstance(content_block, ToolCallBlock):
            pass
        else:
            raise ValueError(f"Unsupported content block type {type(content_block)}")
    return content_chunks


def to_mistral_chatmessage(
    messages: Sequence[ChatMessage],
) -> List[Messages]:
    new_messages = []
    for m in messages:
        unique_tool_calls = []
        tool_calls_li = [
            block for block in m.blocks if isinstance(block, ToolCallBlock)
        ]
        tool_calls = []
        for tool_call_li in tool_calls_li:
            tool_calls.append(
                ToolCall(
                    id=tool_call_li.tool_call_id,
                    function=FunctionCall(
                        name=tool_call_li.tool_name,
                        arguments=tool_call_li.tool_kwargs,
                    ),
                )
            )
            unique_tool_calls.append(
                (tool_call_li.tool_call_id, tool_call_li.tool_name)
            )
        # try with legacy tool calls for compatibility with older chat histories
        if len(m.additional_kwargs.get("tool_calls", [])) > 0:
            tcs = m.additional_kwargs.get("tool_calls", [])
            for tc in tcs:
                if (
                    isinstance(tc, ToolCall)
                    and (tc.id, tc.function.name) not in unique_tool_calls
                ):
                    tool_calls.append(tc)
        chunks = to_mistral_chunks(m.blocks)
        if m.role == MessageRole.USER:
            new_messages.append(UserMessage(content=chunks))
        elif m.role == MessageRole.ASSISTANT:
            new_messages.append(AssistantMessage(content=chunks, tool_calls=tool_calls))
        elif m.role == MessageRole.SYSTEM:
            new_messages.append(SystemMessage(content=chunks))
        elif m.role == MessageRole.TOOL or m.role == MessageRole.FUNCTION:
            new_messages.append(
                ToolMessage(
                    content=chunks,
                    tool_call_id=m.additional_kwargs.get("tool_call_id"),
                    name=m.additional_kwargs.get("name"),
                )
            )
        else:
            raise ValueError(f"Unsupported message role {m.role}")

    return new_messages


def force_single_tool_call(response: ChatResponse) -> None:
    tool_calls = [
        block for block in response.message.blocks if isinstance(block, ToolCallBlock)
    ]
    if len(tool_calls) > 1:
        response.message.blocks = [
            block
            for block in response.message.blocks
            if not isinstance(block, ToolCallBlock)
        ] + [tool_calls[0]]


class MistralAI(FunctionCallingLLM):
    """
    MistralAI LLM.

    Examples:
        `pip install llama-index-llms-mistralai`

        ```python
        from llama_index.llms.mistralai import MistralAI

        # To customize your API key, do this
        # otherwise it will lookup MISTRAL_API_KEY from your env variable
        # llm = MistralAI(api_key="<api_key>")

        # You can specify a custom endpoint by passing the `endpoint` variable or setting
        # MISTRAL_ENDPOINT in your environment
        # llm = MistralAI(endpoint="<endpoint>")

        llm = MistralAI()

        resp = llm.complete("Paul Graham is ")

        print(resp)
        ```

    """

    model: str = Field(
        default=DEFAULT_MISTRALAI_MODEL, description="The mistralai model to use."
    )
    temperature: float = Field(
        default=DEFAULT_TEMPERATURE,
        description="The temperature to use for sampling.",
        ge=0.0,
        le=1.0,
    )
    max_tokens: int = Field(
        default=DEFAULT_MISTRALAI_MAX_TOKENS,
        description="The maximum number of tokens to generate.",
        gt=0,
    )

    timeout: float = Field(
        default=120, description="The timeout to use in seconds.", ge=0
    )
    max_retries: int = Field(
        default=5, description="The maximum number of API retries.", ge=0
    )
    random_seed: Optional[int] = Field(
        default=None, description="The random seed to use for sampling."
    )
    additional_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Additional kwargs for the MistralAI API."
    )
    show_thinking: bool = Field(
        default=False,
        description="Whether to show thinking in the final response. Only available for reasoning models.",
    )

    _client: Mistral = PrivateAttr()

    def __init__(
        self,
        model: str = DEFAULT_MISTRALAI_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MISTRALAI_MAX_TOKENS,
        timeout: int = 120,
        max_retries: int = 5,
        safe_mode: bool = False,
        random_seed: Optional[int] = None,
        api_key: Optional[str] = None,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        callback_manager: Optional[CallbackManager] = None,
        system_prompt: Optional[str] = None,
        messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]] = None,
        completion_to_prompt: Optional[Callable[[str], str]] = None,
        pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
        output_parser: Optional[BaseOutputParser] = None,
        endpoint: Optional[str] = None,
        show_thinking: bool = False,
    ) -> None:
        additional_kwargs = additional_kwargs or {}
        callback_manager = callback_manager or CallbackManager([])

        api_key = get_from_param_or_env("api_key", api_key, "MISTRAL_API_KEY", "")

        if not api_key:
            raise ValueError(
                "You must provide an API key to use mistralai. "
                "You can either pass it in as an argument or set it `MISTRAL_API_KEY`."
            )

        # Use the custom endpoint if provided, otherwise default to DEFAULT_MISTRALAI_ENDPOINT
        endpoint = get_from_param_or_env(
            "endpoint", endpoint, "MISTRAL_ENDPOINT", DEFAULT_MISTRALAI_ENDPOINT
        )

        super().__init__(
            temperature=temperature,
            max_tokens=max_tokens,
            additional_kwargs=additional_kwargs,
            timeout=timeout,
            max_retries=max_retries,
            safe_mode=safe_mode,
            random_seed=random_seed,
            model=model,
            callback_manager=callback_manager,
            system_prompt=system_prompt,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            pydantic_program_mode=pydantic_program_mode,
            output_parser=output_parser,
            show_thinking=show_thinking,
        )

        self._client = Mistral(
            api_key=api_key,
            server_url=endpoint,
        )

    @classmethod
    def class_name(cls) -> str:
        return "MistralAI_LLM"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=mistralai_modelname_to_contextsize(self.model),
            num_output=self.max_tokens,
            is_chat_model=True,
            model_name=self.model,
            random_seed=self.random_seed,
            is_function_calling_model=is_mistralai_function_calling_model(self.model),
        )

    @property
    def _model_kwargs(self) -> Dict[str, Any]:
        base_kwargs = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "random_seed": self.random_seed,
            "retries": self.max_retries,
            "timeout_ms": self.timeout * 1000,
        }
        return {
            **base_kwargs,
            **self.additional_kwargs,
        }

    def _get_all_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
        return {
            **self._model_kwargs,
            **kwargs,
        }

    def _separate_thinking(
        self, response: Union[str, List[ContentChunk]]
    ) -> Tuple[str, str]:
        """Separate the thinking from the response."""
        content = ""
        if isinstance(response, str):
            content = response
        else:
            for chunk in response:
                if isinstance(chunk, ThinkChunk):
                    for c in chunk.thinking:
                        if isinstance(c, TextChunk):
                            content += c.text + "\n"

        match = THINKING_REGEX.search(content)
        if match:
            return match.group(1), content.replace(match.group(0), "")

        match = THINKING_START_REGEX.search(content)
        if match:
            return match.group(0), ""

        return "", content

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        # convert messages to mistral ChatMessage

        messages = to_mistral_chatmessage(messages)
        all_kwargs = self._get_all_kwargs(**kwargs)
        response = self._client.chat.complete(messages=messages, **all_kwargs)
        blocks: List[TextBlock | ThinkingBlock | ToolCallBlock] = []

        if self.model in MISTRAL_AI_REASONING_MODELS:
            thinking_txt, response_txt = self._separate_thinking(
                response.choices[0].message.content or []
            )
            if thinking_txt:
                blocks.append(ThinkingBlock(content=thinking_txt))

            response_txt_think_show = ""
            if response.choices[0].message.content:
                if isinstance(response.choices[0].message.content, str):
                    response_txt_think_show = response.choices[0].message.content
                else:
                    for chunk in response.choices[0].message.content:
                        if isinstance(chunk, TextBlock):
                            response_txt_think_show += chunk.text + "\n"
                        if isinstance(chunk, ThinkChunk):
                            for c in chunk.thinking:
                                if isinstance(c, TextChunk):
                                    response_txt_think_show += c.text + "\n"

            response_txt = (
                response_txt if not self.show_thinking else response_txt_think_show
            )
        else:
            response_txt = response.choices[0].message.content

        blocks.append(TextBlock(text=response_txt))
        tool_calls = response.choices[0].message.tool_calls
        if tool_calls is not None:
            for tool_call in tool_calls:
                if isinstance(tool_call, ToolCall):
                    blocks.append(
                        ToolCallBlock(
                            tool_call_id=tool_call.id,
                            tool_kwargs=tool_call.function.arguments,
                            tool_name=tool_call.function.name,
                        )
                    )

        return ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT,
                blocks=blocks,
            ),
            raw=dict(response),
        )

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        complete_fn = chat_to_completion_decorator(self.chat)
        return complete_fn(prompt, **kwargs)

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        # convert messages to mistral ChatMessage

        messages = to_mistral_chatmessage(messages)
        all_kwargs = self._get_all_kwargs(**kwargs)

        response = self._client.chat.stream(messages=messages, **all_kwargs)

        def gen() -> ChatResponseGen:
            content = ""
            blocks: List[TextBlock | ThinkingBlock | ToolCallBlock] = []
            for chunk in response:
                delta = chunk.data.choices[0].delta
                role = delta.role or MessageRole.ASSISTANT

                # NOTE: Unlike openAI, we are directly injecting the tool calls
                if delta.tool_calls:
                    for tool_call in delta.tool_calls:
                        if isinstance(tool_call, ToolCall):
                            blocks.append(
                                ToolCallBlock(
                                    tool_call_id=tool_call.id,
                                    tool_name=tool_call.function.name,
                                    tool_kwargs=tool_call.function.arguments,
                                )
                            )

                content_delta = delta.content or ""
                content_delta_str = ""
                if isinstance(content_delta, str):
                    content_delta_str = content_delta
                else:
                    for chunk in content_delta:
                        if isinstance(chunk, TextChunk):
                            content_delta_str += chunk.text + "\n"
                        elif isinstance(chunk, ThinkChunk):
                            for c in chunk.thinking:
                                if isinstance(c, TextChunk):
                                    content_delta_str += c.text + "\n"
                        else:
                            continue

                content += content_delta_str

                # decide whether to include thinking in deltas/responses
                if self.model in MISTRAL_AI_REASONING_MODELS:
                    thinking_txt, response_txt = self._separate_thinking(content)

                    if thinking_txt:
                        blocks.append(ThinkingBlock(content=thinking_txt))

                    content = response_txt if not self.show_thinking else content

                    # If thinking hasn't ended, don't include it in the delta
                    if thinking_txt is None and not self.show_thinking:
                        content_delta = ""
                blocks.append(TextBlock(text=content))

                yield ChatResponse(
                    message=ChatMessage(
                        role=role,
                        blocks=blocks,
                    ),
                    delta=content_delta_str,
                    raw=chunk,
                )

        return gen()

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        stream_complete_fn = stream_chat_to_completion_decorator(self.stream_chat)
        return stream_complete_fn(prompt, **kwargs)

    @llm_chat_callback()
    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        # convert messages to mistral ChatMessage

        messages = to_mistral_chatmessage(messages)
        all_kwargs = self._get_all_kwargs(**kwargs)
        response = await self._client.chat.complete_async(
            messages=messages, **all_kwargs
        )

        blocks: List[TextBlock | ThinkingBlock | ToolCallBlock] = []
        additional_kwargs = {}
        if self.model in MISTRAL_AI_REASONING_MODELS:
            thinking_txt, response_txt = self._separate_thinking(
                response.choices[0].message.content or []
            )
            if thinking_txt:
                blocks.append(ThinkingBlock(content=thinking_txt))

            response_txt_think_show = ""
            if response.choices[0].message.content:
                if isinstance(response.choices[0].message.content, str):
                    response_txt_think_show = response.choices[0].message.content
                else:
                    for chunk in response.choices[0].message.content:
                        if isinstance(chunk, TextBlock):
                            response_txt_think_show += chunk.text + "\n"
                        if isinstance(chunk, ThinkChunk):
                            for c in chunk.thinking:
                                if isinstance(c, TextChunk):
                                    response_txt_think_show += c.text + "\n"

            response_txt = (
                response_txt if not self.show_thinking else response_txt_think_show
            )
        else:
            response_txt = response.choices[0].message.content

        blocks.append(TextBlock(text=response_txt))

        tool_calls = response.choices[0].message.tool_calls
        if tool_calls is not None:
            for tool_call in tool_calls:
                if isinstance(tool_call, ToolCall):
                    blocks.append(
                        ToolCallBlock(
                            tool_call_id=tool_call.id,
                            tool_kwargs=tool_call.function.arguments,
                            tool_name=tool_call.function.name,
                        )
                    )
                else:
                    if isinstance(tool_call[1], (str, dict)):
                        blocks.append(
                            ToolCallBlock(
                                tool_kwargs=tool_call[1], tool_name=tool_call[0]
                            )
                        )
            additional_kwargs["tool_calls"] = (
                tool_calls  # keep this to avoid tool calls loss if tool call does not fall within the validation scenarios above
            )

        return ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT,
                blocks=blocks,
                additional_kwargs=additional_kwargs,
            ),
            raw=dict(response),
        )

    @llm_completion_callback()
    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        acomplete_fn = achat_to_completion_decorator(self.achat)
        return await acomplete_fn(prompt, **kwargs)

    @llm_chat_callback()
    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        # convert messages to mistral ChatMessage

        messages = to_mistral_chatmessage(messages)
        all_kwargs = self._get_all_kwargs(**kwargs)

        response = await self._client.chat.stream_async(messages=messages, **all_kwargs)

        async def gen() -> ChatResponseAsyncGen:
            content = ""
            blocks: List[ThinkingBlock | TextBlock | ToolCallBlock] = []
            async for chunk in response:
                delta = chunk.data.choices[0].delta
                role = delta.role or MessageRole.ASSISTANT
                # NOTE: Unlike openAI, we are directly injecting the tool calls
                if delta.tool_calls:
                    for tool_call in delta.tool_calls:
                        if isinstance(tool_call, ToolCall):
                            blocks.append(
                                ToolCallBlock(
                                    tool_call_id=tool_call.id,
                                    tool_name=tool_call.function.name,
                                    tool_kwargs=tool_call.function.arguments,
                                )
                            )

                content_delta = delta.content or ""
                content_delta_str = ""
                if isinstance(content_delta, str):
                    content_delta_str = content_delta
                else:
                    for chunk in content_delta:
                        if isinstance(chunk, TextChunk):
                            content_delta_str += chunk.text + "\n"
                        elif isinstance(chunk, ThinkChunk):
                            for c in chunk.thinking:
                                if isinstance(c, TextChunk):
                                    content_delta_str += c.text + "\n"
                        else:
                            continue

                content += content_delta_str

                # decide whether to include thinking in deltas/responses
                if self.model in MISTRAL_AI_REASONING_MODELS:
                    thinking_txt, response_txt = self._separate_thinking(content)
                    if thinking_txt:
                        blocks.append(ThinkingBlock(content=thinking_txt))

                    content = response_txt if not self.show_thinking else content

                    # If thinking hasn't ended, don't include it in the delta
                    if thinking_txt is None and not self.show_thinking:
                        content_delta = ""

                blocks.append(TextBlock(text=content))

                yield ChatResponse(
                    message=ChatMessage(
                        role=role,
                        blocks=blocks,
                    ),
                    delta=content_delta_str,
                    raw=chunk,
                )

        return gen()

    @llm_completion_callback()
    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        astream_complete_fn = astream_chat_to_completion_decorator(self.astream_chat)
        return await astream_complete_fn(prompt, **kwargs)

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
        # misralai uses the same openai tool format
        tool_specs = [
            tool.metadata.to_openai_tool(skip_length_check=True) for tool in tools
        ]

        if isinstance(user_msg, str):
            user_msg = ChatMessage(role=MessageRole.USER, content=user_msg)

        messages = chat_history or []
        if user_msg:
            messages.append(user_msg)

        return {
            "messages": messages,
            "tools": tool_specs or None,
            "tool_choice": "required" if tool_required else "auto",
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
    ) -> List[ToolSelection]:
        """Predict and call the tool."""
        tool_calls = [
            block
            for block in response.message.blocks
            if isinstance(block, ToolCallBlock)
        ]

        if len(tool_calls) < 1:
            if error_on_no_tool_call:
                raise ValueError(
                    f"Expected at least one tool call, but got {len(tool_calls)} tool calls."
                )
            else:
                return []

        tool_selections = []
        for tool_call in tool_calls:
            if isinstance(tool_call.tool_kwargs, str):
                argument_dict = json.loads(tool_call.tool_kwargs)
            else:
                argument_dict = tool_call.tool_kwargs

            tool_selections.append(
                ToolSelection(
                    tool_id=tool_call.tool_call_id or "",
                    tool_name=tool_call.tool_name,
                    tool_kwargs=argument_dict,
                )
            )

        return tool_selections

    def fill_in_middle(
        self, prompt: str, suffix: str, stop: Optional[List[str]] = None
    ) -> CompletionResponse:
        if not is_mistralai_code_model(self.model):
            raise ValueError(
                "Please provide code model from MistralAI. Currently supported code model is 'codestral-latest'."
            )

        if stop:
            response = self._client.fim.complete(
                model=self.model, prompt=prompt, suffix=suffix, stop=stop
            )
        else:
            response = self._client.fim.complete(
                model=self.model, prompt=prompt, suffix=suffix
            )

        return CompletionResponse(
            text=response.choices[0].message.content, raw=dict(response)
        )
