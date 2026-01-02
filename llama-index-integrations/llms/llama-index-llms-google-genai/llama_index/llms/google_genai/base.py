from __future__ import annotations

import asyncio
import os
import typing
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generator,
    List,
    Literal,
    Optional,
    Sequence,
    Type,
    Union,
)

import google.auth
import google.genai
import google.genai.types as types
import llama_index.core.instrumentation as instrument
from llama_index.core.base.llms.generic_utils import (
    achat_to_completion_decorator,
    astream_chat_to_completion_decorator,
    chat_to_completion_decorator,
    stream_chat_to_completion_decorator,
)
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    LLMMetadata,
    MessageRole,
    ToolCallBlock,
)
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CallbackManager
from llama_index.core.constants import DEFAULT_NUM_OUTPUTS, DEFAULT_TEMPERATURE
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.llms.llm import Model, ToolSelection
from llama_index.core.prompts import PromptTemplate
from llama_index.core.types import PydanticProgramMode

from llama_index.llms.google_genai.client import GenAIClientFactory
from llama_index.llms.google_genai.conversion.messages import MessageConverter
from llama_index.llms.google_genai.conversion.responses import ResponseConverter
from llama_index.llms.google_genai.conversion.tools import ToolSchemaConverter
from llama_index.llms.google_genai.files import FileManager
from llama_index.llms.google_genai.orchestration.chat_session import ChatSessionRunner
from llama_index.llms.google_genai.orchestration.structured import StructuredRunner
from llama_index.llms.google_genai.retry import llm_retry_decorator
from llama_index.llms.google_genai.types import VertexAIConfig

dispatcher = instrument.get_dispatcher(__name__)


DEFAULT_MODEL = "gemini-2.5-flash"


if TYPE_CHECKING:
    from llama_index.core.tools.types import BaseTool


class GoogleGenAI(FunctionCallingLLM):
    """
    Google GenAI LLM.

    Examples:
        `pip install llama-index-llms-google-genai`

        ```python
        from llama_index.llms.google_genai import GoogleGenAI

        llm = GoogleGenAI(model="gemini-2.0-flash", api_key="YOUR_API_KEY")
        resp = llm.complete("Write a poem about a magic backpack")
        print(resp)
        ```

    """

    model: str = Field(default=DEFAULT_MODEL, description="The Gemini model to use.")
    temperature: float = Field(
        default=DEFAULT_TEMPERATURE,
        description="The temperature to use during generation.",
        ge=0.0,
        le=2.0,
    )
    context_window: Optional[int] = Field(
        default=None,
        description=(
            "The context window of the model. If not provided, the default context window 200000 will be used."
        ),
    )
    max_retries: int = Field(
        default=3,
        description="The maximum number of API retries.",
        ge=0,
    )
    is_function_calling_model: bool = Field(
        default=True, description="Whether the model is a function calling model."
    )
    cached_content: Optional[str] = Field(
        default=None,
        description="Cached content to use for the model.",
    )
    built_in_tool: Optional[types.Tool] = Field(
        default=None,
        description="Google GenAI tool to use for the model to augment responses.",
    )
    file_mode: Literal["inline", "fileapi", "hybrid"] = Field(
        default="hybrid",
        description="Whether to use inline-only, FileAPI-only or both for handling files.",
    )

    _max_tokens: int = PrivateAttr()
    _client: google.genai.Client = PrivateAttr()
    _generation_config: Dict[str, Any] = PrivateAttr()
    _model_meta: types.Model = PrivateAttr()

    _client_factory: GenAIClientFactory = PrivateAttr()
    _file_manager: FileManager = PrivateAttr()
    _message_converter: MessageConverter = PrivateAttr()
    _response_converter: ResponseConverter = PrivateAttr()
    _tool_schema_converter: ToolSchemaConverter = PrivateAttr()
    _chat_runner: ChatSessionRunner = PrivateAttr()
    _structured_runner: StructuredRunner = PrivateAttr()

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: Optional[int] = None,
        context_window: Optional[int] = None,
        max_retries: int = 3,
        vertexai_config: Optional[VertexAIConfig] = None,
        http_options: Optional[types.HttpOptions] = None,
        debug_config: Optional[google.genai.client.DebugConfig] = None,
        generation_config: Optional[types.GenerateContentConfig] = None,
        callback_manager: Optional[CallbackManager] = None,
        is_function_calling_model: bool = True,
        cached_content: Optional[str] = None,
        built_in_tool: Optional[types.Tool] = None,
        file_mode: Literal["inline", "fileapi", "hybrid"] = "hybrid",
        **kwargs: Any,
    ):
        api_key = api_key or os.getenv("GOOGLE_API_KEY", None)
        self._client_factory = GenAIClientFactory()

        client, model_meta = self._client_factory.create(
            model=model,
            api_key=api_key,
            vertexai_config=vertexai_config,
            http_options=http_options,
            debug_config=debug_config,
        )

        super().__init__(
            model=model,
            temperature=temperature,
            context_window=context_window,
            callback_manager=callback_manager,
            is_function_calling_model=is_function_calling_model,
            max_retries=max_retries,
            cached_content=cached_content,
            built_in_tool=built_in_tool,
            file_mode=file_mode,
            **kwargs,
        )

        self.model = model
        self._client = client
        self._model_meta = model_meta

        # store this as a dict (not a pydantic model) so we can merge it later,
        if generation_config:
            self._generation_config = generation_config.model_dump()
            if cached_content:
                self._generation_config.setdefault("cached_content", cached_content)
            if built_in_tool is not None:
                if self._generation_config.get("tools") is None:
                    self._generation_config["tools"] = []
                if isinstance(self._generation_config["tools"], list):
                    if len(self._generation_config["tools"]) > 0:
                        raise ValueError(
                            "Providing multiple Google GenAI tools or mixing with custom tools is not supported."
                        )
                self._generation_config["tools"].append(built_in_tool)
        else:
            config_kwargs: Dict[str, Any] = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
                "cached_content": cached_content,
            }
            if built_in_tool:
                config_kwargs["tools"] = [built_in_tool]

            self._generation_config = types.GenerateContentConfig(
                **config_kwargs
            ).model_dump()

        self._max_tokens = (
            max_tokens or model_meta.output_token_limit or DEFAULT_NUM_OUTPUTS
        )

        self._file_manager = FileManager(
            file_mode=file_mode,
            client=self._client,
        )
        self._message_converter = MessageConverter(
            file_manager=self._file_manager,
        )
        self._response_converter = ResponseConverter()
        self._tool_schema_converter = ToolSchemaConverter(
            client=self._client,
        )
        self._chat_runner = ChatSessionRunner(
            client=self._client,
            model=self.model,
            file_manager=self._file_manager,
            message_converter=self._message_converter,
            response_converter=self._response_converter,
        )
        self._structured_runner = StructuredRunner(
            client=self._client,
            model=self.model,
            file_manager=self._file_manager,
            message_converter=self._message_converter,
        )

    def _merge_generation_config_kwargs(
        self, kwargs: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Legacy generation_config merge.

        Matches the old implementation exactly:
        - Mutates kwargs by popping `generation_config`.
        - Merges per-call overrides into the instance base dict.
        """
        kwargs = kwargs or {}
        generation_config = {
            **(self._generation_config or {}),
            **kwargs.pop("generation_config", {}),
        }
        return {**kwargs, "generation_config": generation_config}

    @classmethod
    def class_name(cls) -> str:
        return "GenAI"

    @property
    def metadata(self) -> LLMMetadata:
        """LLM metadata."""
        if self.context_window is None:
            base = self._model_meta.input_token_limit or 200000
            total_tokens = base + self._max_tokens
        else:
            total_tokens = self.context_window

        return LLMMetadata(
            context_window=total_tokens,
            num_output=self._max_tokens,
            model_name=self.model,
            is_chat_model=True,
            is_function_calling_model=self.is_function_calling_model,
        )

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        """Synchronous completion."""
        chat_fn = chat_to_completion_decorator(self._chat)
        return chat_fn(prompt, **kwargs)

    @llm_completion_callback()
    async def acomplete(
        self,
        prompt: str,
        formatted: bool = False,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Asynchronous completion."""
        chat_fn = achat_to_completion_decorator(self._achat)
        return await chat_fn(prompt, **kwargs)

    @llm_completion_callback()
    def stream_complete(
        self,
        prompt: str,
        formatted: bool = False,
        **kwargs: Any,
    ) -> CompletionResponseGen:
        """Streaming synchronous completion."""
        chat_fn = stream_chat_to_completion_decorator(self._stream_chat)
        return chat_fn(prompt, **kwargs)

    @llm_completion_callback()
    async def astream_complete(
        self,
        prompt: str,
        formatted: bool = False,
        **kwargs: Any,
    ) -> CompletionResponseAsyncGen:
        """Streaming asynchronous completion."""
        chat_fn = astream_chat_to_completion_decorator(self.astream_chat)
        return await chat_fn(prompt, **kwargs)

    @llm_retry_decorator
    def _chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        """Internal sync chat implementation."""
        params = self._merge_generation_config_kwargs(kwargs)
        prepared = asyncio.run(self._chat_runner.prepare(messages=messages, **params))
        return self._chat_runner.run(prepared)

    @llm_retry_decorator
    async def _achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        """Internal async chat implementation."""
        params = self._merge_generation_config_kwargs(kwargs)
        prepared = await self._chat_runner.prepare(messages=messages, **params)
        return await self._chat_runner.arun(prepared)

    def _stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        """Internal sync streaming chat implementation."""
        params = self._merge_generation_config_kwargs(kwargs)
        prepared = asyncio.run(self._chat_runner.prepare(messages=messages, **params))
        return self._chat_runner.stream(prepared)

    async def _astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        """Internal async streaming chat implementation."""
        params = self._merge_generation_config_kwargs(kwargs)
        prepared = await self._chat_runner.prepare(messages=messages, **params)
        return await self._chat_runner.astream(prepared)

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        """Public synchronous chat."""
        return self._chat(messages, **kwargs)

    @llm_chat_callback()
    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        """Public asynchronous chat."""
        return await self._achat(messages, **kwargs)

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        """Public synchronous streaming chat."""
        return self._stream_chat(messages, **kwargs)

    @llm_chat_callback()
    async def astream_chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponseAsyncGen:
        """Public async streaming chat."""
        return await self._astream_chat(messages, **kwargs)

    def _prepare_chat_with_tools(
        self,
        tools: Sequence["BaseTool"],
        user_msg: Optional[Union[str, ChatMessage]] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        tool_required: bool = False,
        tool_choice: Optional[Union[str, dict]] = None,
        strict: Optional[bool] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Prepare params for tool calling."""
        if tool_choice is None:
            tool_choice = "any" if tool_required else "auto"

        if isinstance(tool_choice, dict):
            raise ValueError("Gemini does not support tool_choice as a dict")

        if tool_choice == "auto":
            tool_mode = types.FunctionCallingConfigMode.AUTO
        elif tool_choice == "none":
            tool_mode = types.FunctionCallingConfigMode.NONE
        else:
            tool_mode = types.FunctionCallingConfigMode.ANY

        function_calling_config = types.FunctionCallingConfig(mode=tool_mode)

        # If tool_choice is a specific tool name, restrict allowed functions.
        if tool_choice not in {"auto", "none", "any"}:
            tool_names = [tool.metadata.name for tool in tools if tool.metadata.name]
            if tool_choice not in tool_names:
                function_calling_config.allowed_function_names = tool_names
            else:
                function_calling_config.allowed_function_names = [tool_choice]

        tool_config = types.ToolConfig(
            function_calling_config=function_calling_config,
        )

        tool_declarations: List[types.FunctionDeclaration] = []
        for tool in tools:
            tool_declarations.append(
                self._tool_schema_converter.to_function_declaration(tool)
            )

        if isinstance(user_msg, str):
            user_msg = ChatMessage(role=MessageRole.USER, content=user_msg)

        messages = list(chat_history or [])
        if user_msg is not None:
            messages.append(user_msg)

        return {
            "messages": messages,
            "tools": (
                [types.Tool(function_declarations=tool_declarations)]
                if tool_declarations
                else None
            ),
            "tool_config": tool_config,
            **kwargs,
        }

    def get_tool_calls_from_response(
        self,
        response: ChatResponse,
        error_on_no_tool_call: bool = True,
        **kwargs: Any,
    ) -> List[ToolSelection]:
        """Extract tool calls from a response."""
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
            return []

        return [
            ToolSelection(
                tool_id=block.tool_name,
                tool_name=block.tool_name,
                tool_kwargs=typing.cast(Dict[str, Any], block.tool_kwargs),
            )
            for block in tool_calls
        ]

    @dispatcher.span
    def structured_predict_without_function_calling(
        self,
        output_cls: Type[Model],
        prompt: PromptTemplate,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        **prompt_args: Any,
    ) -> Model:
        """Structured predict without function calling."""
        params = self._merge_generation_config_kwargs(llm_kwargs)
        messages = prompt.format_messages(**prompt_args)

        prepared = asyncio.run(
            self._structured_runner.prepare(
                messages=messages,
                output_cls=output_cls,
                **params,
            )
        )
        return self._structured_runner.run_parsed(prepared)

    @dispatcher.span
    def structured_predict(
        self,
        output_cls: Type[Model],
        prompt: PromptTemplate,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        **prompt_args: Any,
    ) -> Model:
        """Structured predict."""
        if self.pydantic_program_mode != PydanticProgramMode.DEFAULT:
            return super().structured_predict(
                output_cls, prompt, llm_kwargs=llm_kwargs, **prompt_args
            )

        params = self._merge_generation_config_kwargs(llm_kwargs)
        messages = prompt.format_messages(**prompt_args)

        prepared = asyncio.run(
            self._structured_runner.prepare(
                messages=messages,
                output_cls=output_cls,
                **params,
            )
        )
        return self._structured_runner.run(prepared)

    @dispatcher.span
    async def astructured_predict(
        self,
        output_cls: Type[Model],
        prompt: PromptTemplate,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        **prompt_args: Any,
    ) -> Model:
        """Async structured predict."""
        if self.pydantic_program_mode != PydanticProgramMode.DEFAULT:
            return await super().astructured_predict(
                output_cls, prompt, llm_kwargs=llm_kwargs, **prompt_args
            )

        params = self._merge_generation_config_kwargs(llm_kwargs)
        messages = prompt.format_messages(**prompt_args)

        prepared = await self._structured_runner.prepare(
            messages=messages,
            output_cls=output_cls,
            **params,
        )
        return await self._structured_runner.arun(prepared)

    @dispatcher.span
    def stream_structured_predict(
        self,
        output_cls: Type[Model],
        prompt: PromptTemplate,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        **prompt_args: Any,
    ) -> Generator[Union[Model, Any], None, None]:
        """Streaming structured predict."""
        params = self._merge_generation_config_kwargs(llm_kwargs)
        messages = prompt.format_messages(**prompt_args)

        prepared = asyncio.run(
            self._structured_runner.prepare(
                messages=messages,
                output_cls=output_cls,
                **params,
            )
        )
        return self._structured_runner.stream(prepared)

    @dispatcher.span
    async def astream_structured_predict(
        self,
        output_cls: Type[Model],
        prompt: PromptTemplate,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        **prompt_args: Any,
    ) -> typing.AsyncGenerator[Union[Model, Any], None]:
        """Async streaming structured predict."""
        params = self._merge_generation_config_kwargs(llm_kwargs)
        messages = prompt.format_messages(**prompt_args)

        prepared = await self._structured_runner.prepare(
            messages=messages,
            output_cls=output_cls,
            **params,
        )
        return await self._structured_runner.astream(prepared)
