"""Google's hosted Gemini API."""

import os
import uuid
import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generator,
    List,
    Optional,
    Sequence,
    Union,
    cast,
)

import google.generativeai as genai
import llama_index.core.instrumentation as instrument
from google.generativeai.types import generation_types
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
)
from llama_index.core.bridge.pydantic import BaseModel, Field, PrivateAttr
from llama_index.core.callbacks import CallbackManager
from llama_index.core.constants import DEFAULT_NUM_OUTPUTS, DEFAULT_TEMPERATURE
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.llms.llm import ToolSelection
from llama_index.core.types import Model
from llama_index.core.utilities.gemini_utils import merge_neighboring_same_role_messages

from .utils import (
    chat_from_gemini_response,
    chat_message_to_gemini,
    completion_from_gemini_response,
    is_function_calling_model,
)

dispatcher = instrument.get_dispatcher(__name__)

GEMINI_MODELS = (
    "models/gemini-2.0-flash",
    "models/gemini-2.0-flash-thinking",
    "models/gemini-2.0-flash-thinking-exp-01-21",
    "models/gemini-2.0-flash-lite",
    "models/gemini-2.0-flash-lite-preview-02-05",
    "models/gemini-2.0-pro-exp-02-05",
    "models/gemini-1.5-flash",
    "models/gemini-1.5-flash-8b",
    "models/gemini-1.0-pro",
)

if TYPE_CHECKING:
    from llama_index.core.tools.types import BaseTool


class Gemini(FunctionCallingLLM):
    """
    Gemini LLM.

    Examples:
        `pip install llama-index-llms-gemini`

        ```python
        from llama_index.llms.gemini import Gemini

        llm = Gemini(model="models/gemini-ultra", api_key="YOUR_API_KEY")
        resp = llm.complete("Write a poem about a magic backpack")
        print(resp)
        ```
    """

    model: str = Field(default=GEMINI_MODELS[0], description="The Gemini model to use.")
    temperature: float = Field(
        default=DEFAULT_TEMPERATURE,
        description="The temperature to use during generation.",
        ge=0.0,
        le=2.0,
    )
    max_tokens: int = Field(
        default=DEFAULT_NUM_OUTPUTS,
        description="The number of tokens to generate.",
        gt=0,
    )
    generate_kwargs: dict = Field(
        default_factory=dict, description="Kwargs for generation."
    )

    _model: genai.GenerativeModel = PrivateAttr()
    _model_meta: genai.types.Model = PrivateAttr()
    _request_options: Optional[genai.types.RequestOptions] = PrivateAttr()

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = GEMINI_MODELS[0],
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: Optional[int] = None,
        generation_config: Optional[genai.types.GenerationConfigDict] = None,
        safety_settings: Optional[genai.types.SafetySettingDict] = None,
        callback_manager: Optional[CallbackManager] = None,
        api_base: Optional[str] = None,
        transport: Optional[str] = None,
        model_name: Optional[str] = None,
        default_headers: Optional[Dict[str, str]] = None,
        request_options: Optional[genai.types.RequestOptions] = None,
        **generate_kwargs: Any,
    ):
        """Creates a new Gemini model interface."""
        if model_name is not None:
            warnings.warn(
                "model_name is deprecated, please use model instead",
                DeprecationWarning,
            )

            model = model_name

        # API keys are optional. The API can be authorised via OAuth (detected
        # environmentally) or by the GOOGLE_API_KEY environment variable.
        config_params: Dict[str, Any] = {
            "api_key": api_key or os.getenv("GOOGLE_API_KEY"),
        }
        if api_base:
            config_params["client_options"] = {"api_endpoint": api_base}
        if transport:
            config_params["transport"] = transport
        if default_headers:
            default_metadata = []
            for key, value in default_headers.items():
                default_metadata.append((key, value))
            # `default_metadata` contains (key, value) pairs that will be sent with every request.
            # When using `transport="rest"`, these will be sent as HTTP headers.
            config_params["default_metadata"] = default_metadata
        # transport: A string, one of: [`rest`, `grpc`, `grpc_asyncio`].
        genai.configure(**config_params)

        base_gen_config = generation_config if generation_config else {}
        # Explicitly passed args take precedence over the generation_config.
        final_gen_config = cast(
            generation_types.GenerationConfigDict,
            {"temperature": temperature, **base_gen_config},
        )

        model_meta = genai.get_model(model)

        genai_model = genai.GenerativeModel(
            model_name=model,
            generation_config=final_gen_config,
            safety_settings=safety_settings,
        )

        supported_methods = model_meta.supported_generation_methods
        if "generateContent" not in supported_methods:
            raise ValueError(
                f"Model {model} does not support content generation, only "
                f"{supported_methods}."
            )

        if not max_tokens:
            max_tokens = model_meta.output_token_limit
        else:
            max_tokens = min(max_tokens, model_meta.output_token_limit)

        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            generate_kwargs=generate_kwargs,
            callback_manager=callback_manager,
        )

        self._model_meta = model_meta
        self._model = genai_model
        self._request_options = request_options
        self._is_function_call_model = is_function_calling_model(model)

    @classmethod
    def class_name(cls) -> str:
        return "Gemini_LLM"

    @property
    def metadata(self) -> LLMMetadata:
        total_tokens = self._model_meta.input_token_limit + self.max_tokens
        return LLMMetadata(
            context_window=total_tokens,
            num_output=self.max_tokens,
            model_name=self.model,
            is_chat_model=True,
            # All gemini models support function calling
            is_function_calling_model=self._is_function_call_model,
        )

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        request_options = self._request_options or kwargs.pop("request_options", None)
        result = self._model.generate_content(
            prompt, request_options=request_options, **kwargs
        )
        return completion_from_gemini_response(result)

    @llm_completion_callback()
    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        request_options = self._request_options or kwargs.pop("request_options", None)
        result = await self._model.generate_content_async(
            prompt, request_options=request_options, **kwargs
        )
        return completion_from_gemini_response(result)

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        request_options = self._request_options or kwargs.pop("request_options", None)

        def gen():
            it = self._model.generate_content(
                prompt, stream=True, request_options=request_options, **kwargs
            )
            for r in it:
                yield completion_from_gemini_response(r)

        return gen()

    @llm_completion_callback()
    def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        request_options = self._request_options or kwargs.pop("request_options", None)

        async def gen():
            it = await self._model.generate_content_async(
                prompt, stream=True, request_options=request_options, **kwargs
            )
            async for r in it:
                yield completion_from_gemini_response(r)

        return gen()

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        request_options = self._request_options or kwargs.pop("request_options", None)
        merged_messages = merge_neighboring_same_role_messages(messages)
        *history, next_msg = map(chat_message_to_gemini, merged_messages)
        chat = self._model.start_chat(history=history)
        response = chat.send_message(
            next_msg,
            request_options=request_options,
            **kwargs,
        )
        return chat_from_gemini_response(response)

    @llm_chat_callback()
    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        request_options = self._request_options or kwargs.pop("request_options", None)
        merged_messages = merge_neighboring_same_role_messages(messages)
        *history, next_msg = map(chat_message_to_gemini, merged_messages)
        chat = self._model.start_chat(history=history)
        response = await chat.send_message_async(
            next_msg, request_options=request_options, **kwargs
        )
        return chat_from_gemini_response(response)

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        request_options = self._request_options or kwargs.pop("request_options", None)
        merged_messages = merge_neighboring_same_role_messages(messages)
        *history, next_msg = map(chat_message_to_gemini, merged_messages)
        chat = self._model.start_chat(history=history)
        response = chat.send_message(
            next_msg, stream=True, request_options=request_options, **kwargs
        )

        def gen() -> ChatResponseGen:
            content = ""
            existing_tool_calls = []
            for r in response:
                top_candidate = r.candidates[0]
                content_delta = top_candidate.content.parts[0].text
                content += content_delta
                llama_resp = chat_from_gemini_response(r)
                existing_tool_calls.extend(
                    llama_resp.message.additional_kwargs.get("tool_calls", [])
                )
                llama_resp.delta = content_delta
                llama_resp.message.content = content
                llama_resp.message.additional_kwargs["tool_calls"] = existing_tool_calls
                yield llama_resp

        return gen()

    @llm_chat_callback()
    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        request_options = self._request_options or kwargs.pop("request_options", None)
        merged_messages = merge_neighboring_same_role_messages(messages)
        *history, next_msg = map(chat_message_to_gemini, merged_messages)
        chat = self._model.start_chat(history=history)
        response = await chat.send_message_async(
            next_msg, stream=True, request_options=request_options, **kwargs
        )

        async def gen() -> ChatResponseAsyncGen:
            content = ""
            existing_tool_calls = []
            async for r in response:
                top_candidate = r.candidates[0]
                content_delta = top_candidate.content.parts[0].text
                content += content_delta
                llama_resp = chat_from_gemini_response(r)
                existing_tool_calls.extend(
                    llama_resp.message.additional_kwargs.get("tool_calls", [])
                )
                llama_resp.delta = content_delta
                llama_resp.message.content = content
                llama_resp.message.additional_kwargs["tool_calls"] = existing_tool_calls
                yield llama_resp

        return gen()

    def _prepare_chat_with_tools(
        self,
        tools: Sequence["BaseTool"],
        user_msg: Optional[Union[str, ChatMessage]] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        tool_choice: Union[str, dict] = "auto",
        strict: Optional[bool] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Predict and call the tool."""
        from google.generativeai.types import FunctionDeclaration, ToolDict
        from google.generativeai.types.content_types import FunctionCallingMode

        if tool_choice == "auto":
            tool_mode = FunctionCallingMode.AUTO
        elif tool_choice == "none":
            tool_mode = FunctionCallingMode.NONE
        else:
            tool_mode = FunctionCallingMode.ANY

        tool_config = {
            "function_calling_config": {
                "mode": tool_mode,
            }
        }

        if tool_choice not in ["auto", "none"]:
            if isinstance(tool_choice, dict):
                raise ValueError("Gemini does not support tool_choice as a dict")

            # assume that the user wants a tool call to be made
            # if the tool choice is not in the list of tools, then we will make a tool call to all tools
            # otherwise, we will make a tool call to the tool choice
            tool_names = [tool.metadata.name for tool in tools]
            if tool_choice not in tool_names:
                tool_config["function_calling_config"][
                    "allowed_function_names"
                ] = tool_names
            else:
                tool_config["function_calling_config"]["allowed_function_names"] = [
                    tool_choice
                ]

        tool_declarations = []
        for tool in tools:
            descriptions = {}
            for param_name, param_schema in tool.metadata.get_parameters_dict()[
                "properties"
            ].items():
                param_description = param_schema.get("description", None)
                if param_description:
                    descriptions[param_name] = param_description

            tool.metadata.fn_schema.__doc__ = tool.metadata.description
            tool_declarations.append(
                FunctionDeclaration.from_function(tool.metadata.fn_schema, descriptions)
            )

        if isinstance(user_msg, str):
            user_msg = ChatMessage(role=MessageRole.USER, content=user_msg)

        messages = chat_history or []
        if user_msg:
            messages.append(user_msg)

        return {
            "messages": messages,
            "tools": (
                ToolDict(function_declarations=tool_declarations)
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
            if not isinstance(tool_call, genai.protos.FunctionCall):
                raise ValueError("Invalid tool_call object")

            tool_selections.append(
                ToolSelection(
                    tool_id=str(uuid.uuid4()),
                    tool_name=tool_call.name,
                    tool_kwargs=dict(tool_call.args),
                )
            )

        return tool_selections

    @dispatcher.span
    def structured_predict(
        self, *args: Any, llm_kwargs: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> BaseModel:
        """Structured predict."""
        llm_kwargs = llm_kwargs or {}
        all_kwargs = {**llm_kwargs, **kwargs}

        if self._is_function_call_model:
            llm_kwargs["tool_choice"] = (
                "required"
                if "tool_choice" not in all_kwargs
                else all_kwargs["tool_choice"]
            )
        # by default structured prediction uses function calling to extract structured outputs
        # here we force tool_choice to be required
        return super().structured_predict(*args, llm_kwargs=llm_kwargs, **kwargs)

    @dispatcher.span
    async def astructured_predict(
        self, *args: Any, llm_kwargs: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> BaseModel:
        """Structured predict."""
        llm_kwargs = llm_kwargs or {}
        all_kwargs = {**llm_kwargs, **kwargs}

        if self._is_function_call_model:
            llm_kwargs["tool_choice"] = (
                "required"
                if "tool_choice" not in all_kwargs
                else all_kwargs["tool_choice"]
            )
        # by default structured prediction uses function calling to extract structured outputs
        # here we force tool_choice to be required
        return await super().astructured_predict(*args, llm_kwargs=llm_kwargs, **kwargs)

    @dispatcher.span
    def stream_structured_predict(
        self, *args: Any, llm_kwargs: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> Generator[Union[Model, List[Model]], None, None]:
        """Stream structured predict."""
        llm_kwargs = llm_kwargs or {}
        all_kwargs = {**llm_kwargs, **kwargs}

        if self._is_function_call_model:
            llm_kwargs["tool_choice"] = (
                "required"
                if "tool_choice" not in all_kwargs
                else all_kwargs["tool_choice"]
            )
        # by default structured prediction uses function calling to extract structured outputs
        # here we force tool_choice to be required
        return super().stream_structured_predict(*args, llm_kwargs=llm_kwargs, **kwargs)

    @dispatcher.span
    async def astream_structured_predict(
        self, *args: Any, llm_kwargs: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> Generator[Union[Model, List[Model]], None, None]:
        """Stream structured predict."""
        llm_kwargs = llm_kwargs or {}
        all_kwargs = {**llm_kwargs, **kwargs}

        if self._is_function_call_model:
            llm_kwargs["tool_choice"] = (
                "required"
                if "tool_choice" not in all_kwargs
                else all_kwargs["tool_choice"]
            )
        # by default structured prediction uses function calling to extract structured outputs
        # here we force tool_choice to be required
        return await super().astream_structured_predict(
            *args, llm_kwargs=llm_kwargs, **kwargs
        )
