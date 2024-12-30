from typing import Any, Dict, Optional, Sequence, Union, Tuple, List

from ibm_watsonx_ai import Credentials, APIClient
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames, GenChatParamsMetaNames

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
from llama_index.core.constants import DEFAULT_CONTEXT_WINDOW
from llama_index.core.bridge.pydantic import (
    ConfigDict,
    Field,
    PrivateAttr,
)

from llama_index.core.llms.llm import ToolSelection

# Import SecretStr directly from pydantic
# since there is not one in llama_index.core.bridge.pydantic
from pydantic import SecretStr

from llama_index.core.llms.function_calling import FunctionCallingLLM

from llama_index.core.callbacks import CallbackManager
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from llama_index.core.base.llms.generic_utils import (
    completion_to_chat_decorator,
    stream_completion_to_chat_decorator,
    acompletion_to_chat_decorator,
)

from llama_index.core.llms.utils import parse_partial_json

from llama_index.llms.ibm.utils import (
    resolve_watsonx_credentials,
    to_watsonx_message_dict,
    from_watsonx_message,
    update_tool_calls,
)

# default max tokens determined by service
DEFAULT_MAX_TOKENS = 20


class WatsonxLLM(FunctionCallingLLM):
    """
    IBM watsonx.ai large language models.

    Example:
        `pip install llama-index-llms-ibm`

        ```python

        from llama_index.llms.ibm import WatsonxLLM
        watsonx_llm = WatsonxLLM(
            model_id="google/flan-ul2",
            url="https://us-south.ml.cloud.ibm.com",
            apikey="*****",
            project_id="*****",
        )
        ```
    """

    model_id: Optional[str] = Field(
        default=None, description="Type of model to use.", frozen=True
    )
    deployment_id: Optional[str] = Field(
        default=None, description="Id of deployed model to use.", frozen=True
    )

    temperature: Optional[float] = Field(
        default=None,
        description="The temperature to use for sampling.",
    )
    max_new_tokens: Optional[int] = Field(
        default=None,
        description="The maximum number of tokens to generate.",
    )
    additional_params: Optional[Dict[str, Any]] = Field(
        default_factory=None,
        description="Additional generation params for the watsonx.ai models.",
    )

    project_id: Optional[str] = Field(
        default=None,
        description="ID of the Watson Studio project.",
        frozen=True,
    )

    space_id: Optional[str] = Field(
        default=None, description="ID of the Watson Studio space.", frozen=True
    )

    url: Optional[SecretStr] = Field(
        default=None,
        description="Url to Watson Machine Learning or CPD instance",
        frozen=True,
    )

    apikey: Optional[SecretStr] = Field(
        default=None,
        description="Apikey to Watson Machine Learning or CPD instance",
        frozen=True,
    )

    token: Optional[SecretStr] = Field(
        default=None, description="Token to CPD instance", frozen=True
    )

    password: Optional[SecretStr] = Field(
        default=None, description="Password to CPD instance", frozen=True
    )

    username: Optional[SecretStr] = Field(
        default=None, description="Username to CPD instance", frozen=True
    )

    instance_id: Optional[SecretStr] = Field(
        default=None, description="Instance_id of CPD instance", frozen=True
    )

    version: Optional[SecretStr] = Field(
        default=None, description="Version of CPD instance", frozen=True
    )

    verify: Union[str, bool, None] = Field(
        default=None,
        description="""
        User can pass as verify one of following:
        the path to a CA_BUNDLE file
        the path of directory with certificates of trusted CAs
        True - default path to truststore will be taken
        False - no verification will be made
        """,
        frozen=True,
    )

    validate_model: bool = Field(
        default=True, description="Model id validation", frozen=True
    )

    _model: ModelInference = PrivateAttr()
    _client: Optional[APIClient] = PrivateAttr()
    _model_info: Optional[Dict[str, Any]] = PrivateAttr()
    _deployment_info: Optional[Dict[str, Any]] = PrivateAttr()
    _context_window: Optional[int] = PrivateAttr()
    _text_generation_params: Dict[str, Any] | None = PrivateAttr()

    def __init__(
        self,
        model_id: Optional[str] = None,
        deployment_id: Optional[str] = None,
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        additional_params: Optional[Dict[str, Any]] = None,
        project_id: Optional[str] = None,
        space_id: Optional[str] = None,
        url: Optional[str] = None,
        apikey: Optional[str] = None,
        token: Optional[str] = None,
        password: Optional[str] = None,
        username: Optional[str] = None,
        instance_id: Optional[str] = None,
        version: Optional[str] = None,
        verify: Union[str, bool, None] = None,
        api_client: Optional[APIClient] = None,
        validate_model: bool = True,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize LLM and watsonx.ai ModelInference.
        """
        callback_manager = callback_manager or CallbackManager([])
        additional_params = additional_params or {}

        creds = (
            resolve_watsonx_credentials(
                url=url,
                apikey=apikey,
                token=token,
                username=username,
                password=password,
                instance_id=instance_id,
            )
            if not isinstance(api_client, APIClient)
            else {}
        )

        super().__init__(
            model_id=model_id,
            deployment_id=deployment_id,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            additional_params=additional_params,
            project_id=project_id,
            space_id=space_id,
            url=creds.get("url"),
            apikey=creds.get("apikey"),
            token=creds.get("token"),
            password=creds.get("password"),
            username=creds.get("username"),
            instance_id=creds.get("instance_id"),
            version=version,
            verify=verify,
            _client=api_client,
            validate_model=validate_model,
            callback_manager=callback_manager,
            **kwargs,
        )
        self._context_window = kwargs.get("context_window")

        generation_params = {}
        if self.temperature is not None:
            generation_params["temperature"] = self.temperature
        if self.max_new_tokens is not None:
            generation_params["max_new_tokens"] = self.max_new_tokens

        generation_params = {**generation_params, **additional_params}

        if generation_params:
            self._text_generation_params, _ = self._split_generation_params(
                generation_params
            )
        else:
            self._text_generation_params = None

        self._client = api_client
        self._model = ModelInference(
            model_id=model_id,
            deployment_id=deployment_id,
            credentials=(
                Credentials.from_dict(
                    {
                        key: value.get_secret_value() if value else None
                        for key, value in self._get_credential_kwargs().items()
                    },
                    _verify=self.verify,
                )
                if creds
                else None
            ),
            params=self._text_generation_params,
            project_id=self.project_id,
            space_id=self.space_id,
            api_client=api_client,
            validate=validate_model,
        )
        self._model_info = None
        self._deployment_info = None

    model_config = ConfigDict(protected_namespaces=(), validate_assignment=True)

    @property
    def model_info(self):
        if self._model.model_id and self._model_info is None:
            self._model_info = self._model.get_details()
        return self._model_info

    @property
    def deployment_info(self):
        if self._model.deployment_id and self._deployment_info is None:
            self._deployment_info = self._model.get_details()
        return self._deployment_info

    @classmethod
    def class_name(cls) -> str:
        """Get Class Name."""
        return "WatsonxLLM"

    def _get_credential_kwargs(self) -> Dict[str, SecretStr | None]:
        return {
            "url": self.url,
            "apikey": self.apikey,
            "token": self.token,
            "password": self.password,
            "username": self.username,
            "instance_id": self.instance_id,
            "version": self.version,
        }

    @property
    def metadata(self) -> LLMMetadata:
        if self.model_id:
            return LLMMetadata(
                context_window=(
                    self.model_info.get("model_limits", {}).get("max_sequence_length")
                ),
                num_output=(self.max_new_tokens or DEFAULT_MAX_TOKENS),
                model_name=self.model_id,
            )
        else:
            model_id = self.deployment_info.get("entity", {}).get("base_model_id")
            context_window = (
                self._model._client.foundation_models.get_model_specs(model_id=model_id)
                .get("model_limits", {})
                .get("max_sequence_length")
            )
            return LLMMetadata(
                context_window=context_window
                or self._context_window
                or DEFAULT_CONTEXT_WINDOW,
                num_output=(self.max_new_tokens or DEFAULT_MAX_TOKENS),
                model_name=model_id or self._model.deployment_id,
            )

    @property
    def sample_generation_text_params(self) -> Dict[str, Any]:
        """Example of Model generation text kwargs that a user can pass to the model."""
        return GenTextParamsMetaNames().get_example_values()

    @property
    def sample_chat_generation_params(self) -> Dict[str, Any]:
        """Example of Model chat generation kwargs that a user can pass to the model."""
        return GenChatParamsMetaNames().get_example_values()

    def _split_generation_params(
        self, data: Dict[str, Any]
    ) -> Tuple[Dict[str, Any] | None, Dict[str, Any]]:
        params = {}
        kwargs = {}
        sample_generation_kwargs_keys = set(self.sample_generation_text_params.keys())
        sample_generation_kwargs_keys.add("prompt_variables")
        for key, value in data.items():
            if key in sample_generation_kwargs_keys:
                params.update({key: value})
            else:
                kwargs.update({key: value})
        return params if params else None, kwargs

    def _split_chat_generation_params(
        self, data: Dict[str, Any]
    ) -> Tuple[Dict[str, Any] | None, Dict[str, Any]]:
        params = {}
        kwargs = {}
        sample_generation_kwargs_keys = set(self.sample_chat_generation_params.keys())
        for key, value in data.items():
            if key in sample_generation_kwargs_keys:
                params.update({key: value})
            else:
                kwargs.update({key: value})
        return params if params else None, kwargs

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        params, generation_kwargs = self._split_generation_params(kwargs)
        if "use_completions" in generation_kwargs:
            del generation_kwargs["use_completions"]
        response = self._model.generate(
            prompt=prompt,
            params=self._text_generation_params or params,
            **generation_kwargs,
        )

        return CompletionResponse(
            text=self._model._return_guardrails_stats(response).get("generated_text"),
            raw=response,
        )

    @llm_completion_callback()
    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        params, generation_kwargs = self._split_generation_params(kwargs)
        if "use_completions" in generation_kwargs:
            del generation_kwargs["use_completions"]

        response = await self._model.agenerate(
            prompt=prompt,
            params=self._text_generation_params or params,
            **generation_kwargs,
        )

        return CompletionResponse(
            text=self._model._return_guardrails_stats(response).get("generated_text"),
            raw=response,
        )

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        params, generation_kwargs = self._split_generation_params(kwargs)

        stream_response = self._model.generate_text_stream(
            prompt=prompt,
            params=self._text_generation_params or params,
            **generation_kwargs,
        )

        def gen() -> CompletionResponseGen:
            content = ""
            if kwargs.get("raw_response"):
                for stream_delta in stream_response:
                    stream_delta_text = self._model._return_guardrails_stats(
                        stream_delta
                    ).get("generated_text", "")
                    content += stream_delta_text
                    yield CompletionResponse(
                        text=content, delta=stream_delta_text, raw=stream_delta
                    )
            else:
                for stream_delta in stream_response:
                    content += stream_delta
                    yield CompletionResponse(text=content, delta=stream_delta)

        return gen()

    @llm_completion_callback()
    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        async def gen() -> CompletionResponseAsyncGen:
            for message in self.stream_complete(prompt, formatted=formatted, **kwargs):
                yield message

        # NOTE: convert generator to async generator
        return gen()

    def _chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        message_dicts = [to_watsonx_message_dict(message) for message in messages]

        params, generation_kwargs = self._split_chat_generation_params(kwargs)
        response = self._model.chat(
            messages=message_dicts,
            params=params,
            tools=generation_kwargs.get("tools"),
            tool_choice=generation_kwargs.get("tool_choice"),
            tool_choice_option=generation_kwargs.get("tool_choice_option"),
        )

        wx_message = response["choices"][0]["message"]
        message = from_watsonx_message(wx_message)

        return ChatResponse(
            message=message,
            raw=response,
        )

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        if kwargs.get("use_completions"):
            chat_fn = completion_to_chat_decorator(self.complete)
        else:
            chat_fn = self._chat

        return chat_fn(messages, **kwargs)

    async def _achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        message_dicts = [to_watsonx_message_dict(message) for message in messages]

        params, generation_kwargs = self._split_chat_generation_params(kwargs)
        response = await self._model.achat(
            messages=message_dicts,
            params=params,
            tools=generation_kwargs.get("tools"),
            tool_choice=generation_kwargs.get("tool_choice"),
            tool_choice_option=generation_kwargs.get("tool_choice_option"),
        )

        wx_message = response["choices"][0]["message"]
        message = from_watsonx_message(wx_message)

        return ChatResponse(
            message=message,
            raw=response,
        )

    @llm_chat_callback()
    async def achat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponse:
        if kwargs.get("use_completions"):
            achat_fn = acompletion_to_chat_decorator(self.acomplete)
        else:
            achat_fn = self._achat

        return await achat_fn(messages, **kwargs)

    def _stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        message_dicts = [to_watsonx_message_dict(message) for message in messages]

        params, generation_kwargs = self._split_chat_generation_params(kwargs)
        stream_response = self._model.chat_stream(
            messages=message_dicts,
            params=params,
            tools=generation_kwargs.get("tools"),
            tool_choice=generation_kwargs.get("tool_choice"),
            tool_choice_option=generation_kwargs.get("tool_choice_option"),
        )

        def stream_gen() -> ChatResponseGen:
            content = ""
            role = None
            tool_calls = []

            for response in stream_response:
                tools_available = False
                wx_message = response["choices"][0]["delta"]

                role = wx_message.get("role") or role or MessageRole.ASSISTANT
                delta = wx_message.get("content", "")
                content += delta

                if "tool_calls" in wx_message:
                    tools_available = True

                additional_kwargs = {}
                if tools_available:
                    tool_calls = update_tool_calls(tool_calls, wx_message["tool_calls"])
                    if tool_calls:
                        additional_kwargs["tool_calls"] = tool_calls

                yield ChatResponse(
                    message=ChatMessage(
                        role=role,
                        content=content,
                        additional_kwargs=additional_kwargs,
                    ),
                    delta=delta,
                    raw=response,
                    additional_kwargs=self._get_response_token_counts(response),
                )

        return stream_gen()

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        if kwargs.get("use_completions"):
            chat_stream_fn = stream_completion_to_chat_decorator(self.stream_complete)
        else:
            chat_stream_fn = self._stream_chat

        return chat_stream_fn(messages, **kwargs)

    @llm_chat_callback()
    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        async def gen() -> ChatResponseAsyncGen:
            for message in self.stream_chat(messages, **kwargs):
                yield message

        # NOTE: convert generator to async generator
        return gen()

    def _prepare_chat_with_tools(
        self,
        tools: List["BaseTool"],
        user_msg: Optional[Union[str, ChatMessage]] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        tool_choice: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Predict and call the tool."""
        # watsonx uses the same openai tool format
        tool_specs = [tool.metadata.to_openai_tool() for tool in tools]

        if isinstance(user_msg, str):
            user_msg = ChatMessage(role=MessageRole.USER, content=user_msg)

        messages = chat_history or []
        if user_msg:
            messages.append(user_msg)

        chat_with_tools_payload = {
            "messages": messages,
            "tools": tool_specs or None,
            **kwargs,
        }
        if tool_choice is not None:
            chat_with_tools_payload.update(
                {"tool_choice": {"type": "function", "function": {"name": tool_choice}}}
            )
        return chat_with_tools_payload

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
            if not isinstance(tool_call, dict):
                raise ValueError("Invalid tool_call object")
            if tool_call.get("type") != "function":
                raise ValueError("Invalid tool type. Unsupported by watsonx.ai")

            # this should handle both complete and partial jsons
            try:
                argument_dict = parse_partial_json(
                    tool_call.get("function", {}).get("arguments")
                )
            except ValueError:
                argument_dict = {}

            tool_selections.append(
                ToolSelection(
                    tool_id=tool_call.get("id"),
                    tool_name=tool_call.get("function").get("name"),
                    tool_kwargs=argument_dict,
                )
            )

        return tool_selections

    def _get_response_token_counts(self, raw_response: Any) -> dict:
        """Get the token usage reported by the response."""
        if isinstance(raw_response, dict):
            usage = raw_response.get("usage", {})
            if usage is None:
                return {}

            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)
        else:
            return {}

        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }
