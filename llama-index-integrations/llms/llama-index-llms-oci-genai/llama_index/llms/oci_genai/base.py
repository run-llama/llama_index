import json
from typing import Any, Callable, Dict, Optional, Sequence, List, Union

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
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CallbackManager

from llama_index.core.constants import (
    DEFAULT_TEMPERATURE,
)
from llama_index.core.llms.callbacks import (
    llm_chat_callback,
    llm_completion_callback,
)
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.types import BaseOutputParser, PydanticProgramMode

from utils import (
    CHAT_MODELS,
    create_client,
    get_provider,
    get_serving_mode,
    get_completion_generator,
    get_chat_generator,
    get_context_size,
)


class OCIGenAI(FunctionCallingLLM):
    """OCI large language models with function calling support."""

    model: str = Field(description="Id of the OCI Generative AI model to use.")
    temperature: float = Field(description="The temperature to use for sampling.")
    max_tokens: int = Field(description="The maximum number of tokens to generate.")
    context_size: int = Field("The maximum number of tokens available for input.")

    service_endpoint: Optional[str] = Field(
        default=None,
        description="service endpoint url.",
    )

    compartment_id: Optional[str] = Field(
        default=None,
        description="OCID of compartment.",
    )

    auth_type: Optional[str] = Field(
        description="Authentication type, can be: API_KEY, SECURITY_TOKEN, INSTANCE_PRINCIPAL, RESOURCE_PRINCIPAL. If not specified, API_KEY will be used",
        default="API_KEY",
    )

    auth_profile: Optional[str] = Field(
        description="The name of the profile in ~/.oci/config. If not specified , DEFAULT will be used",
        default="DEFAULT",
    )

    additional_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional kwargs for the OCI Generative AI request.",
    )

    _client: Any = PrivateAttr()
    _provider: str = PrivateAttr()
    _serving_mode: str = PrivateAttr()
    _completion_generator: str = PrivateAttr()
    _chat_generator: str = PrivateAttr()

    def __init__(
        self,
        model: str,
        temperature: Optional[float] = DEFAULT_TEMPERATURE,
        max_tokens: Optional[int] = 512,
        context_size: Optional[int] = None,
        service_endpoint: Optional[str] = None,
        compartment_id: Optional[str] = None,
        auth_type: Optional[str] = "API_KEY",
        auth_profile: Optional[str] = "DEFAULT",
        client: Optional[Any] = None,
        provider: Optional[str] = None,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        callback_manager: Optional[CallbackManager] = None,
        system_prompt: Optional[str] = None,
        messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]] = None,
        completion_to_prompt: Optional[Callable[[str], str]] = None,
        pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
        output_parser: Optional[BaseOutputParser] = None,
    ) -> None:
        """
        Initializes the OCIGenAI class.

        Args:
            model (str): The Id of the model to be used for generating embeddings, e.g., "meta.llama-2-70b-chat".

            temperature (Optional[float]): The temperature to use for sampling. Default specified in lama_index.core.constants.DEFAULT_TEMPERATURE.

            max_tokens (Optional[int]): The maximum number of tokens to generate. Default is 512.

            context_size (Optional[int]): The maximum number of tokens available for input. If not specified, the default context size for the model will be used.

            service_endpoint (str): service endpoint url, e.g., "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"

            compartment_id (str): OCID of the compartment.

            auth_type (Optional[str]): Authentication type, can be: API_KEY (default), SECURITY_TOKEN, INSTANCEAL, RESOURCE_PRINCIPAL.
                                    If not specified, API_KEY will be used

            auth_profile (Optional[str]): The name of the profile in ~/.oci/config. If not specified , DEFAULT will be used

            client (Optional[Any]): An optional OCI client object. If not provided, the client will be created using the
                                    provided service endpoint and authentifcation method.

            provider (Optional[str]): Provider name of the model. If not specified, the provider will be derived from the model name.

            additional_kwargs (Optional[Dict[str, Any]]): Additional kwargs for the the LLM.
        """
        context_size = get_context_size(model, context_size)

        additional_kwargs = additional_kwargs or {}
        callback_manager = callback_manager or CallbackManager([])

        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            context_size=context_size,
            service_endpoint=service_endpoint,
            compartment_id=compartment_id,
            auth_type=auth_type,
            auth_profile=auth_profile,
            additional_kwargs=additional_kwargs,
            callback_manager=callback_manager,
            system_prompt=system_prompt,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            pydantic_program_mode=pydantic_program_mode,
            output_parser=output_parser,
        )

        self._client = client or create_client(
            auth_type, auth_profile, service_endpoint
        )

        self._provider = get_provider(model, provider)

        self._serving_mode = get_serving_mode(model)

        self._completion_generator = get_completion_generator()

        self._chat_generator = get_chat_generator()

    @classmethod
    def class_name(cls) -> str:
        return "OCIGenAI_LLM"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_size,
            num_output=self.max_tokens,
            is_chat_model=self.model in CHAT_MODELS,
            model_name=self.model,
        )

    @property
    def _model_kwargs(self) -> Dict[str, Any]:
        base_kwargs = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
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

    @llm_completion_callback()
    def complete(
            self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        inference_params = self._get_all_kwargs(**kwargs)
        inference_params["is_stream"] = False
        inference_params["prompt"] = prompt

        request = self._completion_generator(
            compartment_id=self.compartment_id,
            serving_mode=self._serving_mode,
            inference_request=self._provider.oci_completion_request(**inference_params),
        )

        response = self._client.generate_text(request)
        return CompletionResponse(
            text=self._provider.completion_response_to_text(response),
            raw=response.__dict__,
        )

    @llm_completion_callback()
    def stream_complete(
            self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        inference_params = self._get_all_kwargs(**kwargs)
        inference_params["is_stream"] = True
        inference_params["prompt"] = prompt

        request = self._completion_generator(
            compartment_id=self.compartment_id,
            serving_mode=self._serving_mode,
            inference_request=self._provider.oci_completion_request(**inference_params),
        )

        response = self._client.generate_text(request)

        def gen() -> CompletionResponseGen:
            content = ""
            for event in response.data.events():
                content_delta = self._provider.completion_stream_to_text(
                    json.loads(event.data)
                )
                content += content_delta
                yield CompletionResponse(
                    text=content, delta=content_delta, raw=event.__dict__
                )

        return gen()

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        oci_params = self._provider.messages_to_oci_params(messages)
        oci_params["is_stream"] = False
        all_kwargs = self._get_all_kwargs(**kwargs)
        chat_params = {**all_kwargs, **oci_params}

        request = self._chat_generator(
            compartment_id=self.compartment_id,
            serving_mode=self._serving_mode,
            chat_request=self._provider.oci_chat_request(**chat_params),
        )

        response = self._client.chat(request)

        generation_info = self._provider.chat_generation_info(response)

        llm_output = {
            "model_id": response.data.model_id,
            "model_version": response.data.model_version,
            "request_id": response.request_id,
            "content-length": response.headers["content-length"],
        }

        return ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT,
                content=self._provider.chat_response_to_text(response),
                additional_kwargs=generation_info,
            ),
            raw=response.__dict__,
            additional_kwargs=llm_output
        )

    def stream_chat(
            self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        oci_params = self._provider.messages_to_oci_params(messages)
        oci_params["is_stream"] = True
        all_kwargs = self._get_all_kwargs(**kwargs)
        chat_params = {**all_kwargs, **oci_params}

        request = self._chat_generator(
            compartment_id=self.compartment_id,
            serving_mode=self._serving_mode,
            chat_request=self._provider.oci_chat_request(**chat_params),
        )

        response = self._client.chat(request)

        def gen() -> ChatResponseGen:
            content = ""
            for event in response.data.events():
                content_delta = self._provider.chat_stream_to_text(
                    json.loads(event.data)
                )
                content += content_delta
                yield ChatResponse(
                    message=ChatMessage(role=MessageRole.ASSISTANT, content=content),
                    delta=content_delta,
                    raw=event.__dict__,
                )

        return gen()

    # Function tooling integration methods
    def _prepare_chat_with_tools(
            self,
            tools: Sequence["BaseTool"],
            user_msg: Optional[Union[str, ChatMessage]] = None,
            chat_history: Optional[List[ChatMessage]] = None,
            **kwargs: Any,
    ) -> Dict[str, Any]:
        tool_specs = [self._provider.convert_to_oci_tool(tool) for tool in tools]

        if isinstance(user_msg, str):
            user_msg = ChatMessage(role=MessageRole.USER, content=user_msg)

        messages = chat_history or []
        if user_msg:
            messages.append(user_msg)

        oci_params = self._provider.messages_to_oci_params(messages)
        chat_params = self._get_all_kwargs(**kwargs)

        return {
            "messages": messages,
            "tools": tool_specs,
            **oci_params,
            **chat_params,
        }

    def chat_with_tools(
            self,
            tools: Sequence["BaseTool"],
            user_msg: Optional[Union[str, ChatMessage]] = None,
            chat_history: Optional[List[ChatMessage]] = None,
            **kwargs: Any,
    ) -> ChatResponse:
        chat_kwargs = self._prepare_chat_with_tools(
            tools,
            user_msg=user_msg,
            chat_history=chat_history,
            **kwargs,
        )
        response = self.chat(**chat_kwargs)
        return self._validate_chat_with_tools_response(
            response,
            tools,
            **kwargs,
        )

    def _validate_chat_with_tools_response(
            self,
            response: ChatResponse,
            tools: Sequence["BaseTool"],
            **kwargs: Any,
    ) -> ChatResponse:
        # Placeholder for future implementation details
        return response
