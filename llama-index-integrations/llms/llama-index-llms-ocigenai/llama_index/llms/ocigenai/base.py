import json
from typing import Any, Callable, Dict, Optional, Sequence

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    LLMMetadata,
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
from llama_index.core.base.llms.generic_utils import (
    completion_response_to_chat_response,
    stream_completion_response_to_chat_response,
)
from llama_index.core.llms.llm import LLM
from llama_index.core.types import BaseOutputParser, PydanticProgramMode

from llama_index.llms.ocigenai.utils import (
    OCIGENAI_LLMS,
    STREAMING_MODELS,
    CHAT_ONLY_MODELS,
    create_client,
    get_provider,
    get_serving_mode,
    get_request_generator,
)

class OCIGenAI(LLM):
    model: str = Field(description="Id of the OCI Generative AI model to use.")
    temperature: float = Field(description="The temperature to use for sampling.")
    max_tokens: int = Field(description="The maximum number of tokens to generate.")
    context_size: int = Field("The maximum number of tokens available for input.")

    service_endpoint: str = Field(
        default=None,
        description="service endpoint url.",
    )

    compartment_id: str = Field(
        default=None,
        description="OCID of compartment.",
    )

    auth_type: Optional[str] = Field(
        description="Authentication type, can be: API_KEY, SECURITY_TOKEN, INSTANCE_PRINCIPLE, RESOURCE_PRINCIPLE. If not specified, API_KEY will be used",
        default="API_KEY"
    )

    auth_profile: Optional[str] = Field(
        description="The name of the profile in ~/.oci/config. If not specified , DEFAULT will be used",
        default="DEFAULT"
    )

    additional_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional kwargs for the OCI Generative AI request.",
    )

    _client: Any = PrivateAttr()
    _provider: str = PrivateAttr()
    _serving_mode: str = PrivateAttr()
    _request_generator: str = PrivateAttr()

    def __init__(
        self,
        model: str,
        temperature: Optional[float] = DEFAULT_TEMPERATURE,
        max_tokens: Optional[int] = 512,
        context_size: Optional[int] = None,

        service_endpoint: str = None,
        compartment_id: str = None,
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
        
        self._client = client or create_client(auth_type, auth_profile, service_endpoint)

        self._provider = get_provider(model, provider)

        self._serving_mode = get_serving_mode(model)

        self._request_generator = get_request_generator()

        additional_kwargs = additional_kwargs or {}
        callback_manager = callback_manager or CallbackManager([])
        context_size = context_size or OCIGENAI_LLMS[model]
        messages_to_prompt = messages_to_prompt or self._provider.messages_to_prompt
        completion_to_prompt = completion_to_prompt or self._provider.completion_to_prompt
       
        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            context_size=context_size,

            service_endpoint = service_endpoint,
            compartment_id = compartment_id,
            auth_type = auth_type,
            auth_profile = auth_profile,
            provider = provider,
            client = client,
            
            additional_kwargs=additional_kwargs,
            callback_manager=callback_manager,
            system_prompt=system_prompt,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            pydantic_program_mode=pydantic_program_mode,
            output_parser=output_parser,
        )

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "OCIGenAI_LLM"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_size,
            num_output=self.max_tokens,
            is_chat_model=self.model in CHAT_ONLY_MODELS,
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
        if not formatted:
            prompt = self.completion_to_prompt(prompt)
        
        inference_params = self._get_all_kwargs(**kwargs)
        inference_params["is_stream"] = False
        inference_params["prompt"] = prompt

        request = self._request_generator(
            compartment_id=self.compartment_id,
            serving_mode=self._serving_mode,
            inference_request=self._provider.oci_llm_request(**inference_params),
        )
                
        response = self._client.generate_text(request)
        return CompletionResponse(
            text=self._provider.get_text_from_response(response), raw=response.__dict__
        )

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        if self.model in OCIGENAI_LLMS and self.model not in STREAMING_MODELS:
            raise ValueError(f"Model {self.model} does not support streaming")

        if not formatted:
            prompt = self.completion_to_prompt(prompt)

        inference_params = self._get_all_kwargs(**kwargs)
        inference_params["is_stream"] = True
        inference_params["prompt"] = prompt

        request = self._request_generator(
            compartment_id=self.compartment_id,
            serving_mode=self._serving_mode,
            inference_request=self._provider.oci_llm_request(**inference_params),
        )       
        
        response = self._client.generate_text(request)

        def gen() -> CompletionResponseGen:
            content = ""
            for event in response.data.events():
                content_delta = json.loads(event.data)["text"]
                content += content_delta
                yield CompletionResponse(text=content, delta=content_delta, raw=event.__dict__)

        return gen()

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        prompt = self.messages_to_prompt(messages)
        completion_response = self.complete(prompt, formatted=True, **kwargs)
        return completion_response_to_chat_response(completion_response)

    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        prompt = self.messages_to_prompt(messages)
        completion_response = self.stream_complete(prompt, formatted=True, **kwargs)
        return stream_completion_response_to_chat_response(completion_response)

    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        # TODO: do synchronous complete for now
        return self.complete(prompt, formatted=formatted, **kwargs)
        #raise NotImplementedError
        
    
    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        """Chat asynchronously."""
        # TODO: do synchronous chat for now
        #return self.chat(messages, **kwargs)
        raise NotImplementedError

    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        raise NotImplementedError

    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        raise NotImplementedError
