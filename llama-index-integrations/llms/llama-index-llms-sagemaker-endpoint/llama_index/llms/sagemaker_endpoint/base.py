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
from llama_index.core.utilities.aws_utils import get_aws_service_client
from llama_index.llms.llama_cpp.llama_utils import (
    completion_to_prompt,
    messages_to_prompt,
)
from llama_index.llms.sagemaker_endpoint.utils import BaseIOHandler, IOHandler

import warnings

DEFAULT_IO_HANDLER = IOHandler()
LLAMA_MESSAGES_TO_PROMPT = messages_to_prompt
LLAMA_COMPLETION_TO_PROMPT = completion_to_prompt


class SageMakerLLM(LLM):
    r"""SageMaker LLM.

    Examples:
        `pip install llama-index-llms-sagemaker-endpoint`

        ```python
        from llama_index.llms.sagemaker import SageMakerLLM

        # hooks for HuggingFaceH4/zephyr-7b-beta
        # different models may require different formatting
        def messages_to_prompt(messages):
            prompt = ""
            for message in messages:
                if message.role == 'system':
                prompt += f"<|system|>\n{message.content}</s>\n"
                elif message.role == 'user':
                prompt += f"<|user|>\n{message.content}</s>\n"
                elif message.role == 'assistant':
                prompt += f"<|assistant|>\n{message.content}</s>\n"

            # ensure we start with a system prompt, insert blank if needed
            if not prompt.startswith("<|system|>\n"):
                prompt = "<|system|>\n</s>\n" + prompt

            # add final assistant prompt
            prompt = prompt + "<|assistant|>\n"

            return prompt

        def completion_to_prompt(completion):
            return f"<|system|>\n</s>\n<|user|>\n{completion}</s>\n<|assistant|>\n"

        # Additional setup for SageMakerLLM class
        model_name = "HuggingFaceH4/zephyr-7b-beta"
        api_key = "your_api_key"
        region = "your_region"

        llm = SageMakerLLM(
            model_name=model_name,
            api_key=api_key,
            region=region,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
        )
        ```
    """

    endpoint_name: str = Field(description="SageMaker LLM endpoint name")
    endpoint_kwargs: Dict[str, Any] = Field(
        default={},
        description="Additional kwargs for the invoke_endpoint request.",
    )
    model_kwargs: Dict[str, Any] = Field(
        default={},
        description="kwargs to pass to the model.",
    )
    content_handler: BaseIOHandler = Field(
        default=DEFAULT_IO_HANDLER,
        description="used to serialize input, deserialize output, and remove a prefix.",
    )

    profile_name: Optional[str] = Field(
        description="The name of aws profile to use. If not given, then the default profile is used."
    )
    aws_access_key_id: Optional[str] = Field(description="AWS Access Key ID to use")
    aws_secret_access_key: Optional[str] = Field(
        description="AWS Secret Access Key to use"
    )
    aws_session_token: Optional[str] = Field(description="AWS Session Token to use")
    region_name: Optional[str] = Field(
        description="AWS region name to use. Uses region configured in AWS CLI if not passed"
    )
    max_retries: Optional[int] = Field(
        default=3,
        description="The maximum number of API retries.",
        ge=0,
    )
    timeout: Optional[float] = Field(
        default=60.0,
        description="The timeout, in seconds, for API requests.",
        ge=0,
    )
    _client: Any = PrivateAttr()
    _completion_to_prompt: Callable[[str, Optional[str]], str] = PrivateAttr()

    def __init__(
        self,
        endpoint_name: str,
        endpoint_kwargs: Optional[Dict[str, Any]] = {},
        model_kwargs: Optional[Dict[str, Any]] = {},
        content_handler: Optional[BaseIOHandler] = DEFAULT_IO_HANDLER,
        profile_name: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        aws_region_name: Optional[str] = None,
        max_retries: Optional[int] = 3,
        timeout: Optional[float] = 60.0,
        temperature: Optional[float] = 0.5,
        callback_manager: Optional[CallbackManager] = None,
        system_prompt: Optional[str] = None,
        messages_to_prompt: Optional[
            Callable[[Sequence[ChatMessage]], str]
        ] = LLAMA_MESSAGES_TO_PROMPT,
        completion_to_prompt: Callable[
            [str, Optional[str]], str
        ] = LLAMA_COMPLETION_TO_PROMPT,
        pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
        output_parser: Optional[BaseOutputParser] = None,
        **kwargs: Any,
    ) -> None:
        if not endpoint_name:
            raise ValueError(
                "Missing required argument:`endpoint_name`"
                " Please specify the endpoint_name"
            )
        endpoint_kwargs = endpoint_kwargs or {}
        model_kwargs = model_kwargs or {}
        model_kwargs["temperature"] = temperature
        content_handler = content_handler
        callback_manager = callback_manager or CallbackManager([])

        super().__init__(
            endpoint_name=endpoint_name,
            endpoint_kwargs=endpoint_kwargs,
            model_kwargs=model_kwargs,
            content_handler=content_handler,
            profile_name=profile_name,
            region_name=region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            timeout=timeout,
            max_retries=max_retries,
            callback_manager=callback_manager,
            system_prompt=system_prompt,
            messages_to_prompt=messages_to_prompt,
            pydantic_program_mode=pydantic_program_mode,
            output_parser=output_parser,
        )
        self._completion_to_prompt = completion_to_prompt

        region_name = kwargs.pop("region_name", None)
        if region_name is not None:
            warnings.warn(
                "Kwarg `region_name` is deprecated and will be removed in a future version. "
                "Please use `aws_region_name` instead.",
                DeprecationWarning,
            )
            if not aws_region_name:
                aws_region_name = region_name

        self._client = get_aws_service_client(
            service_name="sagemaker-runtime",
            profile_name=profile_name,
            region_name=aws_region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            max_retries=max_retries,
            timeout=timeout,
        )

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        model_kwargs = {**self.model_kwargs, **kwargs}
        if not formatted:
            prompt = self._completion_to_prompt(prompt, self.system_prompt)

        request_body = self.content_handler.serialize_input(prompt, model_kwargs)
        response = self._client.invoke_endpoint(
            EndpointName=self.endpoint_name,
            Body=request_body,
            ContentType=self.content_handler.content_type,
            Accept=self.content_handler.accept,
            **self.endpoint_kwargs,
        )

        response["Body"] = self.content_handler.deserialize_output(response["Body"])
        text = self.content_handler.remove_prefix(response["Body"], prompt)

        return CompletionResponse(
            text=text,
            raw=response,
            additional_kwargs={
                "model_kwargs": model_kwargs,
                "endpoint_kwargs": self.endpoint_kwargs,
            },
        )

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        model_kwargs = {**self.model_kwargs, **kwargs}
        if not formatted:
            prompt = self._completion_to_prompt(prompt, self.system_prompt)

        request_body = self.content_handler.serialize_input(prompt, model_kwargs)

        def gen() -> CompletionResponseGen:
            raw_text = ""
            prev_clean_text = ""
            for response in self._client.invoke_endpoint_with_response_stream(
                EndpointName=self.endpoint_name,
                Body=request_body,
                ContentType=self.content_handler.content_type,
                Accept=self.content_handler.accept,
                **self.endpoint_kwargs,
            )["Body"]:
                delta = self.content_handler.deserialize_streaming_output(
                    response["PayloadPart"]["Bytes"]
                )
                raw_text += delta
                clean_text = self.content_handler.remove_prefix(raw_text, prompt)
                delta = clean_text[len(prev_clean_text) :]
                prev_clean_text = clean_text

                yield CompletionResponse(text=clean_text, delta=delta, raw=response)

        return gen()

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        prompt = self.messages_to_prompt(messages)
        completion_response = self.complete(prompt, formatted=True, **kwargs)
        return completion_response_to_chat_response(completion_response)

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        prompt = self.messages_to_prompt(messages)
        completion_response_gen = self.stream_complete(prompt, formatted=True, **kwargs)
        return stream_completion_response_to_chat_response(completion_response_gen)

    @llm_chat_callback()
    async def achat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponse:
        raise NotImplementedError

    @llm_chat_callback()
    async def astream_chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponseAsyncGen:
        raise NotImplementedError

    @llm_completion_callback()
    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        raise NotImplementedError

    @llm_completion_callback()
    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        raise NotImplementedError

    @classmethod
    def class_name(cls) -> str:
        return "SageMakerLLM"

    @property
    def metadata(self) -> LLMMetadata:
        """LLM metadata."""
        return LLMMetadata(
            model_name=self.endpoint_name,
        )


# Deprecated, kept for backwards compatibility
SageMakerLLMEndPoint = SageMakerLLM
