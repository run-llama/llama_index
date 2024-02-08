import json
from typing import Any, Callable, Dict, Optional, Sequence

from llama_index.bridge.pydantic import Field, PrivateAttr
from llama_index.callbacks import CallbackManager
from llama_index.constants import (
    DEFAULT_TEMPERATURE,
)
from llama_index.core.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.llms.base import (
    llm_chat_callback,
    llm_completion_callback,
)
from llama_index.llms.bedrock_utils import (
    BEDROCK_FOUNDATION_LLMS,
    CHAT_ONLY_MODELS,
    STREAMING_MODELS,
    Provider,
    completion_with_retry,
    get_provider,
)
from llama_index.llms.generic_utils import (
    completion_response_to_chat_response,
    stream_completion_response_to_chat_response,
)
from llama_index.llms.llm import LLM
from llama_index.types import BaseOutputParser, PydanticProgramMode


class Bedrock(LLM):
    model: str = Field(description="The modelId of the Bedrock model to use.")
    temperature: float = Field(description="The temperature to use for sampling.")
    max_tokens: int = Field(description="The maximum number of tokens to generate.")
    context_size: int = Field("The maximum number of tokens available for input.")
    profile_name: Optional[str] = Field(
        description="The name of aws profile to use. If not given, then the default profile is used."
    )
    aws_access_key_id: Optional[str] = Field(
        description="AWS Access Key ID to use", exclude=True
    )
    aws_secret_access_key: Optional[str] = Field(
        description="AWS Secret Access Key to use", exclude=True
    )
    aws_session_token: Optional[str] = Field(
        description="AWS Session Token to use", exclude=True
    )
    region_name: Optional[str] = Field(
        description="AWS region name to use. Uses region configured in AWS CLI if not passed",
        exclude=True,
    )
    botocore_session: Optional[Any] = Field(
        description="Use this Botocore session instead of creating a new default one.",
        exclude=True,
    )
    botocore_config: Optional[Any] = Field(
        description="Custom configuration object to use instead of the default generated one.",
        exclude=True,
    )
    max_retries: int = Field(
        default=10, description="The maximum number of API retries.", gt=0
    )
    timeout: float = Field(
        default=60.0,
        description="The timeout for the Bedrock API request in seconds. It will be used for both connect and read timeouts.",
    )
    additional_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional kwargs for the bedrock invokeModel request.",
    )

    _client: Any = PrivateAttr()
    _aclient: Any = PrivateAttr()
    _provider: Provider = PrivateAttr()

    def __init__(
        self,
        model: str,
        temperature: Optional[float] = DEFAULT_TEMPERATURE,
        max_tokens: Optional[int] = 512,
        context_size: Optional[int] = None,
        profile_name: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        region_name: Optional[str] = None,
        botocore_session: Optional[Any] = None,
        client: Optional[Any] = None,
        timeout: Optional[float] = 60.0,
        max_retries: Optional[int] = 10,
        botocore_config: Optional[Any] = None,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        callback_manager: Optional[CallbackManager] = None,
        system_prompt: Optional[str] = None,
        messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]] = None,
        completion_to_prompt: Optional[Callable[[str], str]] = None,
        pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
        output_parser: Optional[BaseOutputParser] = None,
        **kwargs: Any,
    ) -> None:
        if context_size is None and model not in BEDROCK_FOUNDATION_LLMS:
            raise ValueError(
                "`context_size` argument not provided and"
                "model provided refers to a non-foundation model."
                " Please specify the context_size"
            )

        session_kwargs = {
            "profile_name": profile_name,
            "region_name": region_name,
            "aws_access_key_id": aws_access_key_id,
            "aws_secret_access_key": aws_secret_access_key,
            "aws_session_token": aws_session_token,
            "botocore_session": botocore_session,
        }
        config = None
        try:
            import boto3
            from botocore.config import Config

            config = (
                Config(
                    retries={"max_attempts": max_retries, "mode": "standard"},
                    connect_timeout=timeout,
                    read_timeout=timeout,
                )
                if botocore_config is None
                else botocore_config
            )
            session = boto3.Session(**session_kwargs)
        except ImportError:
            raise ImportError(
                "boto3 package not found, install with" "'pip install boto3'"
            )

        # Prior to general availability, custom boto3 wheel files were
        # distributed that used the bedrock service to invokeModel.
        # This check prevents any services still using those wheel files
        # from breaking
        if client is not None:
            self._client = client
        elif "bedrock-runtime" in session.get_available_services():
            self._client = session.client("bedrock-runtime", config=config)
        else:
            self._client = session.client("bedrock", config=config)

        additional_kwargs = additional_kwargs or {}
        callback_manager = callback_manager or CallbackManager([])
        context_size = context_size or BEDROCK_FOUNDATION_LLMS[model]
        self._provider = get_provider(model)
        messages_to_prompt = messages_to_prompt or self._provider.messages_to_prompt
        completion_to_prompt = (
            completion_to_prompt or self._provider.completion_to_prompt
        )
        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            context_size=context_size,
            profile_name=profile_name,
            timeout=timeout,
            max_retries=max_retries,
            botocore_config=config,
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
        return "Bedrock_LLM"

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
            self._provider.max_tokens_key: self.max_tokens,
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
        all_kwargs = self._get_all_kwargs(**kwargs)
        request_body = self._provider.get_request_body(prompt, all_kwargs)
        request_body_str = json.dumps(request_body)
        response = completion_with_retry(
            client=self._client,
            model=self.model,
            request_body=request_body_str,
            max_retries=self.max_retries,
            **all_kwargs,
        )["body"].read()
        response = json.loads(response)
        return CompletionResponse(
            text=self._provider.get_text_from_response(response), raw=response
        )

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        if self.model in BEDROCK_FOUNDATION_LLMS and self.model not in STREAMING_MODELS:
            raise ValueError(f"Model {self.model} does not support streaming")

        if not formatted:
            prompt = self.completion_to_prompt(prompt)

        all_kwargs = self._get_all_kwargs(**kwargs)
        request_body = self._provider.get_request_body(prompt, all_kwargs)
        request_body_str = json.dumps(request_body)
        response = completion_with_retry(
            client=self._client,
            model=self.model,
            request_body=request_body_str,
            max_retries=self.max_retries,
            stream=True,
            **all_kwargs,
        )["body"]

        def gen() -> CompletionResponseGen:
            content = ""
            for r in response:
                r = json.loads(r["chunk"]["bytes"])
                content_delta = self._provider.get_text_from_stream_response(r)
                content += content_delta
                yield CompletionResponse(text=content, delta=content_delta, raw=r)

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

    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        """Chat asynchronously."""
        # TODO: do synchronous chat for now
        return self.chat(messages, **kwargs)

    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        raise NotImplementedError

    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        raise NotImplementedError

    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        raise NotImplementedError
