import json
from typing import Any, Dict, Optional, Sequence

from llama_index.bridge.pydantic import Field, PrivateAttr
from llama_index.callbacks import CallbackManager
from llama_index.llms.base import (
    LLM,
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    LLMMetadata,
    llm_chat_callback,
    llm_completion_callback,
)
from llama_index.llms.bedrock_utils import (
    BEDROCK_FOUNDATION_LLMS,
    STREAMING_MODELS,
    bedrock_model_to_param_name,
    completion_to_chat_decorator,
    completion_with_retry,
    get_request_body,
    get_text_from_response,
    stream_completion_to_chat_decorator,
)


class Bedrock(LLM):
    model: str = Field(description="The modelId of the Bedrock model to use.")
    temperature: float = Field(description="The temperature to use for sampling.")
    max_tokens: int = Field(description="The maximum number of tokens to generate.")
    context_size: int = Field("The maximum number of tokens available for input.")
    profile_name: Optional[str] = Field(
        description="The name of aws profile to use. If not given, then the default profile is used."
    )
    aws_access_key_id: Optional[str] = Field(description="AWS Access Key ID to use")
    aws_secret_access_key: Optional[str] = Field(
        description="AWS Secret Access Key to use"
    )
    aws_session_token: Optional[str] = Field(description="AWS Session Token to use")
    aws_region_name: Optional[str] = Field(
        description="AWS region name to use. Uses region configured in AWS CLI if not passed"
    )
    max_retries: int = Field(
        default=10, description="The maximum number of API retries."
    )
    additional_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional kwargs for the bedrock invokeModel request.",
    )

    _client: Any = PrivateAttr()
    _aclient: Any = PrivateAttr()

    def __init__(
        self,
        model: str,
        temperature: Optional[float] = 0.5,
        max_tokens: Optional[int] = 512,
        context_size: Optional[int] = None,
        profile_name: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        aws_region_name: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = 10,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        if context_size is None and model not in BEDROCK_FOUNDATION_LLMS:
            raise ValueError(
                "`context_size` argument not provided and"
                "model provided refers to a non-foundation model."
                " Please specify the context_size"
            )
        try:
            import boto3
            import botocore

        except Exception as e:
            raise ImportError(
                "You must install the `boto3` package to use Bedrock."
                "Please `pip install boto3`"
            ) from e
        try:
            if not profile_name and aws_access_key_id:
                session = boto3.Session(
                    aws_access_key_id=aws_access_key_id,
                    aws_secret_access_key=aws_secret_access_key,
                    aws_session_token=aws_session_token,
                    region_name=aws_region_name,
                )
            else:
                session = boto3.Session(profile_name=profile_name)
            # Prior to general availability, custom boto3 wheel files were
            # distributed that used the bedrock service to invokeModel.
            # This check prevents any services still using those wheel files
            # from breaking
            if "bedrock-runtime" in session.get_available_services():
                self._client = session.client("bedrock-runtime")
            else:
                self._client = session.client("bedrock")

        except botocore.exceptions.NoRegionError as e:
            raise ValueError(
                "If default region is not set in AWS CLI, you must provide"
                " the region_name argument to llama_index.llms.Bedrock"
            )

        additional_kwargs = additional_kwargs or {}
        callback_manager = callback_manager or CallbackManager([])
        context_size = context_size or BEDROCK_FOUNDATION_LLMS[model]
        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            context_size=context_size,
            profile_name=profile_name,
            timeout=timeout,
            max_retries=max_retries,
            additional_kwargs=additional_kwargs,
            callback_manager=callback_manager,
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
            is_chat_model=True,
            model_name=self.model,
        )

    @property
    def _model_kwargs(self) -> Dict[str, Any]:
        max_tokens_key = bedrock_model_to_param_name(
            model=self.model, param_name="max_tokens"
        )
        base_kwargs = {"temperature": self.temperature, max_tokens_key: self.max_tokens}
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
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        all_kwargs = self._get_all_kwargs(**kwargs)
        provider = self.model.split(".")[0]
        request_body = get_request_body(provider, prompt, all_kwargs)
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
            text=get_text_from_response(provider, response), raw=response
        )

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        if self.model in BEDROCK_FOUNDATION_LLMS and self.model not in STREAMING_MODELS:
            raise ValueError(f"Model {self.model} does not support streaming")
        all_kwargs = self._get_all_kwargs(**kwargs)
        provider = self.model.split(".")[0]
        request_body = get_request_body(provider, prompt, all_kwargs)
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
                content_delta = get_text_from_response(provider, r, stream=True)
                content += content_delta
                yield CompletionResponse(text=content, delta=content_delta, raw=r)

        return gen()

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        chat_fn = completion_to_chat_decorator(self.complete)
        return chat_fn(messages, **kwargs)

    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        stream_chat_fn = stream_completion_to_chat_decorator(self.stream_complete)
        return stream_chat_fn(messages, **kwargs)

    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        raise NotImplementedError

    async def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        raise NotImplementedError

    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        raise NotImplementedError

    async def astream_complete(
        self, prompt: str, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        raise NotImplementedError
