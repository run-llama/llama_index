from typing import Any, Callable, Dict, Optional, Sequence

import boto3
import aioboto3
from botocore.config import Config
from llama_index.core.base.llms.types import (
    CompletionResponse,
)
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CallbackManager
from llama_index.core.constants import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_NUM_OUTPUTS,
    DEFAULT_TEMPERATURE,
)
from llama_index.core.base.llms.generic_utils import (
    messages_to_prompt as generic_messages_to_prompt,
)
from llama_index.core.multi_modal_llms import (
    MultiModalLLM,
    MultiModalLLMMetadata,
)
from llama_index.core.schema import ImageNode
from llama_index.multi_modal_llms.bedrock.utils import (
    BEDROCK_MULTI_MODAL_MODELS,
    generate_bedrock_multi_modal_message,
    resolve_bedrock_credentials,
    invoke_model_with_retry,
    invoke_model_with_retry_async,
)


class BedrockMultiModal(MultiModalLLM):
    """Bedrock Multi-Modal LLM implementation."""

    model: str = Field(description="The Multi-Modal model to use from Bedrock.")
    temperature: float = Field(description="The temperature to use for sampling.")
    max_tokens: Optional[int] = Field(
        description="The maximum numbers of tokens to generate.",
        gt=0,
    )
    context_window: Optional[int] = Field(
        description="The maximum number of context tokens for the model.",
        gt=0,
    )
    region_name: str = Field(
        default=None,
        description="AWS region name.",
    )
    aws_access_key_id: str = Field(
        default=None,
        description="AWS access key ID.",
        exclude=True,
    )
    aws_secret_access_key: str = Field(
        default=None,
        description="AWS secret access key.",
        exclude=True,
    )
    max_retries: int = Field(
        default=10,
        description="The maximum number of API retries.",
        gt=0,
    )
    timeout: float = Field(
        default=60.0,
        description="The timeout for API requests in seconds.",
        gt=0,
    )
    additional_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional kwargs for the Bedrock API.",
    )

    _messages_to_prompt: Callable = PrivateAttr()
    _completion_to_prompt: Callable = PrivateAttr()
    _client: Any = PrivateAttr()  # boto3 client
    _config: Any = PrivateAttr()  # botocore config
    _asession: Any = PrivateAttr()  # aioboto3 session

    def __init__(
        self,
        model: str = "anthropic.claude-3-sonnet-20240229-v1:0",
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: Optional[int] = 300,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        context_window: Optional[int] = DEFAULT_CONTEXT_WINDOW,
        region_name: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        max_retries: int = 10,
        timeout: float = 60.0,
        messages_to_prompt: Optional[Callable] = None,
        completion_to_prompt: Optional[Callable] = None,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        # Validate model name first
        if model not in BEDROCK_MULTI_MODAL_MODELS:
            raise ValueError(
                f"Invalid model {model}. "
                f"Available models are: {list(BEDROCK_MULTI_MODAL_MODELS.keys())}"
            )

        aws_access_key_id, aws_secret_access_key, region = resolve_bedrock_credentials(
            region_name=region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )

        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            additional_kwargs=additional_kwargs or {},
            context_window=context_window,
            region_name=region,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            max_retries=max_retries,
            timeout=timeout,
            callback_manager=callback_manager,
            **kwargs,
        )
        self._messages_to_prompt = messages_to_prompt or generic_messages_to_prompt
        self._completion_to_prompt = completion_to_prompt or (lambda x: x)
        self._config = Config(
            retries={"max_attempts": max_retries, "mode": "standard"},
            connect_timeout=timeout,
            read_timeout=timeout,
        )
        self._client = self._get_client()
        self._asession = aioboto3.Session(
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            region_name=self.region_name,
        )

    def _get_client(self) -> Any:
        """Get Bedrock client."""
        session = boto3.Session(
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            region_name=self.region_name,
        )
        return session.client("bedrock-runtime", config=self._config)

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "bedrock_multi_modal_llm"

    @property
    def metadata(self) -> MultiModalLLMMetadata:
        """Multi Modal LLM metadata."""
        return MultiModalLLMMetadata(
            num_output=self.max_tokens or DEFAULT_NUM_OUTPUTS,
            model_name=self.model,
        )

    def _get_model_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
        """Get model kwargs."""
        # For Claude models, parameters need to be part of the body
        model_kwargs = {
            "contentType": "application/json",
            "accept": "application/json",
        }

        if self.model.startswith("anthropic.claude"):
            model_kwargs["body"] = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": self.max_tokens if self.max_tokens is not None else 300,
                "temperature": self.temperature,
            }

        # Add any additional kwargs
        if "body" in model_kwargs:
            model_kwargs["body"].update(self.additional_kwargs)
            model_kwargs["body"].update(kwargs)

        return model_kwargs

    def _complete(
        self, prompt: str, image_documents: Sequence[ImageNode], **kwargs: Any
    ) -> CompletionResponse:
        """Complete the prompt with image support."""
        message = generate_bedrock_multi_modal_message(
            prompt=prompt,
            image_documents=image_documents,
        )

        # Get model kwargs and prepare the request body
        model_kwargs = self._get_model_kwargs(**kwargs)
        if "body" in model_kwargs:
            model_kwargs["body"]["messages"] = [message]
        else:
            model_kwargs["body"] = {"messages": [message]}

        # Convert body to JSON string
        if isinstance(model_kwargs.get("body"), dict):
            import json

            body_str = json.dumps(model_kwargs["body"])
            del model_kwargs["body"]
        else:
            body_str = model_kwargs["body"]
            del model_kwargs["body"]

        response = invoke_model_with_retry(
            client=self._client,
            model=self.model,
            messages=body_str,
            max_retries=self.max_retries,
            **model_kwargs,
        )

        # Parse the streaming response body
        response_body = json.loads(response["body"].read())

        # Parse response based on model
        if self.model.startswith("anthropic.claude"):
            completion = response_body["content"][0]["text"]
        else:
            # Add support for other models as needed
            completion = response_body.get("completion", "")

        return CompletionResponse(
            text=completion,
            raw=response_body,
            additional_kwargs={
                "input_tokens": response["ResponseMetadata"]["HTTPHeaders"].get(
                    "x-amzn-bedrock-input-token-count"
                ),
                "output_tokens": response["ResponseMetadata"]["HTTPHeaders"].get(
                    "x-amzn-bedrock-output-token-count"
                ),
            },
        )

    def complete(
        self, prompt: str, image_documents: Sequence[ImageNode], **kwargs: Any
    ) -> CompletionResponse:
        """Complete the prompt with image support."""
        return self._complete(prompt, image_documents, **kwargs)

    async def acomplete(
        self, prompt: str, image_documents: Sequence[ImageNode], **kwargs: Any
    ) -> CompletionResponse:
        """Complete the prompt with image support asynchronously."""
        message = generate_bedrock_multi_modal_message(
            prompt=prompt,
            image_documents=image_documents,
        )

        # Get model kwargs and prepare the request body
        model_kwargs = self._get_model_kwargs(**kwargs)
        if "body" in model_kwargs:
            model_kwargs["body"]["messages"] = [message]
        else:
            model_kwargs["body"] = {"messages": [message]}

        # Convert body to JSON string
        if isinstance(model_kwargs.get("body"), dict):
            import json

            body_str = json.dumps(model_kwargs["body"])
            del model_kwargs["body"]
        else:
            body_str = model_kwargs["body"]
            del model_kwargs["body"]

        response = await invoke_model_with_retry_async(
            session=self._asession,
            config=self._config,
            model=self.model,
            messages=body_str,
            max_retries=self.max_retries,
            **model_kwargs,
        )

        # Parse the streaming response body
        response_body = json.loads(await response["body"].read())

        # Parse response based on model
        if self.model.startswith("anthropic.claude"):
            completion = response_body["content"][0]["text"]
        else:
            # Add support for other models as needed
            completion = response_body.get("completion", "")

        return CompletionResponse(
            text=completion,
            raw=response_body,
            additional_kwargs={
                "input_tokens": response["ResponseMetadata"]["HTTPHeaders"].get(
                    "x-amzn-bedrock-input-token-count"
                ),
                "output_tokens": response["ResponseMetadata"]["HTTPHeaders"].get(
                    "x-amzn-bedrock-output-token-count"
                ),
            },
        )

    def chat(self, messages: Sequence[Any], **kwargs: Any) -> Any:
        """Chat with the model."""
        raise NotImplementedError("Chat is not supported for this model.")

    def stream_chat(self, messages: Sequence[Any], **kwargs: Any) -> Any:
        """Stream chat with the model."""
        raise NotImplementedError("Stream chat is not supported for this model.")

    async def achat(self, messages: Sequence[Any], **kwargs: Any) -> Any:
        """Chat with the model asynchronously."""
        raise NotImplementedError("Async chat is not supported for this model.")

    async def astream_chat(self, messages: Sequence[Any], **kwargs: Any) -> Any:
        """Stream chat with the model asynchronously."""
        raise NotImplementedError("Async stream chat is not supported for this model.")

    def stream_complete(
        self, prompt: str, image_documents: Sequence[ImageNode], **kwargs: Any
    ) -> Any:
        """Complete the prompt with image support in a streaming fashion."""
        raise NotImplementedError(
            "Streaming completion is not supported for this model."
        )

    async def astream_complete(
        self, prompt: str, image_documents: Sequence[ImageNode], **kwargs: Any
    ) -> Any:
        """Complete the prompt with image support in a streaming fashion asynchronously."""
        raise NotImplementedError(
            "Async streaming completion is not supported for this model."
        )
