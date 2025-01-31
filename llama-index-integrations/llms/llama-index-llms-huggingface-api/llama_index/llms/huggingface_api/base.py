import logging
from typing import Any, Callable, Dict, Optional, Sequence, Union

from huggingface_hub import AsyncInferenceClient, InferenceClient, model_info
from huggingface_hub.hf_api import ModelInfo
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
from llama_index.core.constants import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_NUM_OUTPUTS,
)
from llama_index.core.llms.custom import CustomLLM


logger = logging.getLogger(__name__)


class HuggingFaceInferenceAPI(CustomLLM):
    """
    Wrapper on the Hugging Face's Inference API.

    Overview of the design:
    - Synchronous uses InferenceClient, asynchronous uses AsyncInferenceClient
    - chat uses the conversational task: https://huggingface.co/tasks/conversational
    - complete uses the text generation task: https://huggingface.co/tasks/text-generation

    Note: some models that support the text generation task can leverage Hugging
    Face's optimized deployment toolkit called text-generation-inference (TGI).
    Use InferenceClient.get_model_status to check if TGI is being used.

    Relevant links:
    - General Docs: https://huggingface.co/docs/api-inference/index
    - API Docs: https://huggingface.co/docs/huggingface_hub/main/en/package_reference/inference_client
    - Source: https://github.com/huggingface/huggingface_hub/tree/main/src/huggingface_hub/inference
    """

    @classmethod
    def class_name(cls) -> str:
        return "HuggingFaceInferenceAPI"

    # Corresponds with huggingface_hub.InferenceClient
    model_name: Optional[str] = Field(
        default=None,
        description=(
            "The model to run inference with. Can be a model id hosted on the Hugging"
            " Face Hub, e.g. bigcode/starcoder or a URL to a deployed Inference"
            " Endpoint. Defaults to None, in which case a recommended model is"
            " automatically selected for the task (see Field below)."
        ),
    )
    token: Union[str, bool, None] = Field(
        default=None,
        description=(
            "Hugging Face token. Will default to the locally saved token. Pass "
            "token=False if you don’t want to send your token to the server."
        ),
    )
    timeout: Optional[float] = Field(
        default=None,
        description=(
            "The maximum number of seconds to wait for a response from the server."
            " Loading a new model in Inference API can take up to several minutes."
            " Defaults to None, meaning it will loop until the server is available."
        ),
    )
    headers: Dict[str, str] = Field(
        default=None,
        description=(
            "Additional headers to send to the server. By default only the"
            " authorization and user-agent headers are sent. Values in this dictionary"
            " will override the default values."
        ),
    )
    cookies: Dict[str, str] = Field(
        default=None, description="Additional cookies to send to the server."
    )
    task: Optional[str] = Field(
        default=None,
        description=(
            "Optional task to pick Hugging Face's recommended model, used when"
            " model_name is left as default of None."
        ),
    )

    _sync_client: "InferenceClient" = PrivateAttr()
    _async_client: "AsyncInferenceClient" = PrivateAttr()
    _get_model_info: "Callable[..., ModelInfo]" = PrivateAttr()

    context_window: int = Field(
        default=DEFAULT_CONTEXT_WINDOW,
        description=(
            LLMMetadata.model_fields["context_window"].description
            + " This may be looked up in a model's `config.json`."
        ),
    )
    num_output: int = Field(
        default=DEFAULT_NUM_OUTPUTS,
        description=LLMMetadata.model_fields["num_output"].description,
    )
    is_chat_model: bool = Field(
        default=False,
        description=(
            LLMMetadata.model_fields["is_chat_model"].description
            + " Unless chat templating is intentionally applied, Hugging Face models"
            " are not chat models."
        ),
    )
    is_function_calling_model: bool = Field(
        default=False,
        description=(
            LLMMetadata.model_fields["is_function_calling_model"].description
            + " As of 10/17/2023, Hugging Face doesn't support function calling"
            " messages."
        ),
    )

    def _get_inference_client_kwargs(self) -> Dict[str, Any]:
        """Extract the Hugging Face InferenceClient construction parameters."""
        return {
            "model": self.model_name,
            "token": self.token,
            "timeout": self.timeout,
            "headers": self.headers,
            "cookies": self.cookies,
        }

    def __init__(self, **kwargs: Any) -> None:
        """Initialize.

        Args:
            kwargs: See the class-level Fields.
        """
        if kwargs.get("model_name") is None:
            task = kwargs.get("task", "")
            # NOTE: task being None or empty string leads to ValueError,
            # which ensures model is present
            kwargs["model_name"] = InferenceClient.get_recommended_model(task=task)
            logger.debug(
                f"Using Hugging Face's recommended model {kwargs['model_name']}"
                f" given task {task}."
            )
        if kwargs.get("task") is None:
            task = "conversational"
        else:
            task = kwargs["task"].lower()

        super().__init__(**kwargs)  # Populate pydantic Fields
        self._sync_client = InferenceClient(**self._get_inference_client_kwargs())
        self._async_client = AsyncInferenceClient(**self._get_inference_client_kwargs())
        self._get_model_info = model_info

    def validate_supported(self, task: str) -> None:
        """
        Confirm the contained model_name is deployed on the Inference API service.

        Args:
            task: Hugging Face task to check within. A list of all tasks can be
                found here: https://huggingface.co/tasks
        """
        all_models = self._sync_client.list_deployed_models(frameworks="all")
        try:
            if self.model_name not in all_models[task]:
                raise ValueError(
                    "The Inference API service doesn't have the model"
                    f" {self.model_name!r} deployed."
                )
        except KeyError as exc:
            raise KeyError(
                f"Input task {task!r} not in possible tasks {list(all_models.keys())}."
            ) from exc

    def get_model_info(self, **kwargs: Any) -> "ModelInfo":
        """Get metadata on the current model from Hugging Face."""
        return self._get_model_info(self.model_name, **kwargs)

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            is_chat_model=self.is_chat_model,
            is_function_calling_model=self.is_function_calling_model,
            model_name=self.model_name,
        )

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        # default to conversational task as that was the previous functionality
        if self.task == "conversational" or self.task is None:
            output = self._sync_client.chat_completion(
                messages=[
                    {"role": m.role.value, "content": m.content} for m in messages
                ],
                model=self.model_name,
                **kwargs,
            )
            return ChatResponse(
                message=ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=output["choices"][0]["message"]["content"] or "",
                )
            )
        else:
            # try and use text generation
            prompt = self.messages_to_prompt(messages)
            completion = self.complete(prompt)
            return ChatResponse(
                message=ChatMessage(role=MessageRole.ASSISTANT, content=completion.text)
            )

    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        return CompletionResponse(
            text=self._sync_client.text_generation(
                prompt, **{**{"max_new_tokens": self.num_output}, **kwargs}
            )
        )

    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        raise NotImplementedError

    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        raise NotImplementedError

    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        raise NotImplementedError

    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        response = await self._async_client.text_generation(
            prompt, **{**{"max_new_tokens": self.num_output}, **kwargs}
        )
        return CompletionResponse(text=response)

    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        # default to conversational task as that was the previous functionality
        if self.task == "conversational" or self.task is None:
            output = await self._async_client.chat_completion(
                messages=[
                    {"role": m.role.value, "content": m.content} for m in messages
                ],
                model=self.model_name,
                **kwargs,
            )
            return ChatResponse(
                message=ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=output["choices"][0]["message"]["content"] or "",
                )
            )
        else:
            # try and use text generation
            prompt = self.messages_to_prompt(messages)
            completion = await self.acomplete(prompt)
            return ChatResponse(
                message=ChatMessage(role=MessageRole.ASSISTANT, content=completion.text)
            )

    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        raise NotImplementedError
