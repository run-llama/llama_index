import asyncio
import time
from typing import Any, Dict, List, Tuple, Optional, Sequence, Union
from zhipuai import ZhipuAI as ZhipuAIClient
from llama_index.core.base.llms.generic_utils import (
    messages_to_prompt,
    chat_to_completion_decorator,
    achat_to_completion_decorator,
    stream_chat_to_completion_decorator,
    astream_chat_to_completion_decorator,
)
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    ChatResponseAsyncGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    MessageRole,
)
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.constants import DEFAULT_NUM_OUTPUTS
from llama_index.core.multi_modal_llms import (
    MultiModalLLM,
    MultiModalLLMMetadata,
)
from llama_index.core.schema import ImageNode


DEFAULT_REQUEST_TIMEOUT = 30.0
SUCCESS = "SUCCESS"
FAILED = "FAILED"
GLM_MULTI_MODAL_MODELS = {
    "glm-4v-plus": 8_000,
    "glm-4v": 2_000,
    "cogvideox": 500,
    "cogview-3-plus": 1_000,
    "cogview-3": 1_000,
}


def glm_model_to_context_size(model: str) -> Union[int, None]:
    token_limit = GLM_MULTI_MODAL_MODELS.get(model, None)

    if token_limit is None:
        raise ValueError(
            f"Model name {model} not found in {GLM_MULTI_MODAL_MODELS.keys()}"
        )

    return token_limit


def get_additional_kwargs(
    response: Dict[str, Any], exclude: Tuple[str, ...]
) -> Dict[str, Any]:
    return {k: v for k, v in response.items() if k not in exclude}


def async_llm_generate(item):
    try:
        return next(item)
    except StopIteration:
        return None


class ZhipuAIMultiModal(MultiModalLLM):
    """ZhipuAI MultiModal.

    Visit https://open.bigmodel.cn to get more information about ZhipuAI.

    Examples:
        `pip install llama-index-multi-modal-llms-zhipuai`

        ```python
        from llama_index.multi_modal_llms.zhipuai import ZhipuAIMultiModal

        llm = ZhipuAIMultiModal(model="cogview-3", api_key="YOUR API KEY")

        response = llm.complete("draw a bird flying in the sky")
        print(response)
        ```
    """

    model: str = Field(description="The ZhipuAI model to use.")
    api_key: Optional[str] = Field(
        default=None,
        description="The API key to use for the ZhipuAI API.",
    )
    temperature: float = Field(
        default=0.95,
        description="The temperature to use for sampling.",
        ge=0.0,
        le=1.0,
    )
    max_tokens: int = Field(
        default=1024,
        description="The maximum number of tokens for model output.",
        gt=0,
        le=4096,
    )
    timeout: float = Field(
        default=DEFAULT_REQUEST_TIMEOUT,
        description="The timeout for making http request to ZhipuAI API server",
    )
    size: Optional[str] = Field(
        default="1024x1024",
        json_schema_extra={
            "enum": [
                "1024x1024",
                "768x1344",
                "864x1152",
                "1344x768",
                "1152x864",
                "1440x720",
                "720x1440",
            ]
        },
        description="The size of the image to generate.",
    )
    additional_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Additional kwargs for the ZhipuAI API."
    )
    _client: Optional[ZhipuAIClient] = PrivateAttr()

    def __init__(
        self,
        model: str,
        api_key: str,
        temperature: float = 0.95,
        max_tokens: int = 1024,
        timeout: float = DEFAULT_REQUEST_TIMEOUT,
        size: str = "1024x1024",
        additional_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        additional_kwargs = additional_kwargs or {}
        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout or DEFAULT_REQUEST_TIMEOUT,
            size=size,
            additional_kwargs=additional_kwargs,
            **kwargs,
        )

        self._client = ZhipuAIClient(api_key=api_key)

    @classmethod
    def class_name(cls) -> str:
        return "ZhipuAIMultiModal"

    @property
    def metadata(self) -> MultiModalLLMMetadata:
        """MultiModalLLM metadata."""
        return MultiModalLLMMetadata(
            context_window=glm_model_to_context_size(self.model),
            num_output=DEFAULT_NUM_OUTPUTS,
            model_name=self.model,
        )

    @property
    def model_kwargs(self) -> Dict[str, Any]:
        base_kwargs = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        return {
            **base_kwargs,
            **self.additional_kwargs,
        }

    def _convert_to_llm_messages(
        self, messages: Sequence[ChatMessage], **kwargs
    ) -> List[Dict]:
        additional_args = [
            {"type": key, key: {"url": kwargs[key]}}
            for key in ["image_url", "video_url"]
            if key in kwargs
        ]

        def build_message(message: ChatMessage) -> Dict:
            if isinstance(message.content, list):
                content = message.content
            else:
                content = [{"type": "text", "text": message.content}]
            if message.role.value == MessageRole.USER:
                content.extend(additional_args)
            return {"role": message.role.value, "content": content}

        return [build_message(msg) for msg in messages]

    def has_completions_api(self) -> bool:
        return "glm" in self.model

    def has_videos_generations_api(self) -> bool:
        return "cogvideo" in self.model

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        if self.has_completions_api():
            messages_dict = self._convert_to_llm_messages(messages, **kwargs)
            raw_response = self._client.chat.completions.create(
                model=self.model,
                messages=messages_dict,
                stream=False,
                timeout=self.timeout,
                extra_body=self.model_kwargs,
            )
            response = ChatResponse(
                message=ChatMessage(
                    content=raw_response.choices[0].message.content,
                    role=raw_response.choices[0].message.role,
                    additional_kwargs=get_additional_kwargs(
                        raw_response.choices[0].message.model_dump(),
                        ("content", "role"),
                    ),
                ),
                raw=raw_response,
            )
        elif self.has_videos_generations_api():
            raw_response = self._client.videos.generations(
                model=self.model,
                prompt=messages_to_prompt(messages),
                image_url=kwargs.get("image_url", None),
            )
            task_id = raw_response.id
            task_status = raw_response.task_status
            get_count = 0
            while task_status not in [SUCCESS, FAILED] and get_count < self.timeout:
                raw_response = self._client.videos.retrieve_videos_result(id=task_id)
                task_status = raw_response.task_status
                get_count += 1
                time.sleep(1)
            response = ChatResponse(
                message=ChatMessage(
                    content=raw_response.video_result[0].url,
                    role=MessageRole.ASSISTANT,
                    additional_kwargs={},
                ),
                raw=raw_response,
            )
        else:
            raw_response = self._client.images.generations(
                model=self.model, prompt=messages_to_prompt(messages), size=self.size
            )
            response = ChatResponse(
                message=ChatMessage(
                    content=raw_response.data[0].url,
                    role=MessageRole.ASSISTANT,
                    additional_kwargs={},
                ),
                raw=raw_response,
            )
        return response

    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        if not self.has_completions_api():
            raise NotImplementedError("Stream api for cog is not implemented")

        messages_dict = self._convert_to_llm_messages(messages, **kwargs)

        def gen() -> ChatResponseGen:
            raw_response = self._client.chat.completions.create(
                model=self.model,
                messages=messages_dict,
                stream=True,
                timeout=self.timeout,
                extra_body=self.model_kwargs,
            )
            response_txt = ""
            for chunk in raw_response:
                if chunk.choices[0].delta.content is None:
                    continue
                response_txt += chunk.choices[0].delta.content
                yield ChatResponse(
                    message=ChatMessage(
                        content=response_txt,
                        role=chunk.choices[0].delta.role,
                        additional_kwargs=get_additional_kwargs(
                            chunk.choices[0].delta.model_dump(), ("content", "role")
                        ),
                    ),
                    delta=chunk.choices[0].delta.content,
                    raw=chunk,
                )

        return gen()

    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        if not self.has_completions_api():
            raise NotImplementedError("Async stream api for cog is not implemented")

        messages_dict = self._convert_to_llm_messages(messages, **kwargs)

        async def gen() -> ChatResponseAsyncGen:
            # TODO async interfaces don't support streaming
            # needs to find a more suitable implementation method
            raw_response = self._client.chat.completions.create(
                model=self.model,
                messages=messages_dict,
                stream=True,
                timeout=self.timeout,
                extra_body=self.model_kwargs,
            )
            response_txt = ""
            while True:
                chunk = await asyncio.to_thread(async_llm_generate, raw_response)
                if not chunk:
                    break
                if chunk.choices[0].delta.content is None:
                    continue
                response_txt += chunk.choices[0].delta.content
                yield ChatResponse(
                    message=ChatMessage(
                        content=response_txt,
                        role=chunk.choices[0].delta.role,
                        additional_kwargs=get_additional_kwargs(
                            chunk.choices[0].delta.model_dump(), ("content", "role")
                        ),
                    ),
                    delta=chunk.choices[0].delta.content,
                    raw=chunk,
                )

        return gen()

    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        if self.has_completions_api():
            messages_dict = self._convert_to_llm_messages(messages, **kwargs)
            raw_response = self._client.chat.asyncCompletions.create(
                model=self.model,
                messages=messages_dict,
                timeout=self.timeout,
                extra_body=self.model_kwargs,
            )
            task_id = raw_response.id
            task_status = raw_response.task_status
            get_count = 0
            while task_status not in [SUCCESS, FAILED] and get_count < self.timeout:
                task_result = (
                    self._client.chat.asyncCompletions.retrieve_completion_result(
                        task_id
                    )
                )
                raw_response = task_result
                task_status = raw_response.task_status
                get_count += 1
                await asyncio.sleep(1)
            response = ChatResponse(
                message=ChatMessage(
                    content=raw_response.choices[0].message.content,
                    role=raw_response.choices[0].message.role,
                    additional_kwargs=get_additional_kwargs(
                        raw_response.choices[0].message.model_dump(),
                        ("content", "role"),
                    ),
                ),
                raw=raw_response,
            )
        else:
            response = await asyncio.to_thread(self.chat, messages=messages, **kwargs)
        return response

    def complete(
        self, prompt: str, image_documents: Sequence[ImageNode] = None, **kwargs: Any
    ) -> CompletionResponse:
        return chat_to_completion_decorator(self.chat)(prompt, **kwargs)

    async def acomplete(
        self, prompt: str, image_documents: Sequence[ImageNode] = None, **kwargs: Any
    ) -> CompletionResponse:
        return await achat_to_completion_decorator(self.achat)(prompt, **kwargs)

    def stream_complete(
        self, prompt: str, image_documents: Sequence[ImageNode] = None, **kwargs: Any
    ) -> CompletionResponseGen:
        if not self.has_completions_api():
            raise NotImplementedError("Stream api for cog is not implemented")
        return stream_chat_to_completion_decorator(self.stream_chat)(prompt, **kwargs)

    async def astream_complete(
        self, prompt: str, image_documents: Sequence[ImageNode] = None, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        if not self.has_completions_api():
            raise NotImplementedError("Async Stream api for cog is not implemented")
        return await astream_chat_to_completion_decorator(self.astream_chat)(
            prompt, **kwargs
        )
