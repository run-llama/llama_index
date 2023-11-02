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
    MessageRole,
    llm_chat_callback,
    llm_completion_callback,
)
from llama_index.llms.vertex_utils import (
    CHAT_MODELS,
    CODE_CHAT_MODELS,
    CODE_MODELS,
    TEXT_MODELS,
    _parse_chat_history,
    _parse_examples,
    acompletion_with_retry,
    completion_with_retry,
    init_vertexai,
)


class Vertex(LLM):
    model: str = Field(description="The vertex model to use.")
    temperature: float = Field(description="The temperature to use for sampling.")
    max_tokens: int = Field(description="The maximum number of tokens to generate.")
    examples: Optional[ChatMessage] = Field(
        description="Example messages for the chat model."
    )
    max_retries: int = Field(default=10, description="The maximum number of retries.")

    additional_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Additional kwargs for the Vertex."
    )
    iscode: bool = Field(
        default=False, description="Flag to determine if current model is a Code Model"
    )
    _client: Any = PrivateAttr()
    _chatclient: Any = PrivateAttr()

    def __init__(
        self,
        model: str = "text-bison",
        project: Optional[str] = None,
        location: Optional[str] = None,
        credential: Optional[str] = None,
        examples: Optional[ChatMessage] = None,
        temperature: float = 0.1,
        max_tokens: int = 512,
        max_retries: int = 10,
        iscode: bool = False,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        init_vertexai(project=project, location=location, credentials=credential)

        additional_kwargs = additional_kwargs or {}
        callback_manager = callback_manager or CallbackManager([])

        if model in CHAT_MODELS:
            from vertexai.language_models import ChatModel

            self._chatclient = ChatModel.from_pretrained(model)
        elif model in CODE_CHAT_MODELS:
            from vertexai.language_models import CodeChatModel

            self._chatclient = CodeChatModel.from_pretrained(model)
            iscode = True
        elif model in CODE_MODELS:
            from vertexai.language_models import CodeGenerationModel

            self._client = CodeGenerationModel.from_pretrained(model)
            iscode = True
        elif model in TEXT_MODELS:
            from vertexai.language_models import TextGenerationModel

            self._client = TextGenerationModel.from_pretrained(model)
        else:
            raise (ValueError("Model Not Found Please Check the model name"))

        super().__init__(
            temperature=temperature,
            max_tokens=max_tokens,
            additional_kwargs=additional_kwargs,
            max_retries=max_retries,
            model=model,
            examples=examples,
            iscode=iscode,
            callback_manager=callback_manager,
        )

    @classmethod
    def class_name(cls) -> str:
        return "Vertex"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            is_chat_model=True,
            model_name=self.model,
        )

    @property
    def _model_kwargs(self) -> Dict[str, Any]:
        base_kwargs = {
            "temperature": self.temperature,
            "max_output_tokens": self.max_tokens,
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

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        question = messages[-1].content
        chat_history = _parse_chat_history(messages[:-1])
        chat_params = {**chat_history}

        kwargs = kwargs if kwargs else {}

        params = {**self._model_kwargs, **kwargs}

        if self.iscode and "candidate_count" in params:
            raise (ValueError("candidate_count is not supported by the codey model's"))
        if self.examples and "examples" not in params:
            chat_params["examples"] = _parse_examples(self.examples)
        elif "examples" in params:
            raise (
                ValueError(
                    "examples are not supported in chat generation pass them as a constructor parameter"
                )
            )

        generation = completion_with_retry(
            client=self._chatclient,
            prompt=question,
            chat=True,
            stream=False,
            params=chat_params,
            max_retries=self.max_retries,
            **params
        )

        return ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content=generation.text),
            raw=generation.__dict__,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        kwargs = kwargs if kwargs else {}
        params = {**self._model_kwargs, **kwargs}
        if self.iscode and "candidate_count" in params:
            raise (ValueError("candidate_count is not supported by the codey model's"))

        completion = completion_with_retry(
            self._client, prompt, max_retries=self.max_retries, **params
        )
        return CompletionResponse(text=completion.text, raw=completion.__dict__)

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        question = messages[-1].content
        chat_history = _parse_chat_history(messages[:-1])
        chat_params = {**chat_history}
        kwargs = kwargs if kwargs else {}
        params = {**self._model_kwargs, **kwargs}
        if self.iscode and "candidate_count" in params:
            raise (ValueError("candidate_count is not supported by the codey model's"))
        if self.examples and "examples" not in params:
            chat_params["examples"] = _parse_examples(self.examples)
        elif "examples" in params:
            raise (
                ValueError(
                    "examples are not supported in chat generation pass them as a constructor parameter"
                )
            )

        response = completion_with_retry(
            client=self._chatclient,
            prompt=question,
            chat=True,
            stream=True,
            params=chat_params,
            max_retries=self.max_retries,
            **params
        )

        def gen() -> ChatResponseGen:
            content = ""
            role = MessageRole.ASSISTANT
            for r in response:
                if "text" in r.__dict__:
                    content_delta = r.text
                else:
                    content_delta = ""
                content += content_delta
                yield ChatResponse(
                    message=ChatMessage(role=role, content=content),
                    delta=content_delta,
                    raw=r.__dict__,
                )

        return gen()

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        kwargs = kwargs if kwargs else {}
        params = {**self._model_kwargs, **kwargs}
        if "candidate_count" in params:
            raise (ValueError("candidate_count is not supported by the streaming"))

        completion = completion_with_retry(
            client=self._client,
            prompt=prompt,
            stream=True,
            max_retries=self.max_retries,
            **params
        )

        def gen() -> CompletionResponseGen:
            content = ""
            for r in completion:
                content_delta = r.text
                content += content_delta
                yield CompletionResponse(
                    text=content, delta=content_delta, raw=r.__dict__
                )

        return gen()

    @llm_chat_callback()
    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        question = messages[-1].content
        chat_history = _parse_chat_history(messages[:-1])
        chat_params = {**chat_history}
        kwargs = kwargs if kwargs else {}
        params = {**self._model_kwargs, **kwargs}
        if self.iscode and "candidate_count" in params:
            raise (ValueError("candidate_count is not supported by the codey model's"))
        if self.examples and "examples" not in params:
            chat_params["examples"] = _parse_examples(self.examples)
        elif "examples" in params:
            raise (
                ValueError(
                    "examples are not supported in chat generation pass them as a constructor parameter"
                )
            )
        generation = await acompletion_with_retry(
            client=self._chatclient,
            prompt=question,
            chat=True,
            params=chat_params,
            max_retries=self.max_retries,
            **params
        )
        ##this is due to a bug in vertex AI we have to await twice
        if self.iscode:
            generation = await generation
        return ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content=generation.text),
            raw=generation.__dict__,
        )

    @llm_completion_callback()
    async def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        kwargs = kwargs if kwargs else {}
        params = {**self._model_kwargs, **kwargs}
        if self.iscode and "candidate_count" in params:
            raise (ValueError("candidate_count is not supported by the codey model's"))
        completion = await acompletion_with_retry(
            client=self._client, prompt=prompt, max_retries=self.max_retries, **params
        )
        return CompletionResponse(text=completion.text)

    @llm_chat_callback()
    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        raise (ValueError("Not Implemented"))

    @llm_completion_callback()
    async def astream_complete(
        self, prompt: str, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        raise (ValueError("Not Implemented"))
