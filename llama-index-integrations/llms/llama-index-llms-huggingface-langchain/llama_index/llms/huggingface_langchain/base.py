"""LlamaIndex LLM integration using langchain-huggingface.

Provides a LlamaIndex-compatible LLM that wraps langchain-huggingface's
HuggingFaceEndpoint and ChatHuggingFace classes for seamless access to
HuggingFace models via the Inference API or local pipelines.
"""

import logging
from typing import Any, Callable, Generator, Optional, Sequence

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
from llama_index.core.constants import DEFAULT_CONTEXT_WINDOW, DEFAULT_NUM_OUTPUTS
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from llama_index.core.llms.llm import LLM
from llama_index.core.types import BaseOutputParser, PydanticProgramMode

logger = logging.getLogger(__name__)


def _to_langchain_messages(
    messages: Sequence[ChatMessage],
) -> list:
    """Convert LlamaIndex ChatMessages to LangChain message objects."""
    from langchain_core.messages import (
        AIMessage,
        HumanMessage,
        SystemMessage,
    )

    lc_messages = []
    for msg in messages:
        content = msg.content or ""
        if msg.role == MessageRole.SYSTEM:
            lc_messages.append(SystemMessage(content=content))
        elif msg.role == MessageRole.ASSISTANT:
            lc_messages.append(AIMessage(content=content))
        else:
            lc_messages.append(HumanMessage(content=content))
    return lc_messages


def _from_langchain_message(lc_message: Any) -> ChatMessage:
    """Convert a LangChain message to a LlamaIndex ChatMessage."""
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    content = lc_message.content if hasattr(lc_message, "content") else str(lc_message)

    if isinstance(lc_message, SystemMessage):
        role = MessageRole.SYSTEM
    elif isinstance(lc_message, AIMessage):
        role = MessageRole.ASSISTANT
    elif isinstance(lc_message, HumanMessage):
        role = MessageRole.USER
    else:
        role = MessageRole.ASSISTANT

    return ChatMessage(role=role, content=content)


class HuggingFaceLangChainLLM(LLM):
    """LlamaIndex LLM powered by langchain-huggingface.

    Uses HuggingFaceEndpoint (remote API) or HuggingFacePipeline (local)
    for text generation, optionally wrapped with ChatHuggingFace for
    proper chat template handling.

    Examples:
        `pip install llama-index-llms-huggingface-langchain`

        ```python
        from llama_index.llms.huggingface_langchain import HuggingFaceLangChainLLM

        # Remote via HuggingFace Inference API
        llm = HuggingFaceLangChainLLM(
            repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
            max_new_tokens=512,
        )

        response = llm.complete("Explain quantum computing in simple terms.")
        print(response.text)

        # Chat mode
        from llama_index.core.base.llms.types import ChatMessage, MessageRole
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content="You are helpful."),
            ChatMessage(role=MessageRole.USER, content="What is Python?"),
        ]
        chat_response = llm.chat(messages)
        print(chat_response.message.content)

        # Local execution
        llm_local = HuggingFaceLangChainLLM(
            repo_id="google/flan-t5-small",
            backend="local",
            task="text2text-generation",
        )
        ```
    """

    repo_id: str = Field(
        description="HuggingFace model repository ID (e.g. 'meta-llama/Meta-Llama-3-8B-Instruct').",
    )
    task: str = Field(
        default="text-generation",
        description="Model task type: 'text-generation', 'text2text-generation', 'summarization', etc.",
    )
    backend: str = Field(
        default="api",
        description="'api' for HuggingFace Inference API (remote), 'local' for local pipeline via transformers.",
    )
    max_new_tokens: int = Field(
        default=DEFAULT_NUM_OUTPUTS,
        description="Maximum number of tokens to generate.",
    )
    temperature: float = Field(
        default=0.1,
        description="Sampling temperature for generation.",
    )
    do_sample: bool = Field(
        default=False,
        description="Whether to use sampling (True) or greedy decoding (False).",
    )
    context_window: int = Field(
        default=DEFAULT_CONTEXT_WINDOW,
        description="Maximum context window size.",
    )
    is_chat_model: bool = Field(
        default=True,
        description=(
            "Whether to wrap the model with ChatHuggingFace for chat template support. "
            "Set to True for instruction-tuned models."
        ),
    )
    huggingfacehub_api_token: Optional[str] = Field(
        default=None,
        description="HuggingFace API token. Falls back to HF_TOKEN or HUGGINGFACE_TOKEN env vars.",
    )
    model_kwargs: dict = Field(
        default_factory=dict,
        description="Additional keyword arguments passed to the underlying model constructor.",
    )

    _llm: Any = PrivateAttr()
    _chat_model: Any = PrivateAttr(default=None)

    def __init__(
        self,
        repo_id: str,
        task: str = "text-generation",
        backend: str = "api",
        max_new_tokens: int = DEFAULT_NUM_OUTPUTS,
        temperature: float = 0.1,
        do_sample: bool = False,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        is_chat_model: bool = True,
        huggingfacehub_api_token: Optional[str] = None,
        model_kwargs: Optional[dict] = None,
        callback_manager: Optional[CallbackManager] = None,
        system_prompt: Optional[str] = None,
        messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]] = None,
        completion_to_prompt: Optional[Callable[[str], str]] = None,
        pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
        output_parser: Optional[BaseOutputParser] = None,
    ) -> None:
        model_kwargs = model_kwargs or {}

        super().__init__(
            repo_id=repo_id,
            task=task,
            backend=backend,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            context_window=context_window,
            is_chat_model=is_chat_model,
            huggingfacehub_api_token=huggingfacehub_api_token,
            model_kwargs=model_kwargs,
            callback_manager=callback_manager,
            system_prompt=system_prompt,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            pydantic_program_mode=pydantic_program_mode,
            output_parser=output_parser,
        )

        # Resolve API token
        token = huggingfacehub_api_token
        if token is None:
            import os

            token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")

        if backend == "api":
            from langchain_huggingface import HuggingFaceEndpoint

            self._llm = HuggingFaceEndpoint(
                repo_id=repo_id,
                task=task,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                huggingfacehub_api_token=token,
                **model_kwargs,
            )
        elif backend == "local":
            from langchain_huggingface import HuggingFacePipeline

            pipeline_kwargs = {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "do_sample": do_sample,
                **model_kwargs,
            }
            self._llm = HuggingFacePipeline.from_model_id(
                model_id=repo_id,
                task=task,
                pipeline_kwargs=pipeline_kwargs,
            )
        else:
            raise ValueError(
                f"Unknown backend '{backend}'. Supported values: 'api', 'local'."
            )

        if is_chat_model:
            from langchain_huggingface import ChatHuggingFace

            self._chat_model = ChatHuggingFace(llm=self._llm)
        else:
            self._chat_model = None

    @classmethod
    def class_name(cls) -> str:
        return "HuggingFaceLangChainLLM"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.max_new_tokens,
            is_chat_model=self.is_chat_model,
            model_name=self.repo_id,
        )

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        if self._chat_model is not None:
            lc_messages = _to_langchain_messages(messages)
            lc_response = self._chat_model.invoke(lc_messages, **kwargs)
            message = _from_langchain_message(lc_response)
            return ChatResponse(message=message)

        # Fallback: convert messages to prompt and use completion
        prompt = self.messages_to_prompt(messages)
        completion = self.complete(prompt, formatted=True, **kwargs)
        return ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT, content=completion.text
            )
        )

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        if not formatted:
            prompt = self.completion_to_prompt(prompt)

        result = self._llm.invoke(prompt, **kwargs)

        from langchain_core.messages import AIMessage

        if isinstance(result, AIMessage):
            text = result.content
        else:
            text = str(result)

        return CompletionResponse(text=text)

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        model = self._chat_model if self._chat_model is not None else self._llm

        if hasattr(model, "stream"):
            lc_messages = _to_langchain_messages(messages)

            def gen() -> Generator[ChatResponse, None, None]:
                response_str = ""
                for chunk in model.stream(lc_messages, **kwargs):
                    if hasattr(chunk, "content"):
                        delta = chunk.content or ""
                    else:
                        delta = str(chunk)
                    response_str += delta
                    yield ChatResponse(
                        message=ChatMessage(
                            role=MessageRole.ASSISTANT, content=response_str
                        ),
                        delta=delta,
                    )

            return gen()

        # Fallback: non-streaming
        response = self.chat(messages, **kwargs)

        def single() -> Generator[ChatResponse, None, None]:
            yield ChatResponse(
                message=response.message,
                delta=response.message.content or "",
            )

        return single()

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        if not formatted:
            prompt = self.completion_to_prompt(prompt)

        if hasattr(self._llm, "stream"):

            def gen() -> Generator[CompletionResponse, None, None]:
                text = ""
                for chunk in self._llm.stream(prompt, **kwargs):
                    if hasattr(chunk, "content"):
                        delta = chunk.content or ""
                    else:
                        delta = str(chunk)
                    text += delta
                    yield CompletionResponse(delta=delta, text=text)

            return gen()

        # Fallback: non-streaming
        response = self.complete(prompt, formatted=True, **kwargs)

        def single() -> Generator[CompletionResponse, None, None]:
            yield CompletionResponse(delta=response.text, text=response.text)

        return single()

    @llm_chat_callback()
    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        return self.chat(messages, **kwargs)

    @llm_completion_callback()
    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        return self.complete(prompt, formatted=formatted, **kwargs)

    @llm_chat_callback()
    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        async def gen() -> ChatResponseAsyncGen:
            for msg in self.stream_chat(messages, **kwargs):
                yield msg

        return gen()

    @llm_completion_callback()
    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        async def gen() -> CompletionResponseAsyncGen:
            for resp in self.stream_complete(prompt, formatted=formatted, **kwargs):
                yield resp

        return gen()
