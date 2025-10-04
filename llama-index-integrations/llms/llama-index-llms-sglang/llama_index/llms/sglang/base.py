import json
from typing import Any, Callable, Dict, List, Optional, Sequence

from llama_index.core.base.llms.generic_utils import (
    completion_response_to_chat_response,
)
from llama_index.core.base.llms.generic_utils import (
    messages_to_prompt as generic_messages_to_prompt,
)
from llama_index.core.base.llms.generic_utils import (
    stream_completion_response_to_chat_response,
)
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
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from llama_index.core.llms.llm import LLM
from llama_index.core.types import BaseOutputParser, PydanticProgramMode

from llama_index.llms.sglang.utils import get_response, post_http_request


class SGLang(LLM):
    r"""
    SGLang LLM.

    This class connects to an SGLang server for high-performance LLM inference.

    Examples:
        `pip install llama-index-llms-sglang`

        ```python
        from llama_index.llms.sglang import SGLang

        # specific functions to format for mistral instruct
        def messages_to_prompt(messages):
            prompt = "\n".join([str(x) for x in messages])
            return f"<s>[INST] {prompt} [/INST] </s>\n"

        def completion_to_prompt(completion):
            return f"<s>[INST] {completion} [/INST] </s>\n"

        llm = SGLang(
            model="mistralai/Mistral-7B-Instruct-v0.1",
            api_url="http://localhost:30000",
            temperature=0.7,
            max_new_tokens=256,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
        )

        response = llm.complete("What is a black hole?")
        print(response)
        ```

    """

    model: Optional[str] = Field(
        default="default",
        description="The model name (for metadata purposes).",
    )

    api_url: str = Field(
        default="http://localhost:30000",
        description="The API URL for the SGLang server.",
    )

    api_key: Optional[str] = Field(
        default=None,
        description="API key for authentication (if required by server).",
    )

    temperature: float = Field(
        default=1.0,
        description="The temperature to use for sampling.",
    )

    max_new_tokens: int = Field(
        default=512,
        description="Maximum number of tokens to generate per output sequence.",
    )

    top_p: float = Field(
        default=1.0,
        description="Float that controls the cumulative probability of the top tokens to consider.",
    )

    top_k: int = Field(
        default=-1,
        description="Integer that controls the number of top tokens to consider.",
    )

    frequency_penalty: float = Field(
        default=0.0,
        description="Float that penalizes new tokens based on their frequency in the generated text so far.",
    )

    presence_penalty: float = Field(
        default=0.0,
        description="Float that penalizes new tokens based on whether they appear in the generated text so far.",
    )

    stop: Optional[List[str]] = Field(
        default=None,
        description="List of strings that stop the generation when they are generated.",
    )

    n: int = Field(
        default=1,
        description="Number of output sequences to return for the given prompt.",
    )

    skip_special_tokens: bool = Field(
        default=True,
        description="Whether to skip special tokens in the output.",
    )

    regex: Optional[str] = Field(
        default=None,
        description="Optional regex pattern for constrained generation.",
    )

    is_chat_model: bool = Field(
        default=False,
        description=LLMMetadata.model_fields["is_chat_model"].description,
    )

    additional_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional keyword arguments for SGLang API.",
    )

    _client: Any = PrivateAttr()

    def __init__(
        self,
        model: str = "default",
        api_url: str = "http://localhost:30000",
        api_key: Optional[str] = None,
        temperature: float = 1.0,
        max_new_tokens: int = 512,
        top_p: float = 1.0,
        top_k: int = -1,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: Optional[List[str]] = None,
        n: int = 1,
        skip_special_tokens: bool = True,
        regex: Optional[str] = None,
        additional_kwargs: Dict[str, Any] = {},
        callback_manager: Optional[CallbackManager] = None,
        system_prompt: Optional[str] = None,
        messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]] = None,
        completion_to_prompt: Optional[Callable[[str], str]] = None,
        pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
        output_parser: Optional[BaseOutputParser] = None,
        is_chat_model: Optional[bool] = False,
    ) -> None:
        messages_to_prompt = messages_to_prompt or generic_messages_to_prompt
        completion_to_prompt = completion_to_prompt or (lambda x: x)
        callback_manager = callback_manager or CallbackManager([])

        super().__init__(
            model=model,
            api_url=api_url,
            api_key=api_key,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
            n=n,
            skip_special_tokens=skip_special_tokens,
            regex=regex,
            additional_kwargs=additional_kwargs,
            callback_manager=callback_manager,
            system_prompt=system_prompt,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            pydantic_program_mode=pydantic_program_mode,
            output_parser=output_parser,
            is_chat_model=is_chat_model,
        )
        self._client = None

    @classmethod
    def class_name(cls) -> str:
        return "SGLang"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            model_name=self.model,
            is_chat_model=self.is_chat_model,
        )

    @property
    def _model_kwargs(self) -> Dict[str, Any]:
        base_kwargs = {
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "stop": self.stop,
            "n": self.n,
            "skip_special_tokens": self.skip_special_tokens,
        }
        if self.regex:
            base_kwargs["regex"] = self.regex
        return {**base_kwargs, **self.additional_kwargs}

    def _get_all_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
        return {
            **self._model_kwargs,
            **kwargs,
        }

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        kwargs = kwargs if kwargs else {}
        prompt = self.messages_to_prompt(messages)
        completion_response = self.complete(prompt, **kwargs)
        return completion_response_to_chat_response(completion_response)

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        kwargs = kwargs if kwargs else {}
        params = {**self._model_kwargs, **kwargs}

        # Build sampling parameters for SGLang
        sampling_params = dict(**params)
        # SGLang OpenAI-compatible API uses 'prompt' parameter
        sampling_params["prompt"] = prompt
        sampling_params["model"] = self.model

        # Use OpenAI-compatible endpoint
        endpoint = f"{self.api_url}/v1/completions"
        response = post_http_request(
            endpoint, sampling_params, stream=False, api_key=self.api_key
        )
        output = get_response(response)

        return CompletionResponse(text=output[0])

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        prompt = self.messages_to_prompt(messages)
        completion_response = self.stream_complete(prompt, **kwargs)
        return stream_completion_response_to_chat_response(completion_response)

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        kwargs = kwargs if kwargs else {}
        params = {**self._model_kwargs, **kwargs}

        sampling_params = dict(**params)
        sampling_params["text"] = prompt

        # SGLang uses OpenAI-compatible API, so use /v1/completions for streaming
        endpoint = f"{self.api_url}/v1/completions"
        response = post_http_request(
            endpoint, sampling_params, stream=True, api_key=self.api_key
        )

        def gen() -> CompletionResponseGen:
            response_str = ""
            for chunk in response.iter_lines(
                chunk_size=8192, decode_unicode=False, delimiter=b"\n"
            ):
                if chunk:
                    chunk_str = chunk.decode("utf-8")
                    # Handle SSE format
                    if chunk_str.startswith("data: "):
                        chunk_str = chunk_str[6:]

                    if chunk_str.strip() == "[DONE]":
                        break

                    try:
                        data = json.loads(chunk_str)
                        # OpenAI format has choices array
                        if "choices" in data and len(data["choices"]) > 0:
                            delta = data["choices"][0].get("text", "")
                            response_str += delta
                            yield CompletionResponse(text=response_str, delta=delta)
                    except json.JSONDecodeError:
                        continue

        return gen()

    @llm_chat_callback()
    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        kwargs = kwargs if kwargs else {}
        return self.chat(messages, **kwargs)

    @llm_completion_callback()
    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        kwargs = kwargs if kwargs else {}
        return self.complete(prompt, **kwargs)

    @llm_chat_callback()
    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        async def gen() -> ChatResponseAsyncGen:
            for message in self.stream_chat(messages, **kwargs):
                yield message

        return gen()

    @llm_completion_callback()
    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        async def gen() -> CompletionResponseAsyncGen:
            for message in self.stream_complete(prompt, **kwargs):
                yield message

        return gen()
