import json
from typing import Any, Callable, Dict, List, Optional, Sequence

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
from llama_index.core.base.embeddings.base import (
    DEFAULT_EMBED_BATCH_SIZE,
    BaseEmbedding,
)
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CallbackManager
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from llama_index.core.base.llms.generic_utils import (
    completion_response_to_chat_response,
    stream_completion_response_to_chat_response,
)
from llama_index.core.base.llms.generic_utils import (
    messages_to_prompt as generic_messages_to_prompt,
)
from llama_index.core.constants import DEFAULT_CONTEXT_WINDOW
from llama_index.core.llms.llm import LLM
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle, MetadataMode
from llama_index.core.types import BaseOutputParser, PydanticProgramMode
from llama_index.llms.vllm.utils import get_response, post_http_request
import atexit


class Vllm(LLM):
    r"""
    Vllm LLM.

    This class runs a vLLM model locally.

    Examples:
        `pip install llama-index-llms-vllm`


        ```python
        from llama_index.llms.vllm import Vllm

        # specific functions to format for mistral instruct
        def messages_to_prompt(messages):
            prompt = "\n".join([str(x) for x in messages])
            return f"<s>[INST] {prompt} [/INST] </s>\n"

        def completion_to_prompt(completion):
            return f"<s>[INST] {completion} [/INST] </s>\n"

        llm = Vllm(
            model="mistralai/Mistral-7B-Instruct-v0.1",
            tensor_parallel_size=4,
            max_new_tokens=256,
            vllm_kwargs={"swap_space": 1, "gpu_memory_utilization": 0.5},
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
        )

        llm.complete(
            "What is a black hole?"
        )
        ```

    """

    model: Optional[str] = Field(description="The HuggingFace Model to use.")

    temperature: float = Field(description="The temperature to use for sampling.")

    tensor_parallel_size: Optional[int] = Field(
        default=1,
        description="The number of GPUs to use for distributed execution with tensor parallelism.",
    )

    trust_remote_code: Optional[bool] = Field(
        default=True,
        description="Trust remote code (e.g., from HuggingFace) when downloading the model and tokenizer.",
    )

    n: int = Field(
        default=1,
        description="Number of output sequences to return for the given prompt.",
    )

    best_of: Optional[int] = Field(
        default=None,
        description="Number of output sequences that are generated from the prompt.",
    )

    presence_penalty: float = Field(
        default=0.0,
        description="Float that penalizes new tokens based on whether they appear in the generated text so far.",
    )

    frequency_penalty: float = Field(
        default=0.0,
        description="Float that penalizes new tokens based on their frequency in the generated text so far.",
    )

    top_p: float = Field(
        default=1.0,
        description="Float that controls the cumulative probability of the top tokens to consider.",
    )

    top_k: int = Field(
        default=-1,
        description="Integer that controls the number of top tokens to consider.",
    )

    stop: Optional[List[str]] = Field(
        default=None,
        description="List of strings that stop the generation when they are generated.",
    )

    ignore_eos: bool = Field(
        default=False,
        description="Whether to ignore the EOS token and continue generating tokens after the EOS token is generated.",
    )

    max_new_tokens: int = Field(
        default=512,
        description="Maximum number of tokens to generate per output sequence.",
    )

    logprobs: Optional[int] = Field(
        default=None,
        description="Number of log probabilities to return per output token.",
    )

    dtype: str = Field(
        default="auto",
        description="The data type for the model weights and activations.",
    )

    download_dir: Optional[str] = Field(
        default=None,
        description="Directory to download and load the weights. (Default to the default cache dir of huggingface)",
    )

    vllm_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Holds any model parameters valid for `vllm.LLM` call not explicitly specified.",
    )

    api_url: str = Field(description="The api url for vllm server")

    is_chat_model: bool = Field(
        default=False,
        description=LLMMetadata.model_fields["is_chat_model"].description,
    )

    _client: Any = PrivateAttr()

    def __init__(
        self,
        model: str = "facebook/opt-125m",
        temperature: float = 1.0,
        tensor_parallel_size: int = 1,
        trust_remote_code: bool = False,
        n: int = 1,
        best_of: Optional[int] = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        top_p: float = 1.0,
        top_k: int = -1,
        stop: Optional[List[str]] = None,
        ignore_eos: bool = False,
        max_new_tokens: int = 512,
        logprobs: Optional[int] = None,
        dtype: str = "auto",
        download_dir: Optional[str] = None,
        vllm_kwargs: Dict[str, Any] = {},
        api_url: Optional[str] = "",
        callback_manager: Optional[CallbackManager] = None,
        system_prompt: Optional[str] = None,
        messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]] = None,
        completion_to_prompt: Optional[Callable[[str], str]] = None,
        pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
        output_parser: Optional[BaseOutputParser] = None,
        is_chat_model: Optional[bool] = False,
    ) -> None:
        callback_manager = callback_manager or CallbackManager([])
        super().__init__(
            model=model,
            temperature=temperature,
            n=n,
            best_of=best_of,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            top_p=top_p,
            top_k=top_k,
            stop=stop,
            ignore_eos=ignore_eos,
            max_new_tokens=max_new_tokens,
            logprobs=logprobs,
            dtype=dtype,
            download_dir=download_dir,
            vllm_kwargs=vllm_kwargs,
            api_url=api_url,
            callback_manager=callback_manager,
            system_prompt=system_prompt,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            pydantic_program_mode=pydantic_program_mode,
            output_parser=output_parser,
            is_chat_model=is_chat_model,
        )
        if not api_url:
            try:
                from vllm import LLM as VLLModel
            except ImportError:
                raise ImportError(
                    "Could not import vllm python package. "
                    "Please install it with `pip install vllm`."
                )
            self._client = VLLModel(
                model=model,
                tensor_parallel_size=tensor_parallel_size,
                trust_remote_code=trust_remote_code,
                dtype=dtype,
                download_dir=download_dir,
                **vllm_kwargs,
            )
        else:
            self._client = None

    @classmethod
    def class_name(cls) -> str:
        return "Vllm"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(model_name=self.model, is_chat_model=self.is_chat_model)

    @property
    def _model_kwargs(self) -> Dict[str, Any]:
        base_kwargs = {
            "temperature": self.temperature,
            "max_tokens": self.max_new_tokens,
            "n": self.n,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "best_of": self.best_of,
            "ignore_eos": self.ignore_eos,
            "stop": self.stop,
            "logprobs": self.logprobs,
            "top_k": self.top_k,
            "top_p": self.top_p,
        }
        return {**base_kwargs}

    @atexit.register
    def close():
        import torch
        import gc

        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

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

        from vllm import SamplingParams

        # build sampling parameters
        sampling_params = SamplingParams(**params)
        outputs = self._client.generate([prompt], sampling_params)
        return CompletionResponse(text=outputs[0].outputs[0].text)

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        raise (ValueError("Not Implemented"))

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        raise (ValueError("Not Implemented"))

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
        raise (ValueError("Not Implemented"))

    @llm_completion_callback()
    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        raise (ValueError("Not Implemented"))


class VllmServer(Vllm):
    r"""
    Vllm LLM.

    This class connects to a vLLM server (non-openai versions).

    If using the OpenAI-API vLLM server, please see the `OpenAILike` LLM class.

    Examples:
        `pip install llama-index-llms-vllm`


        ```python
        from llama_index.llms.vllm import VllmServer

        # specific functions to format for mistral instruct
        def messages_to_prompt(messages):
            prompt = "\n".join([str(x) for x in messages])
            return f"<s>[INST] {prompt} [/INST] </s>\n"

        def completion_to_prompt(completion):
            return f"<s>[INST] {completion} [/INST] </s>\n"

        llm = VllmServer(
            api_url=api_url,
            max_new_tokens=256,
            temperature=0.1,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
        )

        llm.complete(
            "What is a black hole?"
        )
        ```

    """

    def __init__(
        self,
        model: str = "facebook/opt-125m",
        api_url: str = "http://localhost:8000",
        temperature: float = 1.0,
        tensor_parallel_size: Optional[int] = 1,
        trust_remote_code: Optional[bool] = True,
        n: int = 1,
        best_of: Optional[int] = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        top_p: float = 1.0,
        top_k: int = -1,
        stop: Optional[List[str]] = None,
        ignore_eos: bool = False,
        max_new_tokens: int = 512,
        logprobs: Optional[int] = None,
        dtype: str = "auto",
        download_dir: Optional[str] = None,
        messages_to_prompt: Optional[Callable] = None,
        completion_to_prompt: Optional[Callable] = None,
        vllm_kwargs: Dict[str, Any] = {},
        callback_manager: Optional[CallbackManager] = None,
        output_parser: Optional[BaseOutputParser] = None,
    ) -> None:
        messages_to_prompt = messages_to_prompt or generic_messages_to_prompt
        completion_to_prompt = completion_to_prompt or (lambda x: x)
        callback_manager = callback_manager or CallbackManager([])

        super().__init__(
            model=model,
            temperature=temperature,
            n=n,
            best_of=best_of,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            top_p=top_p,
            top_k=top_k,
            stop=stop,
            ignore_eos=ignore_eos,
            max_new_tokens=max_new_tokens,
            logprobs=logprobs,
            dtype=dtype,
            download_dir=download_dir,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            vllm_kwargs=vllm_kwargs,
            api_url=api_url,
            callback_manager=callback_manager,
            output_parser=output_parser,
        )
        self._client = None

    @classmethod
    def class_name(cls) -> str:
        return "VllmServer"

    def __del__(self) -> None: ...

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        kwargs = kwargs if kwargs else {}
        params = {**self._model_kwargs, **kwargs}

        # build sampling parameters
        sampling_params = dict(**params)
        sampling_params["prompt"] = prompt
        response = post_http_request(self.api_url, sampling_params, stream=False)
        output = get_response(response)

        return CompletionResponse(text=output[0])

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        kwargs = kwargs if kwargs else {}
        params = {**self._model_kwargs, **kwargs}

        sampling_params = dict(**params)
        sampling_params["prompt"] = prompt
        response = post_http_request(self.api_url, sampling_params, stream=True)

        def gen() -> CompletionResponseGen:
            response_str = ""
            prev_prefix_len = len(prompt)
            for chunk in response.iter_lines(
                chunk_size=8192, decode_unicode=False, delimiter=b"\0"
            ):
                if chunk:
                    data = json.loads(chunk.decode("utf-8"))

                    increasing_concat = data["text"][0]
                    pref = prev_prefix_len
                    prev_prefix_len = len(increasing_concat)
                    yield CompletionResponse(
                        text=increasing_concat, delta=increasing_concat[pref:]
                    )

        return gen()

    @llm_completion_callback()
    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        kwargs = kwargs if kwargs else {}
        return self.complete(prompt, **kwargs)

    @llm_completion_callback()
    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        kwargs = kwargs if kwargs else {}
        params = {**self._model_kwargs, **kwargs}

        # build sampling parameters
        sampling_params = dict(**params)
        sampling_params["prompt"] = prompt

        async def gen() -> CompletionResponseAsyncGen:
            for message in self.stream_complete(prompt, **kwargs):
                yield message

        return gen()

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        prompt = self.messages_to_prompt(messages)
        completion_response = self.stream_complete(prompt, **kwargs)
        return stream_completion_response_to_chat_response(completion_response)

    @llm_chat_callback()
    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        async def gen() -> ChatResponseAsyncGen:
            for message in self.stream_chat(messages, **kwargs):
                yield message

        return gen()


class VllmLLM(LLM):
    r"""
    VllmLLM - OpenAI-compatible vLLM server client.

    This class connects to a running ``vllm serve`` server via its
    OpenAI-compatible HTTP API using the ``openai`` Python client.

    Unlike the legacy :class:`Vllm` class this implementation does **not**
    require the ``vllm`` Python package to be installed locally – only the
    ``openai`` package is needed.

    Args:
        model (str):
            The model identifier served by vLLM
            (e.g. ``"mistralai/Mistral-7B-Instruct-v0.2"``).
        api_base (str):
            Base URL of the vLLM OpenAI-compatible server,
            e.g. ``"http://localhost:8000/v1"``.
        api_key (str):
            API key.  vLLM does not require one by default; pass any
            non-empty string (``"EMPTY"``).
        is_chat_model (bool):
            Whether to use the ``/chat/completions`` endpoint.
            Defaults to ``True``.
        context_window (int):
            Token context window.  Defaults to 4096.
        max_tokens (int):
            Maximum tokens to generate.  Defaults to 512.
        temperature (float):
            Sampling temperature.  Defaults to 0.1.
        additional_kwargs (dict):
            Extra parameters forwarded to the API request body.  Useful for
            vLLM-specific options such as ``guided_json`` or ``guided_regex``.

    Examples:
        ``pip install llama-index-llms-vllm``

        .. code-block:: python

            from llama_index.llms.vllm import VllmLLM

            llm = VllmLLM(
                model="mistralai/Mistral-7B-Instruct-v0.2",
                api_base="http://localhost:8000/v1",
                is_chat_model=True,
                context_window=32768,
            )

            response = llm.complete("What is a black hole?")
            print(response)
    """

    model: str = Field(
        default="facebook/opt-125m",
        description="Model identifier served by vLLM.",
    )
    api_base: str = Field(
        default="http://localhost:8000/v1",
        description="Base URL of the vLLM OpenAI-compatible server.",
    )
    api_key: str = Field(
        default="EMPTY",
        description="API key (any non-empty string when auth is disabled).",
    )
    is_chat_model: bool = Field(
        default=True,
        description=LLMMetadata.model_fields["is_chat_model"].description,
    )
    context_window: int = Field(
        default=DEFAULT_CONTEXT_WINDOW,
        description=LLMMetadata.model_fields["context_window"].description,
    )
    max_tokens: int = Field(
        default=512,
        description="Maximum number of tokens to generate.",
    )
    temperature: float = Field(
        default=0.1,
        description="Sampling temperature.",
    )
    additional_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="vLLM-specific extra parameters forwarded to the API.",
    )

    _client: Any = PrivateAttr()
    _aclient: Any = PrivateAttr()

    def __init__(
        self,
        model: str = "facebook/opt-125m",
        api_base: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        is_chat_model: bool = True,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        max_tokens: int = 512,
        temperature: float = 0.1,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        callback_manager: Optional[CallbackManager] = None,
        system_prompt: Optional[str] = None,
        messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]] = None,
        completion_to_prompt: Optional[Callable[[str], str]] = None,
        pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
        output_parser: Optional[BaseOutputParser] = None,
    ) -> None:
        try:
            from openai import AsyncOpenAI, OpenAI
        except ImportError:
            raise ImportError(
                "openai package is required for VllmLLM. "
                "Install it with: pip install openai"
            )
        super().__init__(
            model=model,
            api_base=api_base,
            api_key=api_key,
            is_chat_model=is_chat_model,
            context_window=context_window,
            max_tokens=max_tokens,
            temperature=temperature,
            additional_kwargs=additional_kwargs or {},
            callback_manager=callback_manager,
            system_prompt=system_prompt,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            pydantic_program_mode=pydantic_program_mode,
            output_parser=output_parser,
        )
        self._client = OpenAI(api_key=api_key, base_url=api_base)
        self._aclient = AsyncOpenAI(api_key=api_key, base_url=api_base)

    @classmethod
    def class_name(cls) -> str:
        return "VllmLLM"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            model_name=self.model,
            is_chat_model=self.is_chat_model,
            context_window=self.context_window,
            num_output=self.max_tokens,
        )

    def _build_chat_messages(
        self, messages: Sequence[ChatMessage]
    ) -> List[Dict[str, Any]]:
        return [{"role": m.role.value, "content": m.content} for m in messages]

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        params = {
            "model": self.model,
            "messages": self._build_chat_messages(messages),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **self.additional_kwargs,
            **kwargs,
        }
        response = self._client.chat.completions.create(**params)
        content = response.choices[0].message.content or ""
        return ChatResponse(
            message=ChatMessage(role="assistant", content=content),
            raw=response.model_dump(),
        )

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        params = {
            "model": self.model,
            "messages": self._build_chat_messages(messages),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
            **self.additional_kwargs,
            **kwargs,
        }
        stream = self._client.chat.completions.create(**params)

        def gen() -> ChatResponseGen:
            content = ""
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                content += delta
                yield ChatResponse(
                    message=ChatMessage(role="assistant", content=content),
                    delta=delta,
                    raw=chunk.model_dump(),
                )

        return gen()

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        if self.is_chat_model:
            chat_resp = self.chat(
                [ChatMessage(role="user", content=prompt)], **kwargs
            )
            return CompletionResponse(
                text=chat_resp.message.content or "",
                raw=chat_resp.raw,
            )
        params = {
            "model": self.model,
            "prompt": prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **self.additional_kwargs,
            **kwargs,
        }
        response = self._client.completions.create(**params)
        return CompletionResponse(
            text=response.choices[0].text,
            raw=response.model_dump(),
        )

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        if self.is_chat_model:
            chat_gen = self.stream_chat(
                [ChatMessage(role="user", content=prompt)], **kwargs
            )

            def chat_to_completion() -> CompletionResponseGen:
                for resp in chat_gen:
                    yield CompletionResponse(
                        text=resp.message.content or "",
                        delta=resp.delta,
                        raw=resp.raw,
                    )

            return chat_to_completion()

        params = {
            "model": self.model,
            "prompt": prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
            **self.additional_kwargs,
            **kwargs,
        }
        stream = self._client.completions.create(**params)

        def gen() -> CompletionResponseGen:
            text = ""
            for chunk in stream:
                delta = chunk.choices[0].text or ""
                text += delta
                yield CompletionResponse(
                    text=text,
                    delta=delta,
                    raw=chunk.model_dump(),
                )

        return gen()

    @llm_chat_callback()
    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        params = {
            "model": self.model,
            "messages": self._build_chat_messages(messages),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **self.additional_kwargs,
            **kwargs,
        }
        response = await self._aclient.chat.completions.create(**params)
        content = response.choices[0].message.content or ""
        return ChatResponse(
            message=ChatMessage(role="assistant", content=content),
            raw=response.model_dump(),
        )

    @llm_chat_callback()
    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        params = {
            "model": self.model,
            "messages": self._build_chat_messages(messages),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
            **self.additional_kwargs,
            **kwargs,
        }

        async def gen() -> ChatResponseAsyncGen:
            content = ""
            async with await self._aclient.chat.completions.create(**params) as stream:
                async for chunk in stream:
                    delta = chunk.choices[0].delta.content or ""
                    content += delta
                    yield ChatResponse(
                        message=ChatMessage(role="assistant", content=content),
                        delta=delta,
                        raw=chunk.model_dump(),
                    )

        return gen()

    @llm_completion_callback()
    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        chat_resp = await self.achat(
            [ChatMessage(role="user", content=prompt)], **kwargs
        )
        return CompletionResponse(
            text=chat_resp.message.content or "",
            raw=chat_resp.raw,
        )

    @llm_completion_callback()
    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        chat_gen = await self.astream_chat(
            [ChatMessage(role="user", content=prompt)], **kwargs
        )

        async def gen() -> CompletionResponseAsyncGen:
            async for resp in chat_gen:
                yield CompletionResponse(
                    text=resp.message.content or "",
                    delta=resp.delta,
                    raw=resp.raw,
                )

        return gen()


class VllmEmbedding(BaseEmbedding):
    r"""
    VllmEmbedding - OpenAI-compatible vLLM embedding client.

    Connects to a ``vllm serve`` instance that exposes the
    ``/v1/embeddings`` endpoint (available when the served model supports
    embeddings, e.g. ``BAAI/bge-base-en-v1.5``).

    Args:
        model_name (str):
            Embedding model served by vLLM.
        api_base (str):
            Base URL of the vLLM server, e.g. ``"http://localhost:8000/v1"``.
        api_key (str):
            API key.  Use any non-empty string if vLLM is running without
            auth.  Defaults to ``"EMPTY"``.
        embed_batch_size (int):
            Number of texts to embed in a single request.  Defaults to 10.
        dimensions (int | None):
            Optional – passed through to the API (not all models support it).

    Examples:
        .. code-block:: python

            from llama_index.llms.vllm import VllmEmbedding

            embed_model = VllmEmbedding(
                model_name="BAAI/bge-base-en-v1.5",
                api_base="http://localhost:8000/v1",
            )

            embeddings = embed_model.get_text_embedding("Hello world")
    """

    api_base: str = Field(
        default="http://localhost:8000/v1",
        description="Base URL of the vLLM OpenAI-compatible server.",
    )
    api_key: str = Field(
        default="EMPTY",
        description="API key (any non-empty string when auth is disabled).",
    )
    dimensions: Optional[int] = Field(
        default=None,
        description="Optional embedding dimensions parameter.",
    )

    _client: Any = PrivateAttr()
    _aclient: Any = PrivateAttr()

    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        api_base: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        dimensions: Optional[int] = None,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_name=model_name,
            api_base=api_base,
            api_key=api_key,
            embed_batch_size=embed_batch_size,
            dimensions=dimensions,
            callback_manager=callback_manager,
            **kwargs,
        )
        try:
            from openai import AsyncOpenAI, OpenAI
        except ImportError:
            raise ImportError(
                "openai package is required for VllmEmbedding. "
                "Install it with: pip install openai"
            )
        self._client = OpenAI(api_key=api_key, base_url=api_base)
        self._aclient = AsyncOpenAI(api_key=api_key, base_url=api_base)

    @classmethod
    def class_name(cls) -> str:
        return "VllmEmbedding"

    def _embed(self, texts: List[str]) -> List[List[float]]:
        kwargs: Dict[str, Any] = {"input": texts, "model": self.model_name}
        if self.dimensions is not None:
            kwargs["dimensions"] = self.dimensions
        response = self._client.embeddings.create(**kwargs)
        return [item.embedding for item in response.data]

    async def _aembed(self, texts: List[str]) -> List[List[float]]:
        kwargs: Dict[str, Any] = {"input": texts, "model": self.model_name}
        if self.dimensions is not None:
            kwargs["dimensions"] = self.dimensions
        response = await self._aclient.embeddings.create(**kwargs)
        return [item.embedding for item in response.data]

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._embed([query])[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._embed([text])[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return (await self._aembed([query]))[0]

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return (await self._aembed([text]))[0]

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return await self._aembed(texts)


class VllmRerank(BaseNodePostprocessor):
    r"""
    VllmRerank - Cohere-compatible vLLM reranking client.

    Connects to a ``vllm serve`` instance that exposes the
    ``/v1/rerank`` endpoint using the Cohere-compatible reranking API.
    This is supported by vLLM when serving a cross-encoder / reranker model
    (e.g. ``BAAI/bge-reranker-base``).

    Args:
        model (str):
            Reranker model served by vLLM.
        api_base (str):
            Base URL of the vLLM server, e.g. ``"http://localhost:8000"``.
            The ``/v1/rerank`` path is appended automatically.
        api_key (str):
            API key.  Use any non-empty string when auth is disabled.
        top_n (int):
            Number of highest-scoring nodes to return.  Defaults to 3.

    Examples:
        .. code-block:: python

            from llama_index.llms.vllm import VllmRerank

            reranker = VllmRerank(
                model="BAAI/bge-reranker-base",
                api_base="http://localhost:8000",
                top_n=5,
            )

            nodes = reranker.postprocess_nodes(nodes, query_bundle=query)
    """

    model: str = Field(description="Reranker model name served by vLLM.")
    api_base: str = Field(
        default="http://localhost:8000",
        description="Base URL of the vLLM server (without /v1/rerank).",
    )
    api_key: str = Field(
        default="EMPTY",
        description="API key (any non-empty string when auth is disabled).",
    )
    top_n: int = Field(default=3, description="Number of top nodes to return.")

    @classmethod
    def class_name(cls) -> str:
        return "VllmRerank"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        if query_bundle is None:
            raise ValueError("VllmRerank requires a query bundle.")
        if not nodes:
            return []

        try:
            import requests as _requests
        except ImportError:
            raise ImportError(
                "requests package is required for VllmRerank. "
                "Install it with: pip install requests"
            )

        documents = [
            node.node.get_content(metadata_mode=MetadataMode.EMBED) for node in nodes
        ]
        endpoint = self.api_base.rstrip("/") + "/v1/rerank"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model,
            "query": query_bundle.query_str,
            "documents": documents,
            "top_n": self.top_n,
        }
        response = _requests.post(endpoint, json=payload, headers=headers)
        response.raise_for_status()
        results = response.json().get("results", [])

        reranked: List[NodeWithScore] = []
        for result in results:
            idx = result["index"]
            score = result.get("relevance_score", 0.0)
            reranked.append(NodeWithScore(node=nodes[idx].node, score=score))

        return reranked
