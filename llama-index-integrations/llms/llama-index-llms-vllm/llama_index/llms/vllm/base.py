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
from llama_index.core.llms.llm import LLM
from llama_index.core.types import BaseOutputParser, PydanticProgramMode
from llama_index.llms.vllm.utils import get_response, post_http_request
import atexit


class Vllm(LLM):
    r"""Vllm LLM.

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
                **vllm_kwargs
            )
        else:
            self._client = None

    @classmethod
    def class_name(cls) -> str:
        return "Vllm"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(model_name=self.model)

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
    r"""Vllm LLM.

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

    def __del__(self) -> None:
        ...

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
