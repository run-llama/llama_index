import os
from typing import Any, Callable, Dict, Optional, Sequence

import requests
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CallbackManager
from llama_index.core.constants import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_NUM_OUTPUTS,
    DEFAULT_TEMPERATURE,
)
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from llama_index.core.llms.custom import CustomLLM
from llama_index.core.base.llms.generic_utils import (
    completion_response_to_chat_response,
    stream_completion_response_to_chat_response,
)
from llama_index.core.types import BaseOutputParser, PydanticProgramMode
from llama_index.core.utils import get_cache_dir
from tqdm import tqdm

from llama_cpp import Llama

DEFAULT_LLAMA_CPP_GGML_MODEL = (
    "https://huggingface.co/TheBloke/Llama-2-13B-chat-GGML/resolve"
    "/main/llama-2-13b-chat.ggmlv3.q4_0.bin"
)
DEFAULT_LLAMA_CPP_GGUF_MODEL = (
    "https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve"
    "/main/llama-2-13b-chat.Q4_0.gguf"
)
DEFAULT_LLAMA_CPP_MODEL_VERBOSITY = True


class LlamaCPP(CustomLLM):
    r"""
    LlamaCPP LLM.

    Examples:
        Install llama-cpp-python following instructions:
        https://github.com/abetlen/llama-cpp-python

        Then `pip install llama-index-llms-llama-cpp`

        ```python
        from llama_index.llms.llama_cpp import LlamaCPP

        def messages_to_prompt(messages):
            prompt = ""
            for message in messages:
                if message.role == 'system':
                prompt += f"<|system|>\n{message.content}</s>\n"
                elif message.role == 'user':
                prompt += f"<|user|>\n{message.content}</s>\n"
                elif message.role == 'assistant':
                prompt += f"<|assistant|>\n{message.content}</s>\n"

            # ensure we start with a system prompt, insert blank if needed
            if not prompt.startswith("<|system|>\n"):
                prompt = "<|system|>\n</s>\n" + prompt

            # add final assistant prompt
            prompt = prompt + "<|assistant|>\n"

            return prompt

        def completion_to_prompt(completion):
            return f"<|system|>\n</s>\n<|user|>\n{completion}</s>\n<|assistant|>\n"

        model_url = "https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF/resolve/main/zephyr-7b-beta.Q4_0.gguf"

        llm = LlamaCPP(
            model_url=model_url,
            model_path=None,
            temperature=0.1,
            max_new_tokens=256,
            context_window=3900,
            generate_kwargs={},
            model_kwargs={"n_gpu_layers": -1},  # if compiled to use GPU
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            verbose=True,
        )

        response = llm.complete("Hello, how are you?")
        print(str(response))
        ```

    """

    model_url: Optional[str] = Field(
        description="The URL llama-cpp model to download and use."
    )
    model_path: Optional[str] = Field(
        description="The path to the llama-cpp model to use."
    )
    temperature: float = Field(
        default=DEFAULT_TEMPERATURE,
        description="The temperature to use for sampling.",
        ge=0.0,
        le=1.0,
    )
    max_new_tokens: int = Field(
        default=DEFAULT_NUM_OUTPUTS,
        description="The maximum number of tokens to generate.",
        gt=0,
    )
    context_window: int = Field(
        default=DEFAULT_CONTEXT_WINDOW,
        description="The maximum number of context tokens for the model.",
        gt=0,
    )
    generate_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Kwargs used for generation."
    )
    model_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Kwargs used for model initialization."
    )
    verbose: bool = Field(
        default=DEFAULT_LLAMA_CPP_MODEL_VERBOSITY,
        description="Whether to print verbose output.",
    )

    _model: Any = PrivateAttr()

    def __init__(
        self,
        model_url: Optional[str] = None,
        model_path: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_new_tokens: int = DEFAULT_NUM_OUTPUTS,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        callback_manager: Optional[CallbackManager] = None,
        generate_kwargs: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        verbose: bool = DEFAULT_LLAMA_CPP_MODEL_VERBOSITY,
        system_prompt: Optional[str] = None,
        messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]] = None,
        completion_to_prompt: Optional[Callable[[str], str]] = None,
        pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
        output_parser: Optional[BaseOutputParser] = None,
    ) -> None:
        model_kwargs = {
            **{"n_ctx": context_window, "verbose": verbose},
            **(model_kwargs or {}),  # Override defaults via model_kwargs
        }

        # check if model is cached
        if model_path is not None:
            if not os.path.exists(model_path):
                raise ValueError(
                    "Provided model path does not exist. "
                    "Please check the path or provide a model_url to download."
                )
            else:
                model = Llama(model_path=model_path, **model_kwargs)
        else:
            cache_dir = get_cache_dir()
            model_url = model_url or self._get_model_path_for_version()
            model_name = os.path.basename(model_url)
            model_path = os.path.join(cache_dir, "models", model_name)
            if not os.path.exists(model_path):
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                self._download_url(model_url, model_path)
                assert os.path.exists(model_path)

            model = Llama(model_path=model_path, **model_kwargs)

        model_path = model_path
        generate_kwargs = generate_kwargs or {}
        generate_kwargs.update(
            {"temperature": temperature, "max_tokens": max_new_tokens}
        )

        super().__init__(
            model_path=model_path,
            model_url=model_url,
            temperature=temperature,
            context_window=context_window,
            max_new_tokens=max_new_tokens,
            callback_manager=callback_manager,
            generate_kwargs=generate_kwargs,
            model_kwargs=model_kwargs,
            verbose=verbose,
            system_prompt=system_prompt,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            pydantic_program_mode=pydantic_program_mode,
            output_parser=output_parser,
        )
        self._model = model

    @classmethod
    def class_name(cls) -> str:
        return "LlamaCPP_llm"

    @property
    def metadata(self) -> LLMMetadata:
        """LLM metadata."""
        return LLMMetadata(
            context_window=self._model.context_params.n_ctx,
            num_output=self.max_new_tokens,
            model_name=self.model_path,
        )

    def _get_model_path_for_version(self) -> str:
        """Get model path for the current llama-cpp version."""
        import pkg_resources

        version = pkg_resources.get_distribution("llama-cpp-python").version
        major, minor, patch = version.split(".")

        # NOTE: llama-cpp-python<=0.1.78 supports GGML, newer support GGUF
        if int(major) <= 0 and int(minor) <= 1 and int(patch) <= 78:
            return DEFAULT_LLAMA_CPP_GGML_MODEL
        else:
            return DEFAULT_LLAMA_CPP_GGUF_MODEL

    def _download_url(self, model_url: str, model_path: str) -> None:
        completed = False
        try:
            print("Downloading url", model_url, "to path", model_path)
            with requests.get(model_url, stream=True) as r:
                with open(model_path, "wb") as file:
                    total_size = int(r.headers.get("Content-Length") or "0")
                    if total_size < 1000 * 1000:
                        raise ValueError(
                            "Content should be at least 1 MB, but is only",
                            r.headers.get("Content-Length"),
                            "bytes",
                        )
                    print("total size (MB):", round(total_size / 1000 / 1000, 2))
                    chunk_size = 1024 * 1024  # 1 MB
                    for chunk in tqdm(
                        r.iter_content(chunk_size=chunk_size),
                        total=int(total_size / chunk_size),
                    ):
                        file.write(chunk)
            completed = True
        except Exception as e:
            print("Error downloading model:", e)
        finally:
            if not completed:
                print("Download incomplete.", "Removing partially downloaded file.")
                os.remove(model_path)
                raise ValueError("Download incomplete.")

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        prompt = self.messages_to_prompt(messages)
        completion_response = self.complete(prompt, formatted=True, **kwargs)
        return completion_response_to_chat_response(completion_response)

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        prompt = self.messages_to_prompt(messages)
        completion_response = self.stream_complete(prompt, formatted=True, **kwargs)
        return stream_completion_response_to_chat_response(completion_response)

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        self.generate_kwargs.update({"stream": False})

        if not formatted:
            prompt = self.completion_to_prompt(prompt)

        response = self._model(prompt=prompt, **self.generate_kwargs)

        return CompletionResponse(text=response["choices"][0]["text"], raw=response)

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        self.generate_kwargs.update({"stream": True})

        if not formatted:
            prompt = self.completion_to_prompt(prompt)

        response_iter = self._model(prompt=prompt, **self.generate_kwargs)

        def gen() -> CompletionResponseGen:
            text = ""
            for response in response_iter:
                delta = response["choices"][0]["text"]
                text += delta
                yield CompletionResponse(delta=delta, text=text, raw=response)

        return gen()
