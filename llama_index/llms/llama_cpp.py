import os
import requests
from tqdm import tqdm
from typing import Any, Callable, Dict, Optional, Sequence

from llama_index.callbacks import CallbackManager
from llama_index.constants import DEFAULT_CONTEXT_WINDOW, DEFAULT_NUM_OUTPUTS
from llama_index.llms.base import (
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
    llm_chat_callback,
    llm_completion_callback,
)
from llama_index.llms.custom import CustomLLM
from llama_index.llms.generic_utils import completion_response_to_chat_response
from llama_index.llms.generic_utils import (
    messages_to_prompt as generic_messages_to_prompt,
)
from llama_index.llms.generic_utils import stream_completion_response_to_chat_response
from llama_index.utils import get_cache_dir


DEFAULT_LLAMA_CPP_MODEL = (
    "https://huggingface.co/TheBloke/Llama-2-13B-chat-GGML/resolve"
    "/main/llama-2-13b-chat.ggmlv3.q4_0.bin"
)


class LlamaCPP(CustomLLM):
    def __init__(
        self,
        model_url: str = DEFAULT_LLAMA_CPP_MODEL,
        model_path: Optional[str] = None,
        temperature: float = 0.1,
        max_new_tokens: int = DEFAULT_NUM_OUTPUTS,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        messages_to_prompt: Optional[Callable] = None,
        completion_to_prompt: Optional[Callable] = None,
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = True,
        generate_kwargs: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "Could not import llama_cpp library."
                "Please install llama_cpp with `pip install llama-cpp-python`."
                "See the full installation guide for GPU support at "
                "`https://github.com/abetlen/llama-cpp-python`"
            )

        self._model_kwargs = model_kwargs or {}
        self._model_kwargs.update({"n_ctx": context_window, "verbose": verbose})

        # check if model is cached
        if model_path is not None:
            if not os.path.exists(model_path):
                raise ValueError(
                    "Provided model path does not exist. "
                    "Please check the path or provide a model_url to download."
                )
            else:
                self._model = Llama(model_path=model_path, **self._model_kwargs)
        else:
            cache_dir = get_cache_dir()
            model_name = os.path.basename(model_url)
            model_path = os.path.join(cache_dir, "models", model_name)
            if not os.path.exists(model_path):
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                self._download_url(model_url, model_path)
                assert os.path.exists(model_path)

            self._model = Llama(model_path=model_path, **self._model_kwargs)

        self._model_path = model_path
        self._messages_to_prompt = messages_to_prompt or generic_messages_to_prompt
        self._completion_to_prompt = completion_to_prompt or (lambda x: x)
        self.callback_manager = callback_manager or CallbackManager([])

        # model kwargs
        self._context_window = context_window
        self._temperature = temperature
        self._max_new_tokens = max_new_tokens
        self._generate_kwargs = generate_kwargs or {}
        self._generate_kwargs.update(
            {"temperature": temperature, "max_tokens": max_new_tokens}
        )

    @property
    def metadata(self) -> LLMMetadata:
        """LLM metadata."""
        return LLMMetadata(
            context_window=self._context_window,
            num_output=self._max_new_tokens,
            model_name=self._model_path,
        )

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
        prompt = self._messages_to_prompt(messages)
        completion_response = self.complete(prompt, **kwargs)
        return completion_response_to_chat_response(completion_response)

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        prompt = self._messages_to_prompt(messages)
        completion_response = self.stream_complete(prompt, **kwargs)
        return stream_completion_response_to_chat_response(completion_response)

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        self._generate_kwargs.update({"stream": False})
        prompt = self._completion_to_prompt(prompt)

        response = self._model(prompt=prompt, **self._generate_kwargs)

        return CompletionResponse(text=response["choices"][0]["text"], raw=response)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        self._generate_kwargs.update({"stream": True})
        prompt = self._completion_to_prompt(prompt)

        response_iter = self._model(prompt=prompt, **self._generate_kwargs)

        def gen() -> CompletionResponseGen:
            text = ""
            for response in response_iter:
                delta = response["choices"][0]["text"]
                text += delta
                yield CompletionResponse(delta=delta, text=text, raw=response)

        return gen()
