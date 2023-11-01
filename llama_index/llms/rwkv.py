import os
from typing import Any, Optional, Sequence

import requests
from tqdm import tqdm

from llama_index.bridge.pydantic import Field, PrivateAttr
from llama_index.llms import (
    CompletionResponse,
    CustomLLM,
    LLMMetadata,
)
from llama_index.llms.base import (
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
    llm_completion_callback,
)
from llama_index.utils import get_cache_dir

context_window = 1536
num_output = 300

DEFAULT_RWKV_MODEL = "https://huggingface.co/BlinkDL/rwkv-4-world/resolve/main/RWKV-4-World-0.4B-v1-20230529-ctx4096.pth"


class RWKVModel(CustomLLM):
    _model: Any = PrivateAttr()
    _pipeline: Any = None
    _args: Any = None

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_url: Optional[str] = None,
    ):
        # Setting environment variables
        os.environ["RWKV_JIT_ON"] = "1"
        os.environ["RWKV_CUDA_ON"] = "0"

        try:
            from rwkv.model import RWKV
            from rwkv.utils import PIPELINE, PIPELINE_ARGS
        except ImportError as e:
            raise ImportError(
                "Required modules for RWKVModel are not available. Ensure you have installed the necessary dependencies."
            ) from e

        strategy = "cpu fp32"
        tokenizer_path = "rwkv_vocab_v20230424"

        if model_path is not None:
            if not os.path.exists(model_path):
                raise ValueError(
                    "Provided model path does not exist. "
                    "Please check the path or provide a model_url to download."
                )
            else:
                self._model = RWKV(model=model_path, strategy=strategy)
        else:
            cache_dir = get_cache_dir()
            model_url = model_url or DEFAULT_RWKV_MODEL
            model_name = os.path.basename(model_url)
            model_path = os.path.join(cache_dir, "models", model_name)

            if not os.path.exists(model_path):
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                self._download_url(model_url, model_path)
                assert os.path.exists(model_path)

            self._model = RWKV(model=model_path, strategy=strategy)

        self._pipeline = PIPELINE(self._model, tokenizer_path)
        self._args = PIPELINE_ARGS(
            temperature=1.0,
            top_p=0.7,
            top_k=100,
            alpha_frequency=0.25,
            alpha_presence=0.25,
            alpha_decay=0.996,
            token_ban=[0],
            token_stop=[],
            chunk_len=256,
        )

    def my_print(self, text: str) -> None:
        return

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

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        result = self._pipeline.generate(
            prompt, token_count=200, args=self._args, callback=self.my_print
        )
        return CompletionResponse(text=result)

    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        # Assuming the callback for `pipeline.generate` yields results
        # For now, this implementation might be simplistic and not perfect
        yield self._pipeline.generate(
            prompt, token_count=200, args=self._args, callback=None
        )

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        raise NotImplementedError(
            "stream_chat is not implemented for RWKVModel. " "Please use chat instead."
        )

    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        raise NotImplementedError(
            "stream_chat is not implemented for RWKVModel. " "Please use chat instead."
        )

    @classmethod
    def class_name(cls) -> str:
        return "rwkv_model"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=context_window,
            num_output=num_output,
            model_name=Field(
                "rwkv_model",
                description="The name of the model used for the LLM.",
            ),
        )
