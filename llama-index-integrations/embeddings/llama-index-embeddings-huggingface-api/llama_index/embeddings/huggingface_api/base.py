import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from huggingface_hub import (
    AsyncInferenceClient,
    InferenceClient,
    model_info,
)
from huggingface_hub.hf_api import ModelInfo
from llama_index.core.base.embeddings.base import (
    BaseEmbedding,
    Embedding,
)
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.embeddings.huggingface_api.pooling import Pooling
from llama_index.utils.huggingface import (
    format_query,
    format_text,
)

logger = logging.getLogger(__name__)


class HuggingFaceInferenceAPIEmbedding(BaseEmbedding):  # type: ignore[misc]
    """
    Wrapper on the Hugging Face's Inference API for embeddings.

    Overview of the design:
    - Uses the feature extraction task: https://huggingface.co/tasks/feature-extraction
    """

    pooling: Optional[Pooling] = Field(
        default=Pooling.CLS,
        description="Pooling strategy. If None, the model's default pooling is used.",
    )
    query_instruction: Optional[str] = Field(
        default=None, description="Instruction to prepend during query embedding."
    )
    text_instruction: Optional[str] = Field(
        default=None, description="Instruction to prepend during text embedding."
    )

    # Corresponds with huggingface_hub.InferenceClient
    model_name: Optional[str] = Field(
        default=None,
        description="Hugging Face model name. If None, the task will be used.",
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
    headers: Optional[Dict[str, str]] = Field(
        default=None,
        description=(
            "Additional headers to send to the server. By default only the"
            " authorization and user-agent headers are sent. Values in this dictionary"
            " will override the default values."
        ),
    )
    cookies: Optional[Dict[str, str]] = Field(
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
        """
        Initialize.

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
            print(kwargs["model_name"], flush=True)
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

    @classmethod
    def class_name(cls) -> str:
        return "HuggingFaceInferenceAPIEmbedding"

    async def _async_embed_single(self, text: str) -> Embedding:
        embedding = await self._async_client.feature_extraction(text)
        if len(embedding.shape) == 1:
            return embedding.tolist()
        embedding = embedding.squeeze(axis=0)
        if len(embedding.shape) == 1:  # Some models pool internally
            return embedding.tolist()
        try:
            return self.pooling(embedding).tolist()  # type: ignore[misc]
        except TypeError as exc:
            raise ValueError(
                f"Pooling is required for {self.model_name} because it returned"
                " a > 1-D value, please specify pooling as not None."
            ) from exc

    async def _async_embed_bulk(self, texts: Sequence[str]) -> List[Embedding]:
        """
        Embed a sequence of text, in parallel and asynchronously.

        NOTE: this uses an externally created asyncio event loop.
        """
        tasks = [self._async_embed_single(text) for text in texts]
        return await asyncio.gather(*tasks)

    def _get_query_embedding(self, query: str) -> Embedding:
        """
        Embed the input query synchronously.

        NOTE: a new asyncio event loop is created internally for this.
        """
        return asyncio.run(self._aget_query_embedding(query))

    def _get_text_embedding(self, text: str) -> Embedding:
        """
        Embed the text query synchronously.

        NOTE: a new asyncio event loop is created internally for this.
        """
        return asyncio.run(self._aget_text_embedding(text))

    def _get_text_embeddings(self, texts: List[str]) -> List[Embedding]:
        """
        Embed the input sequence of text synchronously and in parallel.

        NOTE: a new asyncio event loop is created internally for this.
        """
        loop = asyncio.new_event_loop()
        try:
            tasks = [
                loop.create_task(self._aget_text_embedding(text)) for text in texts
            ]
            loop.run_until_complete(asyncio.wait(tasks))
        finally:
            loop.close()
        return [task.result() for task in tasks]

    async def _aget_query_embedding(self, query: str) -> Embedding:
        return await self._async_embed_single(
            text=format_query(query, self.model_name, self.query_instruction)
        )

    async def _aget_text_embedding(self, text: str) -> Embedding:
        return await self._async_embed_single(
            text=format_text(text, self.model_name, self.text_instruction)
        )

    async def _aget_text_embeddings(self, texts: List[str]) -> List[Embedding]:
        return await self._async_embed_bulk(
            texts=[
                format_text(text, self.model_name, self.text_instruction)
                for text in texts
            ]
        )
