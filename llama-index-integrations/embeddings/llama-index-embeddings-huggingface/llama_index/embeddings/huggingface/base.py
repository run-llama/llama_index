import asyncio
from io import BytesIO
import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from deprecated import deprecated
from huggingface_hub import (
    AsyncInferenceClient,
    InferenceClient,
    model_info,
)
from huggingface_hub.hf_api import ModelInfo
from llama_index.core.base.embeddings.base import (
    DEFAULT_EMBED_BATCH_SIZE,
    BaseEmbedding,
    Embedding,
)
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CallbackManager
from llama_index.embeddings.huggingface.pooling import Pooling
from llama_index.core.embeddings.multi_modal_base import MultiModalEmbedding
from llama_index.core.utils import get_cache_dir, infer_torch_device
from llama_index.embeddings.huggingface.utils import (
    DEFAULT_HUGGINGFACE_EMBEDDING_MODEL,
    format_query,
    format_text,
    get_query_instruct_for_model_name,
    get_text_instruct_for_model_name,
)
from llama_index.core.schema import ImageType
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_exponential

DEFAULT_HUGGINGFACE_LENGTH = 512
logger = logging.getLogger(__name__)


class HuggingFaceEmbedding(MultiModalEmbedding):
    """
    HuggingFace class for text and image embeddings.

    Args:
        model_name (str, optional): If it is a filepath on disc, it loads the model from that path.
            If it is not a path, it first tries to download a pre-trained SentenceTransformer model.
            If that fails, tries to construct a model from the Hugging Face Hub with that name.
            Defaults to DEFAULT_HUGGINGFACE_EMBEDDING_MODEL.
        max_length (Optional[int], optional): Max sequence length to set in Model's config. If None,
            it will use the Model's default max_seq_length. Defaults to None.
        query_instruction (Optional[str], optional): Instruction to prepend to query text.
            Defaults to None.
        text_instruction (Optional[str], optional): Instruction to prepend to text.
            Defaults to None.
        normalize (bool, optional): Whether to normalize returned vectors.
            Defaults to True.
        embed_batch_size (int, optional): The batch size used for the computation.
            Defaults to DEFAULT_EMBED_BATCH_SIZE.
        cache_folder (Optional[str], optional): Path to store models. Defaults to None.
        trust_remote_code (bool, optional): Whether or not to allow for custom models defined on the
            Hub in their own modeling files. This option should only be set to True for repositories
            you trust and in which you have read the code, as it will execute code present on the Hub
            on your local machine. Defaults to False.
        device (Optional[str], optional): Device (like "cuda", "cpu", "mps", "npu", ...) that should
            be used for computation. If None, checks if a GPU can be used. Defaults to None.
        callback_manager (Optional[CallbackManager], optional): Callback Manager. Defaults to None.
        parallel_process (bool, optional): If True it will start a multi-process pool to process the
            encoding with several independent processes. Great for vast amount of texts.
            Defaults to False.
        target_devices (Optional[List[str]], optional): PyTorch target devices, e.g.
            ["cuda:0", "cuda:1", ...], ["npu:0", "npu:1", ...], or ["cpu", "cpu", "cpu", "cpu"].
            If target_devices is None and CUDA/NPU is available, then all available CUDA/NPU devices
            will be used. If target_devices is None and CUDA/NPU is not available, then 4 CPU devices
            will be used. This parameter will only be used if `parallel_process = True`.
            Defaults to None.
        num_workers (int, optional): The number of workers to use for async embedding calls.
            Defaults to None.
        show_progress_bar (bool, optional): Whether to show a progress bar.
            Defaults to False.
        **model_kwargs: Other model kwargs to use
        tokenizer_name (Optional[str], optional): "Deprecated"
        pooling (str, optional): "Deprecated"
        model (Optional[Any], optional): "Deprecated"
        tokenizer (Optional[Any], optional): "Deprecated"

    Examples:
        `pip install llama-index-embeddings-huggingface`

        ```python
        from llama_index.core import Settings
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        # Set up the HuggingFaceEmbedding class with the required model to use with llamaindex core.
        embed_model  = HuggingFaceEmbedding(model_name = "BAAI/bge-small-en")
        Settings.embed_model = embed_model

        # Or if you want to Embed some text separately
        embeddings = embed_model.get_text_embedding("I want to Embed this text!")

        ```

    """

    max_length: int = Field(
        default=DEFAULT_HUGGINGFACE_LENGTH, description="Maximum length of input.", gt=0
    )
    normalize: bool = Field(default=True, description="Normalize embeddings or not.")
    query_instruction: Optional[str] = Field(
        description="Instruction to prepend to query text.", default=None
    )
    text_instruction: Optional[str] = Field(
        description="Instruction to prepend to text.", default=None
    )
    cache_folder: Optional[str] = Field(
        description="Cache folder for Hugging Face files.", default=None
    )
    show_progress_bar: bool = Field(
        description="Whether to show a progress bar.", default=False
    )
    _model: SentenceTransformer = PrivateAttr()
    _device: str = PrivateAttr()
    _parallel_process: bool = PrivateAttr()
    _target_devices: Optional[List[str]] = PrivateAttr()

    def __init__(
        self,
        model_name: str = DEFAULT_HUGGINGFACE_EMBEDDING_MODEL,
        tokenizer_name: Optional[str] = "deprecated",
        pooling: str = "deprecated",
        max_length: Optional[int] = None,
        query_instruction: Optional[str] = None,
        text_instruction: Optional[str] = None,
        normalize: bool = True,
        model: Optional[Any] = "deprecated",
        tokenizer: Optional[Any] = "deprecated",
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        cache_folder: Optional[str] = None,
        trust_remote_code: bool = False,
        device: Optional[str] = None,
        callback_manager: Optional[CallbackManager] = None,
        parallel_process: bool = False,
        target_devices: Optional[List[str]] = None,
        show_progress_bar: bool = False,
        **model_kwargs,
    ):
        device = device or infer_torch_device()
        cache_folder = cache_folder or get_cache_dir()

        for variable, value in [
            ("model", model),
            ("tokenizer", tokenizer),
            ("pooling", pooling),
            ("tokenizer_name", tokenizer_name),
        ]:
            if value != "deprecated":
                raise ValueError(
                    f"{variable} is deprecated. Please remove it from the arguments."
                )
        if model_name is None:
            raise ValueError("The `model_name` argument must be provided.")

        model = SentenceTransformer(
            model_name,
            device=device,
            cache_folder=cache_folder,
            trust_remote_code=trust_remote_code,
            prompts={
                "query": query_instruction
                or get_query_instruct_for_model_name(model_name),
                "text": text_instruction
                or get_text_instruct_for_model_name(model_name),
            },
            **model_kwargs,
        )
        if max_length:
            model.max_seq_length = max_length
        else:
            max_length = model.max_seq_length

        super().__init__(
            embed_batch_size=embed_batch_size,
            callback_manager=callback_manager,
            model_name=model_name,
            max_length=max_length,
            normalize=normalize,
            query_instruction=query_instruction,
            text_instruction=text_instruction,
            show_progress_bar=show_progress_bar,
        )
        self._device = device
        self._model = model
        self._parallel_process = parallel_process
        self._target_devices = target_devices

    @classmethod
    def class_name(cls) -> str:
        return "HuggingFaceEmbedding"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True,
    )
    def _embed_with_retry(
        self,
        inputs: List[Union[str, BytesIO]],
        prompt_name: Optional[str] = None,
    ) -> List[List[float]]:
        """
        Generates embeddings with retry mechanism.

        Args:
            inputs: List of texts or images to embed
            prompt_name: Optional prompt type

        Returns:
            List of embedding vectors

        Raises:
            Exception: If embedding fails after retries

        """
        try:
            if self._parallel_process:
                pool = self._model.start_multi_process_pool(
                    target_devices=self._target_devices
                )
                emb = self._model.encode_multi_process(
                    inputs,
                    pool=pool,
                    batch_size=self.embed_batch_size,
                    prompt_name=prompt_name,
                    normalize_embeddings=self.normalize,
                    show_progress_bar=self.show_progress_bar,
                )
                self._model.stop_multi_process_pool(pool=pool)
            else:
                emb = self._model.encode(
                    inputs,
                    batch_size=self.embed_batch_size,
                    prompt_name=prompt_name,
                    normalize_embeddings=self.normalize,
                    show_progress_bar=self.show_progress_bar,
                )
            return emb.tolist()
        except Exception as e:
            logger.warning(f"Embedding attempt failed: {e!s}")
            raise

    def _embed(
        self,
        inputs: List[Union[str, BytesIO]],
        prompt_name: Optional[str] = None,
    ) -> List[List[float]]:
        """
        Generates Embeddings with input validation and retry mechanism.

        Args:
            sentences: Texts or Sentences to embed
            prompt_name: The name of the prompt to use for encoding

        Returns:
            List of embedding vectors

        Raises:
            ValueError: If any input text is invalid
            Exception: If embedding fails after retries

        """
        return self._embed_with_retry(inputs, prompt_name)

    def _get_query_embedding(self, query: str) -> List[float]:
        """
        Generates Embeddings for Query.

        Args:
            query (str): Query text/sentence

        Returns:
            List[float]: numpy array of embeddings

        """
        return self._embed([query], prompt_name="query")[0]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """
        Generates Embeddings for Query Asynchronously.

        Args:
            query (str): Query text/sentence

        Returns:
            List[float]: numpy array of embeddings

        """
        return await asyncio.to_thread(self._get_query_embedding, query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """
        Generates Embeddings for text Asynchronously.

        Args:
            text (str): Text/Sentence

        Returns:
            List[float]: numpy array of embeddings

        """
        return await asyncio.to_thread(self._get_text_embedding, text)

    def _get_text_embedding(self, text: str) -> List[float]:
        """
        Generates Embeddings for text.

        Args:
            text (str): Text/sentences

        Returns:
            List[float]: numpy array of embeddings

        """
        return self._embed([text], prompt_name="text")[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generates Embeddings for text.

        Args:
            texts (List[str]): Texts / Sentences

        Returns:
            List[List[float]]: numpy array of embeddings

        """
        return self._embed(texts, prompt_name="text")

    def _get_image_embedding(self, img_file_path: ImageType) -> List[float]:
        """Generate embedding for an image."""
        return self._embed([img_file_path])[0]

    async def _aget_image_embedding(self, img_file_path: ImageType) -> List[float]:
        """Generate embedding for an image asynchronously."""
        return self._get_image_embedding(img_file_path)

    def _get_image_embeddings(
        self, img_file_paths: List[ImageType]
    ) -> List[List[float]]:
        """Generate embeddings for multiple images."""
        return self._embed(img_file_paths)

    async def _aget_image_embeddings(
        self, img_file_paths: List[ImageType]
    ) -> List[List[float]]:
        """Generate embeddings for multiple images asynchronously."""
        return self._get_image_embeddings(img_file_paths)


@deprecated(
    "Deprecated in favor of `HuggingFaceInferenceAPIEmbedding` from `llama-index-embeddings-huggingface-api` which should be used instead.",
    action="always",
)
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
    headers: Dict[str, str] = Field(
        default=None,
        description=(
            "Additional headers to send to the server. By default only the"
            " authorization and user-agent headers are sent. Values in this dictionary"
            " will override the default values."
        ),
    )
    cookies: Dict[str, str] = Field(
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


HuggingFaceInferenceAPIEmbeddings = HuggingFaceInferenceAPIEmbedding
