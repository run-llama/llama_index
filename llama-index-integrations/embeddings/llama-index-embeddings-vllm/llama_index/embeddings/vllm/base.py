from io import BytesIO
import logging
from typing import Any, Dict, List, Optional, Union

from llama_index.core.base.embeddings.base import DEFAULT_EMBED_BATCH_SIZE
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CallbackManager
from llama_index.core.embeddings.multi_modal_base import MultiModalEmbedding
from llama_index.core.schema import ImageType
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential
import atexit

SUPPORT_EMBED_TYPES = ["image", "text"]
logger = logging.getLogger(__name__)


class VllmEmbedding(MultiModalEmbedding):
    """
    Vllm LLM.

    This class runs a vLLM embedding model locally.
    """

    tensor_parallel_size: Optional[int] = Field(
        default=1,
        description="The number of GPUs to use for distributed execution with tensor parallelism.",
    )

    trust_remote_code: Optional[bool] = Field(
        default=True,
        description="Trust remote code (e.g., from HuggingFace) when downloading the model and tokenizer.",
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

    _client: Any = PrivateAttr()

    _image_token_id: Union[int, None] = PrivateAttr()

    def __init__(
        self,
        model_name: str = "facebook/opt-125m",
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        tensor_parallel_size: int = 1,
        trust_remote_code: bool = False,
        dtype: str = "auto",
        download_dir: Optional[str] = None,
        vllm_kwargs: Dict[str, Any] = {},
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        callback_manager = callback_manager or CallbackManager([])
        super().__init__(
            model_name=model_name,
            embed_batch_size=embed_batch_size,
            callback_manager=callback_manager,
        )
        try:
            from vllm import LLM as VLLModel
        except ImportError:
            raise ImportError(
                "Could not import vllm python package. "
                "Please install it with `pip install vllm`."
            )
        self._client = VLLModel(
            model=model_name,
            task="embed",
            max_num_seqs=embed_batch_size,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=trust_remote_code,
            dtype=dtype,
            download_dir=download_dir,
            **vllm_kwargs,
        )
        try:
            self._image_token_id = (
                self._client.llm_engine.model_config.hf_config.image_token_id
            )
        except AttributeError:
            self._image_token_id = None

    @classmethod
    def class_name(cls) -> str:
        return "VllmEmbedding"

    @atexit.register
    def close():
        import torch
        import gc

        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True,
    )
    def _embed_with_retry(
        self, inputs: List[Union[str, BytesIO]], embed_type: str = "text"
    ) -> List[List[float]]:
        """
        Generates embeddings with retry mechanism.

        Args:
            inputs: List of texts or images to embed

        Returns:
            List of embedding vectors

        Raises:
            Exception: If embedding fails after retries

        """
        try:
            if embed_type == "image":
                inputs = [
                    {
                        "prompt_token_ids": [self._image_token_id],
                        "multi_modal_data": {"image": x},
                    }
                    for x in inputs
                ]
            emb = self._client.embed(inputs)
            return [x.outputs.embedding for x in emb]
        except Exception as e:
            logger.warning(f"Embedding attempt failed: {e!s}")
            raise

    def _embed(
        self, inputs: List[Union[str, BytesIO]], embed_type: str = "text"
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
        if embed_type not in SUPPORT_EMBED_TYPES:
            raise (ValueError("Not Implemented"))
        return self._embed_with_retry(inputs, embed_type)

    def _get_query_embedding(self, query: str) -> List[float]:
        """
        Generates Embeddings for Query.

        Args:
            query (str): Query text/sentence

        Returns:
            List[float]: numpy array of embeddings

        """
        return self._embed([query])[0]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """
        Generates Embeddings for Query Asynchronously.

        Args:
            query (str): Query text/sentence

        Returns:
            List[float]: numpy array of embeddings

        """
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """
        Generates Embeddings for text Asynchronously.

        Args:
            text (str): Text/Sentence

        Returns:
            List[float]: numpy array of embeddings

        """
        return self._get_text_embedding(text)

    def _get_text_embedding(self, text: str) -> List[float]:
        """
        Generates Embeddings for text.

        Args:
            text (str): Text/sentences

        Returns:
            List[float]: numpy array of embeddings

        """
        return self._embed([text])[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generates Embeddings for text.

        Args:
            texts (List[str]): Texts / Sentences

        Returns:
            List[List[float]]: numpy array of embeddings

        """
        return self._embed(texts)

    def _get_image_embedding(self, img_file_path: ImageType) -> List[float]:
        """Generate embedding for an image."""
        image = Image.open(img_file_path)
        return self._embed([image], "image")[0]

    async def _aget_image_embedding(self, img_file_path: ImageType) -> List[float]:
        """Generate embedding for an image asynchronously."""
        return self._get_image_embedding(img_file_path)

    def _get_image_embeddings(
        self, img_file_paths: List[ImageType]
    ) -> List[List[float]]:
        images = [Image.open(x) for x in img_file_paths]
        """Generate embeddings for multiple images."""
        return self._embed(images, "image")

    async def _aget_image_embeddings(
        self, img_file_paths: List[ImageType]
    ) -> List[List[float]]:
        """Generate embeddings for multiple images asynchronously."""
        return self._get_image_embeddings(img_file_paths)
