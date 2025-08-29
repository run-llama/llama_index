from typing import Any, Dict, List, Optional

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.constants import DEFAULT_EMBED_BATCH_SIZE
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.bridge.pydantic import PrivateAttr

import random
import numpy as np
import tritonclient.http as triton
import tritonclient.http.aio as async_triton

DEFAULT_INPUT_TENSOR_NAME = "INPUT_TEXT"
DEFAULT_OUTPUT_TENSOR_NAME = "OUTPUT_EMBEDDINGS"


class NvidiaTritonEmbedding(BaseEmbedding):
    """
    Nvidia Triton Embedding.

    This connector allows for llama_index to interact with embedding models hosted on a Triton
    inference server over HTTP.

    [Triton Inference Server Github](https://github.com/triton-inference-server/server)

    Examples:
        `pip install llama-index-embeddings-nvidia-triton`

        ```python
        from llama_index.embeddings.nvidia_triton import NvidiaTritonEmbedding

        # Ensure a Triton server instance is running and provide the correct HTTP URL for your Triton server instance
        triton_url = "localhost:8000"

        # Instantiate the NvidiaTritonEmbedding class
        emb_client = NvidiaTritonEmbedding(
            server_url=triton_url,
            model_name="text_embeddings",
        )

        # Get a text embedding
        embedding = emb_client.get_text_embedding("hello world")
        print(f"Embedding for 'hello world': {embedding}")
        print(f"Embedding length: {len(embedding)}")
        ```

    """

    _input_tensor_name: str = PrivateAttr()
    _output_tensor_name: str = PrivateAttr()
    _url: str = PrivateAttr()
    _client_kwargs: Dict[str, Any] = PrivateAttr()

    def __init__(
        self,
        model_name: str,
        server_url: str = "localhost:8000",
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        input_tensor_name: str = DEFAULT_INPUT_TENSOR_NAME,
        output_tensor_name: str = DEFAULT_OUTPUT_TENSOR_NAME,
        callback_manager: Optional[CallbackManager] = None,
        client_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_name=model_name,
            embed_batch_size=embed_batch_size,
            callback_manager=callback_manager,  # type: ignore
            **kwargs,
        )

        self._url = server_url
        self._client_kwargs = client_kwargs or {}
        self._input_tensor_name = input_tensor_name
        self._output_tensor_name = output_tensor_name

    @classmethod
    def class_name(cls) -> str:
        return "NvidiaTritonEmbedding"

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        return self.get_general_text_embeddings([query])[0]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """The asynchronous version of _get_query_embedding."""
        embs = await self.aget_general_text_embeddings([query])
        return embs[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        return self.get_general_text_embeddings([text])[0]

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Asynchronously get text embedding."""
        embs = await self.aget_general_text_embeddings([text])
        return embs[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings."""
        return self.get_general_text_embeddings(texts)

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Asynchronously get text embeddings."""
        return await self.aget_general_text_embeddings(texts)

    def get_general_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get Triton embedding."""
        input_data = triton.InferInput(self._input_tensor_name, [len(texts)], "BYTES")
        input_data.set_data_from_numpy(np.array(texts, dtype=np.object_))
        output_data = triton.InferRequestedOutput(self._output_tensor_name)
        request_id = str(random.randint(1, 9999999))  # nosec

        client = triton.InferenceServerClient(
            url=self._url,
            **self._client_kwargs,
        )

        response = client.infer(
            model_name=self.model_name,
            inputs=[input_data],
            outputs=[output_data],
            request_id=request_id,
        )

        client.close()

        embeddings = response.as_numpy(self._output_tensor_name)
        if embeddings is None:
            raise ValueError("No embeddings returned from Triton server.")
        return [e.tolist() for e in embeddings]

    async def aget_general_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Asynchronously get Triton embedding."""
        input_data = async_triton.InferInput(
            self._input_tensor_name, [len(texts)], "BYTES"
        )
        input_data.set_data_from_numpy(np.array(texts, dtype=np.object_))
        output_data = async_triton.InferRequestedOutput(self._output_tensor_name)
        request_id = str(random.randint(1, 9999999))  # nosec

        aclient = async_triton.InferenceServerClient(
            url=self._url,
            **self._client_kwargs,
        )

        response = await aclient.infer(
            model_name=self.model_name,
            inputs=[input_data],
            outputs=[output_data],
            request_id=request_id,
        )

        await aclient.close()

        embeddings = response.as_numpy(self._output_tensor_name)
        if embeddings is None:
            raise ValueError("No embeddings returned from Triton server.")
        return [e.tolist() for e in embeddings]
