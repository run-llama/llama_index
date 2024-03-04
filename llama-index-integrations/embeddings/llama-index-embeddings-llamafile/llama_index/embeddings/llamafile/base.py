from typing import Any, Dict, List, Optional

import requests

from llama_index.core.base.embeddings.base import BaseEmbedding, Embedding
from llama_index.core.bridge.pydantic import Field
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.constants import DEFAULT_EMBED_BATCH_SIZE


class LlamafileEmbedding(BaseEmbedding):
    """Class for llamafile embeddings.

    llamafile lets you distribute and run large language models with a
    single file.

    To get started, see: https://github.com/Mozilla-Ocho/llamafile

    To use this class, you will need to first:

    1. Download a llamafile.
    2. Make the downloaded file executable: `chmod +x path/to/model.llamafile`
    3. Start the llamafile in server mode with embeddings enabled:

        `./path/to/model.llamafile --server --nobrowser --embedding`

    """

    base_url: str = Field(
        description="base url of the llamafile server",
        default="http://localhost:8080"
    )

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        super().__init__(
            base_url=base_url,
            callback_manager=callback_manager,
        )

    @classmethod
    def class_name(cls) -> str:
        return "LlamafileEmbedding"

    def _get_query_embedding(self, query: str) -> Embedding:
        return self._get_text_embedding(query)

    async def _aget_query_embedding(self, query: str) -> Embedding:
        raise NotImplementedError("not yet implemented")

    def _get_text_embedding(self, text: str) -> Embedding:
        """
        Embed the input text synchronously.
        """
        request_body = {
            "content": text,
            # "model": self.model_name,
            # "options": self.ollama_additional_kwargs,
        }

        response = requests.post(
            url=f"{self.base_url}/embedding",
            headers={"Content-Type": "application/json"},
            json=request_body,
        )
        response.encoding = "utf-8"
        response.raise_for_status()

        return response.json()["embedding"]
