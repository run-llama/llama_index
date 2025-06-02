"""LlamaPack class."""

from typing import Any, Dict, List

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.llama_pack.base import BaseLlamaPack
from llama_index.core.schema import Document
from llama_index.llms.ollama import Ollama

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"


class OllamaQueryEnginePack(BaseLlamaPack):
    def __init__(
        self,
        model: str,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        documents: List[Document] = None,
    ) -> None:
        self._model = model
        self._base_url = base_url
        self.llm = Ollama(model=self._model, base_url=self._base_url)

        Settings.llm = self.llm
        Settings.embed_model = OllamaEmbedding(
            model_name=self._model, base_url=self._base_url
        )
        self.index = VectorStoreIndex.from_documents(documents)

    def get_modules(self) -> Dict[str, Any]:
        """Get modules."""
        return {"llm": self.llm, "index": self.index}

    def run(self, query_str: str, **kwargs: Any) -> Any:
        """Run the pipeline."""
        query_engine = self.index.as_query_engine(**kwargs)
        return query_engine.query(query_str)


class OllamaEmbedding(BaseEmbedding):
    """
    Class for Ollama embeddings.

    Args:
        model_name (str): Model for embedding.

        base_url (str): Ollama url. Defaults to http://localhost:11434.

    """

    _base_url: str = PrivateAttr()
    _verbose: bool = PrivateAttr()

    def __init__(
        self,
        model_name: str,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_name=model_name,
            **kwargs,
        )

        self._verbose = verbose
        self._base_url = base_url

    @classmethod
    def class_name(cls) -> str:
        return "OllamaEmbedding"

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        return self.get_general_text_embedding(query)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """The asynchronous version of _get_query_embedding."""
        return self.get_general_text_embedding(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        return self.get_general_text_embedding(text)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Asynchronously get text embedding."""
        return self.get_general_text_embedding(text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings."""
        embeddings_list: List[List[float]] = []
        for text in texts:
            embeddings = self.get_general_text_embedding(text)
            embeddings_list.append(embeddings)

        return embeddings_list

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Asynchronously get text embeddings."""
        return self._get_text_embeddings(texts)

    def get_general_text_embedding(self, input: str) -> List[float]:
        """Get Ollama embedding."""
        try:
            import requests
        except ImportError:
            raise ImportError(
                "Could not import requests library."
                "Please install requests with `pip install requests`"
            )
        # all_kwargs = self._get_all_kwargs()
        response = requests.post(
            url=f"{self._base_url}/api/embeddings",
            headers={"Content-Type": "application/json"},
            json={"prompt": input, "model": self.model_name},
        )
        response.encoding = "utf-8"
        if response.status_code != 200:
            optional_detail = response.json().get("error")
            raise ValueError(
                f"Ollama call failed with status code {response.status_code}."
                f" Details: {optional_detail}"
            )

        try:
            embeddings = response.json()["embedding"]
            if self._verbose:
                print(f"Text={input}")
                print(embeddings)
            return embeddings
        except requests.exceptions.JSONDecodeError as e:
            raise ValueError(
                f"Error raised for Ollama Call: {e}.\nResponse: {response.text}"
            )
