from typing import Any, List, Optional

from llama_index.bridge.pydantic import Field, PrivateAttr
from llama_index.callbacks import CallbackManager
from llama_index.embeddings.base import DEFAULT_EMBED_BATCH_SIZE, BaseEmbedding
from llama_index.embeddings.huggingface_utils import (
    DEFAULT_INSTRUCT_MODEL,
    get_query_instruct_for_model_name,
    get_text_instruct_for_model_name,
)


class InstructorEmbedding(BaseEmbedding):
    query_instruction: Optional[str] = Field(
        description="Instruction to prepend to query text."
    )
    text_instruction: Optional[str] = Field(
        description="Instruction to prepend to text."
    )
    cache_folder: Optional[str] = Field(
        description="Cache folder for huggingface files."
    )

    _model: Any = PrivateAttr()

    def __init__(
        self,
        model_name: str = DEFAULT_INSTRUCT_MODEL,
        query_instruction: Optional[str] = None,
        text_instruction: Optional[str] = None,
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        cache_folder: Optional[str] = None,
        device: Optional[str] = None,
        callback_manager: Optional[CallbackManager] = None,
    ):
        try:
            from InstructorEmbedding import INSTRUCTOR
        except ImportError:
            raise ImportError(
                "InstructorEmbedding requires instructor to be installed.\n"
                "Please install transformers with `pip install InstructorEmbedding`."
            )
        self._model = INSTRUCTOR(model_name, cache_folder=cache_folder, device=device)

        super().__init__(
            embed_batch_size=embed_batch_size,
            callback_manager=callback_manager,
            model_name=model_name,
            query_instruction=query_instruction,
            text_instruction=text_instruction,
            cache_folder=cache_folder,
        )

    @classmethod
    def class_name(cls) -> str:
        return "InstructorEmbedding"

    def _format_query_text(self, query_text: str) -> List[str]:
        """Format query text."""
        instruction = self.text_instruction

        if instruction is None:
            instruction = get_query_instruct_for_model_name(self.model_name)

        return [instruction, query_text]

    def _format_text(self, text: str) -> List[str]:
        """Format text."""
        instruction = self.text_instruction

        if instruction is None:
            instruction = get_text_instruct_for_model_name(self.model_name)

        return [instruction, text]

    def _embed(self, instruct_sentence_pairs: List[List[str]]) -> List[List[float]]:
        """Embed sentences."""
        return self._model.encode(instruct_sentence_pairs)

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        query_pair = self._format_query_text(query)
        return self._embed([query_pair])[0]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Get query embedding async."""
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Get text embedding async."""
        return self._get_text_embedding(text)

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        text_pair = self._format_text(text)
        return self._embed([text_pair])[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings."""
        text_pairs = [self._format_text(text) for text in texts]
        return self._embed(text_pairs)
