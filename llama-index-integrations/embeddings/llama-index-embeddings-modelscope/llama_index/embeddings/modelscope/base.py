import logging
from typing import Any, Optional, List
from llama_index.core.base.embeddings.base import (
    BaseEmbedding,
    Embedding,
)
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.types import PydanticProgramMode
from llama_index.embeddings.modelscope.utils import (
    sentence_to_input,
    sentences_to_input,
    output_to_embedding,
    outputs_to_embeddings,
)

from modelscope.pipelines import pipeline as pipeline_builder

DEFAULT_MODELSCOPE_MODEL = "iic/nlp_gte_sentence-embedding_chinese-base"
DEFAULT_MODELSCOPE_MODEL_REVISION = "master"
DEFAULT_MODELSCOPE_TASK = "sentence-embedding"
DEFAULT_MODELSCOPE_SEQUENCE_LENGTH = 128
logger = logging.getLogger(__name__)


class ModelScopeEmbedding(BaseEmbedding):
    """ModelScope Embedding."""

    model_name: str = Field(
        default=DEFAULT_MODELSCOPE_MODEL,
        description=(
            "The model name to use from ModelScope. "
            "Unused if `model` is passed in directly."
        ),
    )
    model_revision: str = Field(
        default=DEFAULT_MODELSCOPE_MODEL_REVISION,
        description=(
            "The model revision to use from ModelScope. "
            "Unused if `model` is passed in directly."
        ),
    )
    task_name: str = Field(
        default=DEFAULT_MODELSCOPE_TASK,
        description=(
            "The ModelScope task type, for embedding use default sentence_embedding."
        ),
    )
    sequence_length: int = Field(
        default=128,
        description="The maximum length of the input sequence. Defaults to 128.",
    )
    model_kwargs: dict = Field(
        default_factory=dict,
        description="The kwargs to pass to the model during initialization.",
    )
    generate_kwargs: dict = Field(
        default_factory=dict,
        description="The kwargs to pass to the model during generation.",
    )

    _pipeline: Any = PrivateAttr()

    def __init__(
        self,
        model_name: str = DEFAULT_MODELSCOPE_MODEL,
        model_revision: str = DEFAULT_MODELSCOPE_MODEL_REVISION,
        task_name: str = DEFAULT_MODELSCOPE_TASK,
        sequence_length: int = DEFAULT_MODELSCOPE_SEQUENCE_LENGTH,
        model: Optional[Any] = None,
        model_kwargs: Optional[dict] = None,
        generate_kwargs: Optional[dict] = None,
        pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
    ) -> None:
        """Initialize params."""
        model_kwargs = model_kwargs or {}
        if model:
            pipeline = model
        else:
            pipeline = pipeline_builder(
                task=task_name,
                model=model_name,
                model_revision=model_revision,
                sequence_length=sequence_length,
            )

        super().__init__(
            model_kwargs=model_kwargs or {},
            generate_kwargs=generate_kwargs or {},
            pydantic_program_mode=pydantic_program_mode,
        )
        self._pipeline = pipeline

    def _get_query_embedding(self, query: str) -> Embedding:
        """Get the embedding for a query."""
        return output_to_embedding(self._pipeline(sentence_to_input(query)))

    async def _aget_query_embedding(self, query: str) -> Embedding:
        """Get the embedding for a query."""
        return output_to_embedding(self._pipeline(sentence_to_input(query)))

    def _get_text_embedding(self, text: str) -> Embedding:
        """Get the embedding for a text."""
        return output_to_embedding(self._pipeline(sentence_to_input(text)))

    def _get_text_embeddings(self, texts: List[str]) -> List[Embedding]:
        """Get the embeddings for a list of texts."""
        return outputs_to_embeddings(self._pipeline(sentences_to_input(texts)))
