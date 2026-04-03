import warnings

warnings.warn(
    "llama-index-packs-sub-question-weaviate is deprecated and no longer maintained. "
    "It will not receive any further updates.",
    DeprecationWarning,
    stacklevel=2,
)

from llama_index.packs.sub_question_weaviate.base import WeaviateSubQuestionPack

__all__ = ["WeaviateSubQuestionPack"]
