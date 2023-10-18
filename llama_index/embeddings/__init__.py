"""Init file."""

from llama_index.embeddings.adapter import (
    AdapterEmbeddingModel,
    LinearAdapterEmbeddingModel,
)
from llama_index.embeddings.base import SimilarityMode
from llama_index.embeddings.clarifai import ClarifaiEmbedding
from llama_index.embeddings.elasticsearch import ElasticsearchEmbeddings
from llama_index.embeddings.google import GoogleUnivSentEncoderEmbedding
from llama_index.embeddings.gradient import GradientEmbedding
from llama_index.embeddings.huggingface import (
    HuggingFaceEmbedding,
    HuggingFaceInferenceAPIEmbeddings,
)
from llama_index.embeddings.huggingface_optimum import OptimumEmbedding
from llama_index.embeddings.huggingface_utils import DEFAULT_HUGGINGFACE_EMBEDDING_MODEL
from llama_index.embeddings.instructor import InstructorEmbedding
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.embeddings.llm_rails import LLMRailsEmbeddings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.pooling import Pooling
from llama_index.embeddings.text_embeddings_inference import TextEmbeddingsInference
from llama_index.embeddings.utils import resolve_embed_model

__all__ = [
    "AdapterEmbeddingModel",
    "ClarifaiEmbedding",
    "DEFAULT_HUGGINGFACE_EMBEDDING_MODEL",
    "ElasticsearchEmbeddings",
    "GoogleUnivSentEncoderEmbedding",
    "GradientEmbedding",
    "HuggingFaceInferenceAPIEmbeddings",
    "HuggingFaceEmbedding",
    "InstructorEmbedding",
    "LangchainEmbedding",
    "LinearAdapterEmbeddingModel",
    "LLMRailsEmbeddings",
    "OpenAIEmbedding",
    "OptimumEmbedding",
    "Pooling",
    "SimilarityMode",
    "TextEmbeddingsInference",
    "resolve_embed_model",
]
