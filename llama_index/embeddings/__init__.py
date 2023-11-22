"""Init file."""

from llama_index.embeddings.adapter import (
    AdapterEmbeddingModel,
    LinearAdapterEmbeddingModel,
)
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.embeddings.base import BaseEmbedding, SimilarityMode
from llama_index.embeddings.bedrock import BedrockEmbedding
from llama_index.embeddings.clarifai import ClarifaiEmbedding
from llama_index.embeddings.clip import ClipEmbedding
from llama_index.embeddings.cohereai import CohereEmbedding
from llama_index.embeddings.elasticsearch import (
    ElasticsearchEmbedding,
    ElasticsearchEmbeddings,
)
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.embeddings.google import GoogleUnivSentEncoderEmbedding
from llama_index.embeddings.google_palm import GooglePaLMEmbedding
from llama_index.embeddings.gradient import GradientEmbedding
from llama_index.embeddings.huggingface import (
    HuggingFaceEmbedding,
    HuggingFaceInferenceAPIEmbedding,
    HuggingFaceInferenceAPIEmbeddings,
)
from llama_index.embeddings.huggingface_optimum import OptimumEmbedding
from llama_index.embeddings.huggingface_utils import DEFAULT_HUGGINGFACE_EMBEDDING_MODEL
from llama_index.embeddings.instructor import InstructorEmbedding
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.embeddings.llm_rails import LLMRailsEmbedding, LLMRailsEmbeddings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.pooling import Pooling
from llama_index.embeddings.text_embeddings_inference import TextEmbeddingsInference
from llama_index.embeddings.utils import resolve_embed_model
from llama_index.embeddings.voyageai import VoyageEmbedding

__all__ = [
    "AdapterEmbeddingModel",
    "BedrockEmbedding",
    "ClarifaiEmbedding",
    "ClipEmbedding",
    "CohereEmbedding",
    "BaseEmbedding",
    "DEFAULT_HUGGINGFACE_EMBEDDING_MODEL",
    "ElasticsearchEmbedding",
    "FastEmbedEmbedding",
    "GoogleUnivSentEncoderEmbedding",
    "GradientEmbedding",
    "HuggingFaceInferenceAPIEmbedding",
    "HuggingFaceEmbedding",
    "InstructorEmbedding",
    "LangchainEmbedding",
    "LinearAdapterEmbeddingModel",
    "LLMRailsEmbedding",
    "OpenAIEmbedding",
    "AzureOpenAIEmbedding",
    "OptimumEmbedding",
    "Pooling",
    "GooglePaLMEmbedding",
    "SimilarityMode",
    "TextEmbeddingsInference",
    "resolve_embed_model",
    # Deprecated, kept for backwards compatibility
    "LLMRailsEmbeddings",
    "ElasticsearchEmbeddings",
    "HuggingFaceInferenceAPIEmbeddings",
    "VoyageEmbedding",
]
