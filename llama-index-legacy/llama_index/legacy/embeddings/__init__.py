"""Init file."""

from llama_index.legacy.embeddings.adapter import (
    AdapterEmbeddingModel,
    LinearAdapterEmbeddingModel,
)
from llama_index.legacy.embeddings.anyscale import AnyscaleEmbedding
from llama_index.legacy.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.legacy.embeddings.base import BaseEmbedding, SimilarityMode
from llama_index.legacy.embeddings.bedrock import BedrockEmbedding
from llama_index.legacy.embeddings.clarifai import ClarifaiEmbedding
from llama_index.legacy.embeddings.clip import ClipEmbedding
from llama_index.legacy.embeddings.cohereai import CohereEmbedding
from llama_index.legacy.embeddings.dashscope import (
    DashScopeBatchTextEmbeddingModels,
    DashScopeEmbedding,
    DashScopeMultiModalEmbeddingModels,
    DashScopeTextEmbeddingModels,
    DashScopeTextEmbeddingType,
)
from llama_index.legacy.embeddings.elasticsearch import (
    ElasticsearchEmbedding,
    ElasticsearchEmbeddings,
)
from llama_index.legacy.embeddings.fastembed import FastEmbedEmbedding
from llama_index.legacy.embeddings.gemini import GeminiEmbedding
from llama_index.legacy.embeddings.google import GoogleUnivSentEncoderEmbedding
from llama_index.legacy.embeddings.google_palm import GooglePaLMEmbedding
from llama_index.legacy.embeddings.gradient import GradientEmbedding
from llama_index.legacy.embeddings.huggingface import (
    HuggingFaceEmbedding,
    HuggingFaceInferenceAPIEmbedding,
    HuggingFaceInferenceAPIEmbeddings,
)
from llama_index.legacy.embeddings.huggingface_optimum import OptimumEmbedding
from llama_index.legacy.embeddings.huggingface_utils import (
    DEFAULT_HUGGINGFACE_EMBEDDING_MODEL,
)
from llama_index.legacy.embeddings.instructor import InstructorEmbedding
from llama_index.legacy.embeddings.langchain import LangchainEmbedding
from llama_index.legacy.embeddings.llm_rails import (
    LLMRailsEmbedding,
    LLMRailsEmbeddings,
)
from llama_index.legacy.embeddings.mistralai import MistralAIEmbedding
from llama_index.legacy.embeddings.nomic import NomicEmbedding
from llama_index.legacy.embeddings.ollama_embedding import OllamaEmbedding
from llama_index.legacy.embeddings.openai import OpenAIEmbedding
from llama_index.legacy.embeddings.pooling import Pooling
from llama_index.legacy.embeddings.sagemaker_embedding_endpoint import (
    SageMakerEmbedding,
)
from llama_index.legacy.embeddings.text_embeddings_inference import (
    TextEmbeddingsInference,
)
from llama_index.legacy.embeddings.together import TogetherEmbedding
from llama_index.legacy.embeddings.utils import resolve_embed_model
from llama_index.legacy.embeddings.voyageai import VoyageEmbedding

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
    "MistralAIEmbedding",
    "OpenAIEmbedding",
    "AzureOpenAIEmbedding",
    "AnyscaleEmbedding",
    "OptimumEmbedding",
    "Pooling",
    "SageMakerEmbedding",
    "GooglePaLMEmbedding",
    "SimilarityMode",
    "TextEmbeddingsInference",
    "TogetherEmbedding",
    "resolve_embed_model",
    "NomicEmbedding",
    # Deprecated, kept for backwards compatibility
    "LLMRailsEmbeddings",
    "ElasticsearchEmbeddings",
    "HuggingFaceInferenceAPIEmbeddings",
    "VoyageEmbedding",
    "OllamaEmbedding",
    "GeminiEmbedding",
    "DashScopeEmbedding",
    "DashScopeTextEmbeddingModels",
    "DashScopeTextEmbeddingType",
    "DashScopeBatchTextEmbeddingModels",
    "DashScopeMultiModalEmbeddingModels",
]
