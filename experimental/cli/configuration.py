import os
from configparser import ConfigParser, SectionProxy
from typing import Any, Type
from llama_index.embeddings.openai import OpenAIEmbedding
from langchain import OpenAI
from langchain.base_language import BaseLanguageModel
from llama_index.indices.base import BaseIndex
from llama_index.embeddings.base import BaseEmbedding
from llama_index import (
    VectorStoreIndex,
    SimpleKeywordTableIndex,
    ServiceContext,
    LLMPredictor,
)
from llama_index.indices.loading import load_index_from_storage
from llama_index.llm_predictor import StructuredLLMPredictor
from llama_index.storage.storage_context import StorageContext


CONFIG_FILE_NAME = "config.ini"
DEFAULT_PERSIST_DIR = "./storage"
DEFAULT_CONFIG = {
    "store": {"persist_dir": DEFAULT_PERSIST_DIR},
    "index": {"type": "default"},
    "embed_model": {"type": "default"},
    "llm_predictor": {"type": "default"},
}


def load_config(root: str = ".") -> ConfigParser:
    """Load configuration from file"""
    config = ConfigParser()
    config.read_dict(DEFAULT_CONFIG)
    config.read(os.path.join(root, CONFIG_FILE_NAME))
    return config


def save_config(config: ConfigParser, root: str = ".") -> None:
    """Load configuration to file"""
    with open(os.path.join(root, CONFIG_FILE_NAME), "w") as fd:
        config.write(fd)


def load_index(root: str = ".") -> BaseIndex[Any]:
    """Load existing index file"""
    config = load_config(root)
    service_context = _load_service_context(config)

    # Index type
    index_type: Type
    if config["index"]["type"] == "default" or config["index"]["type"] == "vector":
        index_type = VectorStoreIndex
    elif config["index"]["type"] == "keyword":
        index_type = SimpleKeywordTableIndex
    else:
        raise KeyError(f"Unknown index.type {config['index']['type']}")

    try:
        # try loading index
        storage_context = _load_storage_context(config)
        index = load_index_from_storage(storage_context)
    except ValueError:
        # build index
        storage_context = StorageContext.from_defaults()
        index = index_type(
            nodes=[], service_context=service_context, storage_context=storage_context
        )
    return index


def save_index(index: BaseIndex[Any], root: str = ".") -> None:
    """Save index to file"""
    config = load_config(root)
    persist_dir = config["store"]["persist_dir"]
    index.storage_context.persist(persist_dir=persist_dir)


def _load_service_context(config: ConfigParser) -> ServiceContext:
    """Internal function to load service context based on configuration"""
    embed_model = _load_embed_model(config)
    llm_predictor = _load_llm_predictor(config)
    return ServiceContext.from_defaults(
        llm_predictor=llm_predictor, embed_model=embed_model
    )


def _load_storage_context(config: ConfigParser) -> StorageContext:
    persist_dir = config["store"]["persist_dir"]
    return StorageContext.from_defaults(persist_dir=persist_dir)


def _load_llm_predictor(config: ConfigParser) -> LLMPredictor:
    """Internal function to load LLM predictor based on configuration"""
    model_type = config["llm_predictor"]["type"].lower()
    if model_type == "default":
        llm = _load_llm(config["llm_predictor"])
        return LLMPredictor(llm=llm)
    elif model_type == "structured":
        llm = _load_llm(config["llm_predictor"])
        return StructuredLLMPredictor(llm=llm)
    else:
        raise KeyError("llm_predictor.type")


def _load_llm(section: SectionProxy) -> BaseLanguageModel:
    if "engine" in section:
        return OpenAI(engine=section["engine"])
    else:
        return OpenAI()


def _load_embed_model(config: ConfigParser) -> BaseEmbedding:
    """Internal function to load embedding model based on configuration"""
    model_type = config["embed_model"]["type"]
    if model_type == "default":
        return OpenAIEmbedding()
    else:
        raise KeyError("embed_model.type")
