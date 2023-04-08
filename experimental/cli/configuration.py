import os
from configparser import ConfigParser
from typing import Any
from llama_index.embeddings.openai import OpenAIEmbedding
from langchain import OpenAI
from llama_index.indices.base import BaseGPTIndex
from llama_index.embeddings.base import BaseEmbedding
from llama_index import GPTSimpleVectorIndex, ServiceContext, LLMPredictor
from llama_index.data_structs.data_structs_v2 import SimpleIndexDict


CONFIG_FILE_NAME = "config.ini"
JSON_INDEX_FILE_NAME = "index.json"
DEFAULT_CONFIG = {
    "store": {"type": "json"},
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


def load_index(root: str = ".") -> BaseGPTIndex[Any]:
    """Load existing index file"""
    config = load_config(root)
    service_context = _load_service_context(config)
    if config["store"]["type"] == "json":
        index_file = os.path.join(root, JSON_INDEX_FILE_NAME)
    else:
        raise KeyError(f"Unknown index.type {config['index']['type']}")
    if os.path.exists(index_file):
        return GPTSimpleVectorIndex.load_from_disk(
            index_file, service_context=service_context
        )
    else:
        return GPTSimpleVectorIndex(
            index_struct=SimpleIndexDict(), service_context=service_context
        )


def save_index(index: BaseGPTIndex[Any], root: str = ".") -> None:
    """Save index to file"""
    config = load_config(root)
    if config["store"]["type"] == "json":
        index_file = os.path.join(root, JSON_INDEX_FILE_NAME)
    else:
        raise KeyError(f"Unknown index.type {config['index']['type']}")
    index.save_to_disk(index_file)


def _load_service_context(config: ConfigParser) -> ServiceContext:
    """Internal function to load service context based on configuration"""
    embed_model = _load_embed_model(config)
    llm_predictor = _load_llm_predictor(config)
    return ServiceContext.from_defaults(
        llm_predictor=llm_predictor, embed_model=embed_model
    )


def _load_llm_predictor(config: ConfigParser) -> LLMPredictor:
    """Internal function to load LLM predictor based on configuration"""
    model_type = config["llm_predictor"]["type"].lower()
    if model_type == "default":
        return LLMPredictor()
    if model_type == "azure":
        engine = config["llm_predictor"]["engine"]
        return LLMPredictor(llm=OpenAI(engine=engine))
    else:
        raise KeyError("llm_predictor.type")


def _load_embed_model(config: ConfigParser) -> BaseEmbedding:
    """Internal function to load embedding model based on configuration"""
    model_type = config["embed_model"]["type"]
    if model_type == "default":
        return OpenAIEmbedding()
    else:
        raise KeyError("embed_model.type")
