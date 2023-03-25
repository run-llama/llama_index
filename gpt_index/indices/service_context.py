
from dataclasses import dataclass
from typing import Optional

from gpt_index.embeddings.base import BaseEmbedding
from gpt_index.embeddings.openai import OpenAIEmbedding
from gpt_index.indices.prompt_helper import PromptHelper
from gpt_index.langchain_helpers.chain_wrapper import LLMPredictor
from gpt_index.logger import LlamaLogger
from gpt_index.node_parser.interface import NodeParser
from gpt_index.node_parser.simple import SimpleNodeParser


@dataclass
class ServiceContext:
    llm_predictor: Optional[LLMPredictor] = None,
    prompt_helper: Optional[PromptHelper] = None,
    embed_model: Optional[BaseEmbedding] = None,
    node_parser: Optional[NodeParser] = None,
    chunk_size_limit: Optional[int] = None,
    llama_logger: Optional[LlamaLogger] = None,

    def __post_init__(self) -> None:
        self.llm_predictor = self.llm_predictor or LLMPredictor()
        # NOTE: the embed_model isn't used in all indices
        self.embed_model = self.embed_model or OpenAIEmbedding()
        self.prompt_helper = self.prompt_helper or PromptHelper.from_llm_predictor(
            self.llm_predictor, chunk_size_limit=self.chunk_size_limit
        )
        self.node_parser = self.node_parser or SimpleNodeParser()
        self.llama_logger = self.llama_logger or LlamaLogger()


