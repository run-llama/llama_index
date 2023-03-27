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
    llm_predictor: LLMPredictor
    prompt_helper: PromptHelper
    embed_model: BaseEmbedding
    node_parser: NodeParser
    llama_logger: LlamaLogger
    chunk_size_limit: Optional[int] = None

    @classmethod
    def from_defaults(
        cls,
        llm_predictor: Optional[LLMPredictor] = None,
        prompt_helper: Optional[PromptHelper] = None,
        embed_model: Optional[BaseEmbedding] = None,
        node_parser: Optional[NodeParser] = None,
        llama_logger: Optional[LlamaLogger] = None,
        chunk_size_limit: Optional[int] = None,
    ) -> "ServiceContext":
        llm_predictor = llm_predictor or LLMPredictor()
        # NOTE: the embed_model isn't used in all indices
        embed_model = embed_model or OpenAIEmbedding()
        prompt_helper = prompt_helper or PromptHelper.from_llm_predictor(
            llm_predictor, chunk_size_limit=chunk_size_limit
        )
        node_parser = node_parser or SimpleNodeParser()
        llama_logger = llama_logger or LlamaLogger()

        return cls(
            llm_predictor=llm_predictor,
            embed_model=embed_model,
            prompt_helper=prompt_helper,
            node_parser=node_parser,
            llama_logger=llama_logger,
            chunk_size_limit=chunk_size_limit,
        )
