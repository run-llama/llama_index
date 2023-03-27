from dataclasses import dataclass
from typing import Optional

from gpt_index.embeddings.base import BaseEmbedding
from gpt_index.embeddings.openai import OpenAIEmbedding
from gpt_index.indices.prompt_helper import PromptHelper
from gpt_index.langchain_helpers.chain_wrapper import LLMPredictor
from gpt_index.langchain_helpers.text_splitter import TokenTextSplitter
from gpt_index.logger import LlamaLogger
from gpt_index.node_parser.interface import NodeParser
from gpt_index.node_parser.simple import SimpleNodeParser


def _get_default_node_parser(chunk_size_limit: Optional[int] = None) -> NodeParser:
    """Get default node parser."""
    if chunk_size_limit is None:
        # use default chunk size (3900)
        token_text_splitter = TokenTextSplitter()
    else:
        token_text_splitter = TokenTextSplitter(chunk_size=chunk_size_limit)
    return SimpleNodeParser(text_splitter=token_text_splitter)


@dataclass
class ServiceContext:
    """Service Context container.

    The service context container is a utility container for LlamaIndex
    index and query classes. It contains the following:
    - llm_predictor: LLMPredictor
    - prompt_helper: PromptHelper
    - embed_model: BaseEmbedding
    - node_parser: NodeParser
    - llama_logger: LlamaLogger
    - chunk_size_limit: chunk size limit

    """

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
        """Create a ServiceContext from defaults.
        If an argument is specified, then use the argument value provided for that
        parameter. If an argument is not specified, then use the default value.

        Args:
            llm_predictor (Optional[LLMPredictor]): LLMPredictor
            prompt_helper (Optional[PromptHelper]): PromptHelper
            embed_model (Optional[BaseEmbedding]): BaseEmbedding
            node_parser (Optional[NodeParser]): NodeParser
            llama_logger (Optional[LlamaLogger]): LlamaLogger
            chunk_size_limit (Optional[int]): chunk_size_limit

        """
        llm_predictor = llm_predictor or LLMPredictor()
        # NOTE: the embed_model isn't used in all indices
        embed_model = embed_model or OpenAIEmbedding()
        prompt_helper = prompt_helper or PromptHelper.from_llm_predictor(
            llm_predictor, chunk_size_limit=chunk_size_limit
        )
        node_parser = node_parser or _get_default_node_parser(
            chunk_size_limit=chunk_size_limit
        )
        llama_logger = llama_logger or LlamaLogger()

        return cls(
            llm_predictor=llm_predictor,
            embed_model=embed_model,
            prompt_helper=prompt_helper,
            node_parser=node_parser,
            llama_logger=llama_logger,
            chunk_size_limit=chunk_size_limit,
        )
