from dataclasses import dataclass
from typing import Optional

from llama_index.callbacks.base import CallbackManager
from llama_index.embeddings.base import BaseEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.indices.prompt_helper import PromptHelper
from llama_index.langchain_helpers.chain_wrapper import LLMPredictor
from llama_index.langchain_helpers.text_splitter import TokenTextSplitter
from llama_index.llm_predictor.base import BaseLLMPredictor
from llama_index.logger import LlamaLogger
from llama_index.node_parser.interface import NodeParser
from llama_index.node_parser.simple import SimpleNodeParser


def _get_default_node_parser(
    chunk_size_limit: Optional[int] = None,
    callback_manager: Optional[CallbackManager] = None,
) -> NodeParser:
    """Get default node parser."""
    callback_manager = callback_manager or CallbackManager([])
    if chunk_size_limit is None:
        token_text_splitter = TokenTextSplitter(
            callback_manager=callback_manager
        )  # use default chunk size
    else:
        token_text_splitter = TokenTextSplitter(
            chunk_size=chunk_size_limit, callback_manager=callback_manager
        )
    return SimpleNodeParser(
        text_splitter=token_text_splitter, callback_manager=callback_manager
    )


@dataclass
class ServiceContext:
    """Service Context container.

    The service context container is a utility container for LlamaIndex
    index and query classes. It contains the following:
    - llm_predictor: BaseLLMPredictor
    - prompt_helper: PromptHelper
    - embed_model: BaseEmbedding
    - node_parser: NodeParser
    - llama_logger: LlamaLogger (deprecated)
    - callback_manager: CallbackManager
    - chunk_size_limit: chunk size limit

    """

    llm_predictor: BaseLLMPredictor
    prompt_helper: PromptHelper
    embed_model: BaseEmbedding
    node_parser: NodeParser
    llama_logger: LlamaLogger
    callback_manager: CallbackManager
    chunk_size_limit: Optional[int] = None

    @classmethod
    def from_defaults(
        cls,
        llm_predictor: Optional[BaseLLMPredictor] = None,
        prompt_helper: Optional[PromptHelper] = None,
        embed_model: Optional[BaseEmbedding] = None,
        node_parser: Optional[NodeParser] = None,
        llama_logger: Optional[LlamaLogger] = None,
        callback_manager: Optional[CallbackManager] = None,
        chunk_size_limit: Optional[int] = None,
    ) -> "ServiceContext":
        """Create a ServiceContext from defaults.
        If an argument is specified, then use the argument value provided for that
        parameter. If an argument is not specified, then use the default value.

        Args:
            llm_predictor (Optional[BaseLLMPredictor]): LLMPredictor
            prompt_helper (Optional[PromptHelper]): PromptHelper
            embed_model (Optional[BaseEmbedding]): BaseEmbedding
            node_parser (Optional[NodeParser]): NodeParser
            llama_logger (Optional[LlamaLogger]): LlamaLogger (deprecated)
            chunk_size_limit (Optional[int]): chunk_size_limit

        """
        callback_manager = callback_manager or CallbackManager([])
        llm_predictor = llm_predictor or LLMPredictor()
        llm_predictor.callback_manager = callback_manager

        # NOTE: the embed_model isn't used in all indices
        embed_model = embed_model or OpenAIEmbedding()
        embed_model.callback_manager = callback_manager

        prompt_helper = prompt_helper or PromptHelper.from_llm_predictor(
            llm_predictor, chunk_size_limit=chunk_size_limit
        )

        node_parser = node_parser or _get_default_node_parser(
            chunk_size_limit=chunk_size_limit, callback_manager=callback_manager
        )

        llama_logger = llama_logger or LlamaLogger()

        return cls(
            llm_predictor=llm_predictor,
            embed_model=embed_model,
            prompt_helper=prompt_helper,
            node_parser=node_parser,
            llama_logger=llama_logger,  # deprecated
            callback_manager=callback_manager,
            chunk_size_limit=chunk_size_limit,
        )
