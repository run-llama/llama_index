import dataclasses
import logging
from dataclasses import dataclass
from typing import Optional

import llama_index
from llama_index.callbacks.base import CallbackManager
from llama_index.constants import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from llama_index.embeddings.base import BaseEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.indices.prompt_helper import PromptHelper
from llama_index.langchain_helpers.chain_wrapper import LLMPredictor
from llama_index.langchain_helpers.text_splitter import TokenTextSplitter
from llama_index.llm_predictor.base import BaseLLMPredictor, LLMMetadata
from llama_index.logger import LlamaLogger
from llama_index.node_parser.interface import NodeParser
from llama_index.node_parser.simple import SimpleNodeParser
from langchain.base_language import BaseLanguageModel


logger = logging.getLogger(__name__)


def _get_default_node_parser(
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    callback_manager: Optional[CallbackManager] = None,
) -> NodeParser:
    """Get default node parser."""
    callback_manager = callback_manager or CallbackManager([])
    chunk_size = chunk_size or DEFAULT_CHUNK_SIZE
    chunk_overlap = chunk_overlap or DEFAULT_CHUNK_OVERLAP

    token_text_splitter = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        callback_manager=callback_manager,
    )
    return SimpleNodeParser(
        text_splitter=token_text_splitter, callback_manager=callback_manager
    )


def _get_default_prompt_helper(
    llm_metadata: LLMMetadata,
    context_window: Optional[int] = None,
    num_output: Optional[int] = None,
) -> PromptHelper:
    """Get default prompt helper."""
    if context_window is not None:
        llm_metadata = dataclasses.replace(llm_metadata, context_window=context_window)
    if num_output is not None:
        llm_metadata = dataclasses.replace(llm_metadata, num_output=num_output)
    return PromptHelper.from_llm_metadata(llm_metadata=llm_metadata)


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

    """

    llm_predictor: BaseLLMPredictor
    prompt_helper: PromptHelper
    embed_model: BaseEmbedding
    node_parser: NodeParser
    llama_logger: LlamaLogger
    callback_manager: CallbackManager

    @classmethod
    def from_defaults(
        cls,
        llm_predictor: Optional[BaseLLMPredictor] = None,
        llm: Optional[BaseLanguageModel] = None,
        prompt_helper: Optional[PromptHelper] = None,
        embed_model: Optional[BaseEmbedding] = None,
        node_parser: Optional[NodeParser] = None,
        llama_logger: Optional[LlamaLogger] = None,
        callback_manager: Optional[CallbackManager] = None,
        # node parser kwargs
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        # prompt helper kwargs
        context_window: Optional[int] = None,
        num_output: Optional[int] = None,
        # deprecated kwargs
        chunk_size_limit: Optional[int] = None,
    ) -> "ServiceContext":
        """Create a ServiceContext from defaults.
        If an argument is specified, then use the argument value provided for that
        parameter. If an argument is not specified, then use the default value.

        You can change the base defaults by setting llama_index.global_service_context
        to a ServiceContext object with your desired settings.

        Args:
            llm_predictor (Optional[BaseLLMPredictor]): LLMPredictor
            prompt_helper (Optional[PromptHelper]): PromptHelper
            embed_model (Optional[BaseEmbedding]): BaseEmbedding
            node_parser (Optional[NodeParser]): NodeParser
            llama_logger (Optional[LlamaLogger]): LlamaLogger (deprecated)
            chunk_size (Optional[int]): chunk_size
            callback_manager (Optional[CallbackManager]): CallbackManager

        Deprecated Args:
            chunk_size_limit (Optional[int]): renamed to chunk_size

        """
        if chunk_size_limit is not None and chunk_size is None:
            logger.warning(
                "chunk_size_limit is deprecated, please specify chunk_size instead"
            )
            chunk_size = chunk_size_limit

        if llama_index.global_service_context is not None:
            return cls.from_service_context(
                llama_index.global_service_context,
                llm_predictor=llm_predictor,
                prompt_helper=prompt_helper,
                embed_model=embed_model,
                node_parser=node_parser,
                llama_logger=llama_logger,
                callback_manager=callback_manager,
                chunk_size=chunk_size,
                chunk_size_limit=chunk_size_limit,
            )

        callback_manager = callback_manager or CallbackManager([])
        if llm is not None:
            if llm_predictor is not None:
                raise ValueError("Cannot specify both llm and llm_predictor")
            llm_predictor = LLMPredictor(llm=llm)
        llm_predictor = llm_predictor or LLMPredictor()
        llm_predictor.callback_manager = callback_manager

        # NOTE: the embed_model isn't used in all indices
        embed_model = embed_model or OpenAIEmbedding()
        embed_model.callback_manager = callback_manager

        prompt_helper = prompt_helper or _get_default_prompt_helper(
            llm_metadata=llm_predictor.get_llm_metadata(),
            context_window=context_window,
            num_output=num_output,
        )

        node_parser = node_parser or _get_default_node_parser(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            callback_manager=callback_manager,
        )

        llama_logger = llama_logger or LlamaLogger()

        return cls(
            llm_predictor=llm_predictor,
            embed_model=embed_model,
            prompt_helper=prompt_helper,
            node_parser=node_parser,
            llama_logger=llama_logger,  # deprecated
            callback_manager=callback_manager,
        )

    @classmethod
    def from_service_context(
        cls,
        service_context: "ServiceContext",
        llm_predictor: Optional[BaseLLMPredictor] = None,
        llm: Optional[BaseLanguageModel] = None,
        prompt_helper: Optional[PromptHelper] = None,
        embed_model: Optional[BaseEmbedding] = None,
        node_parser: Optional[NodeParser] = None,
        llama_logger: Optional[LlamaLogger] = None,
        callback_manager: Optional[CallbackManager] = None,
        # node parser kwargs
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        # prompt helper kwargs
        context_window: Optional[int] = None,
        num_output: Optional[int] = None,
        # deprecated kwargs
        chunk_size_limit: Optional[int] = None,
    ) -> "ServiceContext":
        """Instantiate a new service context using a previous as the defaults."""
        if chunk_size_limit is not None and chunk_size is None:
            logger.warning(
                "chunk_size_limit is deprecated, please specify chunk_size",
                DeprecationWarning,
            )
            chunk_size = chunk_size_limit

        callback_manager = callback_manager or service_context.callback_manager
        if llm is not None:
            if llm_predictor is not None:
                raise ValueError("Cannot specify both llm and llm_predictor")
            llm_predictor = LLMPredictor(llm=llm)

        llm_predictor = llm_predictor or service_context.llm_predictor
        llm_predictor.callback_manager = callback_manager

        # NOTE: the embed_model isn't used in all indices
        embed_model = embed_model or service_context.embed_model
        embed_model.callback_manager = callback_manager

        prompt_helper = prompt_helper or _get_default_prompt_helper(
            llm_metadata=llm_predictor.get_llm_metadata(),
            context_window=context_window,
            num_output=num_output,
        )

        node_parser = node_parser or service_context.node_parser
        if chunk_size is not None or chunk_overlap is not None:
            node_parser = _get_default_node_parser(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                callback_manager=callback_manager,
            )

        llama_logger = llama_logger or service_context.llama_logger

        return cls(
            llm_predictor=llm_predictor,
            embed_model=embed_model,
            prompt_helper=prompt_helper,
            node_parser=node_parser,
            llama_logger=llama_logger,  # deprecated
            callback_manager=callback_manager,
        )


def set_global_service_context(service_context: Optional[ServiceContext]) -> None:
    """Helper function to set the global service context."""
    llama_index.global_service_context = service_context
