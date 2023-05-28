import importlib
import logging
import os
from pathlib import Path
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


logger = logging.getLogger(__name__)
global_service_context = None


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
        is_global: Optional[bool] = False,
    ) -> "ServiceContext":
        """Create a ServiceContext from defaults.
        If an argument is specified, then use the argument value provided for that
        parameter. If an argument is not specified, then use the default value.

        is_global is used to specify if the from_defaults call should set the global
        service context. If a global service context is set, then the values set on
        the global service context will be used for the attributes not provided in
        the from_defaults call.

        Args:
            llm_predictor (Optional[BaseLLMPredictor]): LLMPredictor
            prompt_helper (Optional[PromptHelper]): PromptHelper
            embed_model (Optional[BaseEmbedding]): BaseEmbedding
            node_parser (Optional[NodeParser]): NodeParser
            llama_logger (Optional[LlamaLogger]): LlamaLogger (deprecated)
            chunk_size_limit (Optional[int]): chunk_size_limit
            callback_manager (Optional[CallbackManager]): CallbackManager
            is_global: Optional[bool]:
                Specifies if the arguments should create a global service context

        """
        global global_service_context

        service_context_file = os.environ.get("LLAMA_SERVICE_CONTEXT", "")
        if (
            not is_global
            and global_service_context is None
            and service_context_file
            and os.path.exists(service_context_file)
        ):
            # dynamically load module from string path
            module_name = Path(service_context_file).stem
            spec = importlib.util.spec_from_file_location(
                module_name, service_context_file
            )

            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                if hasattr(module, "service_context"):
                    global_service_context = module.service_context
                    # return global_service_context
                else:
                    logger.warning(
                        "LLAMA_SERVICE_CONTEXT does not point to a valid python file "
                        "that defines a service_context object. Continuing without it."
                    )
            else:
                logger.warning(
                    "LLAMA_SERVICE_CONTEXT does not point to a valid python file. "
                    "Continuing without it."
                )

        if global_service_context is not None:
            return cls.from_service_context(
                global_service_context,
                llm_predictor=llm_predictor,
                prompt_helper=prompt_helper,
                embed_model=embed_model,
                node_parser=node_parser,
                llama_logger=llama_logger,
                callback_manager=callback_manager,
                chunk_size_limit=chunk_size_limit,
            )

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

    @classmethod
    def from_service_context(
        cls,
        service_context: "ServiceContext",
        llm_predictor: Optional[BaseLLMPredictor] = None,
        prompt_helper: Optional[PromptHelper] = None,
        embed_model: Optional[BaseEmbedding] = None,
        node_parser: Optional[NodeParser] = None,
        llama_logger: Optional[LlamaLogger] = None,
        callback_manager: Optional[CallbackManager] = None,
        chunk_size_limit: Optional[int] = None,
    ) -> "ServiceContext":
        """Instaniate a new serivce context using a previous as the defaults."""

        callback_manager = callback_manager or service_context.callback_manager
        llm_predictor = llm_predictor or service_context.llm_predictor
        llm_predictor.callback_manager = callback_manager

        # NOTE: the embed_model isn't used in all indices
        embed_model = embed_model or service_context.embed_model
        embed_model.callback_manager = callback_manager

        # need to ensure chunk_size_limit can still be overwritten from the global
        prompt_helper = prompt_helper or service_context.prompt_helper
        if chunk_size_limit:
            prompt_helper = PromptHelper.from_llm_predictor(
                llm_predictor, chunk_size_limit=chunk_size_limit
            )

        node_parser = node_parser or service_context.node_parser
        if chunk_size_limit:
            node_parser = _get_default_node_parser(
                chunk_size_limit=chunk_size_limit, callback_manager=callback_manager
            )

        llama_logger = llama_logger or service_context.llama_logger

        return cls(
            llm_predictor=llm_predictor,
            embed_model=embed_model,
            prompt_helper=prompt_helper,
            node_parser=node_parser,
            llama_logger=llama_logger,  # deprecated
            callback_manager=callback_manager,
            chunk_size_limit=chunk_size_limit,
        )
