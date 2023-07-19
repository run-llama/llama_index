import logging
from dataclasses import dataclass
from typing import Optional, Union
import os

import llama_index
from llama_index.utils import get_cache_dir
from llama_index.callbacks.base import CallbackManager
from llama_index.embeddings.base import BaseEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.indices.prompt_helper import PromptHelper
from llama_index.llm_predictor import LLMPredictor
from llama_index.llm_predictor.base import BaseLLMPredictor, LLMMetadata
from llama_index.llms.base import LLM
from llama_index.llms.utils import LLMType
from llama_index.logger import LlamaLogger
from llama_index.node_parser.interface import NodeParser
from llama_index.node_parser.simple import SimpleNodeParser
from llama_index.embeddings import (
    DEFAULT_HUGGINGFACE_EMBEDDING_MODEL,
    LangchainEmbedding,
)

logger = logging.getLogger(__name__)


def _get_default_node_parser(
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    callback_manager: Optional[CallbackManager] = None,
) -> NodeParser:
    """Get default node parser."""
    return SimpleNodeParser.from_defaults(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        callback_manager=callback_manager,
    )


def _get_default_prompt_helper(
    llm_metadata: LLMMetadata,
    context_window: Optional[int] = None,
    num_output: Optional[int] = None,
) -> PromptHelper:
    """Get default prompt helper."""
    if context_window is not None:
        llm_metadata.context_window = context_window
    if num_output is not None:
        llm_metadata.num_output = num_output
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
        llm_predictor: Optional[Union[BaseLLMPredictor, str]] = None,
        llm: Optional[LLMType] = None,
        prompt_helper: Optional[PromptHelper] = None,
        embed_model: Optional[Union[BaseEmbedding, str]] = None,
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
            llm_predictor (Optional[Union[BaseLLMPredictor, str]]): LLMPredictor
                or "local" (use local model)
            prompt_helper (Optional[PromptHelper]): PromptHelper
            embed_model (Optional[Union[BaseEmbedding, str]]): BaseEmbedding
                or "local" (use local model)
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

        if isinstance(embed_model, str):
            embed_model = _get_embed_model_from_str(embed_model)

        if isinstance(llm_predictor, str):
            llm_predictor = _get_llm_predictor_from_str(llm_predictor)

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
            llm_metadata=llm_predictor.metadata,
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
        llm: Optional[LLM] = None,
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
            llm_metadata=llm_predictor.metadata,
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

    @property
    def llm(self) -> LLM:
        if not isinstance(self.llm_predictor, LLMPredictor):
            raise ValueError("llm_predictor must be an instance of LLMPredictor")
        return self.llm_predictor.llm


def set_global_service_context(service_context: Optional[ServiceContext]) -> None:
    """Helper function to set the global service context."""
    llama_index.global_service_context = service_context


def _get_embed_model_from_str(config: str) -> BaseEmbedding:
    splits = config.split(":", 1)
    is_local = splits[0]
    model_name = splits[1] if len(splits) > 1 else None
    if is_local != "local":
        raise ValueError(
            "embed_model must start with str 'local' or of type BaseEmbedding"
        )
    try:
        from langchain.embeddings import HuggingFaceEmbeddings
    except ImportError as exc:
        raise ImportError(
            "Could not import sentence_transformers or langchain package. "
            "Please install with `pip install sentence-transformers langchain`."
        ) from exc

    embed_model = LangchainEmbedding(
        HuggingFaceEmbeddings(
            model_name=model_name or DEFAULT_HUGGINGFACE_EMBEDDING_MODEL
        )
    )
    return embed_model


DEFAULT_LOCAL_LLM_NAME = "llama-2-13b"
DEFAULT_LOCAL_LLM_URL = """
https://huggingface.co/TheBloke/Llama-2-13B-chat-GGML/resolve\
/main/llama-2-13b-chat.ggmlv3.q4_0.bin\
"""


def _get_llm_predictor_from_str(config: str) -> BaseLLMPredictor:
    if config != "local":
        raise ValueError(
            "llm_predictor must start with str 'local' or of type BaseLlmPredictor"
        )
    try:
        from langchain.llms import LlamaCpp
        from llama_index import LLMPredictor
    except ImportError as exc:
        raise ImportError(
            "Could not import llama-cpp-python or langchain package. "
            "Please install with `pip install llama-cpp-python langchain`."
            "More advanced installation (e.g. GPU/BLAS offloading): ", 
            "https://github.com/abetlen/llama-cpp-python"
        ) from exc

    model_path = os.path.join(
        get_cache_dir(), "models", DEFAULT_LOCAL_LLM_NAME, "ggml-model-q4_0.bin"
    )
    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        _download_to_cache_dir(
            url=DEFAULT_LOCAL_LLM_URL,
            path=model_path,
        )
        assert os.path.exists(model_path)

    llm_predictor = LLMPredictor(
        LlamaCpp(
            model_path=model_path,
            n_ctx=4096,
            suffix="[/INST]",
        )
    )
    print("LLM metadata:", llm_predictor.llm.metadata)
    return llm_predictor


def _download_to_cache_dir(url: str, path: str) -> None:
    import requests
    from tqdm import tqdm

    # use a context manager to make an HTTP request and file
    completed = False
    try:
        print("Downloading url", url, "to path", path)
        with requests.get(url, stream=True) as r:
            with open(path, "wb") as file:
                total_size = int(r.headers.get("Content-Length") or "0")
                if total_size < 1000 * 1000:
                    raise ValueError(
                        "Content should be at least 1 MB, but is only",
                        r.headers.get("Content-Length"),
                        "bytes",
                    )
                print("total size (MB):", round(total_size / 1000 / 1000, 2))
                chunk_size = 1024 * 1024  # 1 MB
                for chunk in tqdm(
                    r.iter_content(chunk_size=chunk_size),
                    total=int(total_size / chunk_size),
                ):
                    file.write(chunk)
        completed = True
    finally:
        if not completed:
            print("Download incomplete.", "Removing partially downloaded file.")
            os.remove(path)
