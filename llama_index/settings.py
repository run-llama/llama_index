from typing import TYPE_CHECKING, Any, Callable, List, Optional

if TYPE_CHECKING:
    from llama_index import ServiceContext

from llama_index.bridge.pydantic import BaseModel, PrivateAttr
from llama_index.callbacks.base import BaseCallbackHandler, CallbackManager
from llama_index.embeddings import BaseEmbedding
from llama_index.embeddings.utils import resolve_embed_model
from llama_index.llm_predictor.base import BaseLLMPredictor, LLMPredictor
from llama_index.llms import LLM, ChatMessage
from llama_index.llms.utils import resolve_llm
from llama_index.utils import get_tokenizer, set_global_tokenizer


class _Settings(BaseModel):
    """Settings for the Llama Index, lazily initialized."""

    system_prompt: Optional[str] = None
    messages_to_prompt: Optional[Callable[[List[ChatMessage]], str]] = None
    completion_to_prompt: Optional[Callable[[str], str]] = None

    # lazy initialization
    _llm: Optional[LLM] = PrivateAttr(None)
    _llm_predictor: Optional[BaseLLMPredictor] = PrivateAttr(None)
    _embed_model: Optional[LLM] = PrivateAttr(None)
    _callback_manager: Optional[CallbackManager] = PrivateAttr(None)
    _tokenizer: Optional[Callable[[str], List[Any]]] = PrivateAttr(None)

    # -- LLM --

    @property
    def llm(self) -> LLM:
        """Get the LLM."""
        if self._llm is None:
            self._llm = resolve_llm("default")
        return self._llm

    @llm.setter
    def llm(self, llm: LLM) -> None:
        """Set the LLM."""
        self._llm = llm

    @property
    def llm_predictor(self) -> BaseLLMPredictor:
        """Get the LLM predictor."""
        if self._llm_predictor is None:
            self._llm_predictor = LLMPredictor(
                self.llm,
                system_prompt=self.system_prompt,
            )
        return self._llm_predictor

    @llm_predictor.setter
    def llm_predictor(self, llm_predictor: BaseLLMPredictor) -> None:
        """Set the LLM predictor."""
        self._llm_predictor = llm_predictor

    def get_llm_predictor_from_llm(
        self,
        llm: Optional[LLM] = None,
        llm_predictor: Optional[BaseLLMPredictor] = None,
    ) -> BaseLLMPredictor:
        if llm_predictor is not None:
            return llm_predictor
        elif llm is not None:
            return LLMPredictor(llm, system_prompt=self.system_prompt)
        else:
            return self.llm_predictor

    # -- Embedding --

    @property
    def embed_model(self) -> BaseEmbedding:
        """Get the embedding model."""
        if self._embed_model is None:
            self._embed_model = resolve_embed_model("default")
        return self._embed_model

    @embed_model.setter
    def embed_model(self, embed_model: BaseEmbedding) -> None:
        """Set the embedding model."""
        self._embed_model = embed_model

    # -- Callbacks --

    @property
    def global_handler(self) -> Optional[BaseCallbackHandler]:
        """Get the global handler."""
        import llama_index

        # TODO: deprecated?
        return llama_index.global_handler

    @global_handler.setter
    def global_handler(self, eval_mode: str, **eval_params: Any) -> None:
        """Set the global handler."""
        from llama_index import set_global_handler

        # TODO: deprecated?
        set_global_handler(eval_mode, **eval_params)

    @property
    def callback_manager(self) -> Optional[BaseCallbackHandler]:
        """Get the callback manager."""
        if self._callback_manager is None:
            self._callback_manager = CallbackManager()
        return self._callback_manager

    @callback_manager.setter
    def callback_manager(self, callback_manager: CallbackManager) -> None:
        """Set the callback manager."""
        self._callback_manager = callback_manager

    # -- Tokenizer --

    @property
    def tokenizer(self) -> Callable[[str], List[Any]]:
        """Get the tokenizer."""
        if self._tokenizer is None:
            self._tokenizer = get_tokenizer()

        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer: Callable[[str], List[Any]]) -> None:
        """Set the tokenizer."""
        self._tokenizer = tokenizer

        # TODO: deprecated
        set_global_tokenizer(tokenizer)


# Singleton
Settings = _Settings()


# -- Helper functions for deprecation/migration --


def llm_from_settings_or_context(
    settings: _Settings, context: Optional["ServiceContext"]
) -> LLM:
    """Get settings from either settings or context."""
    if context is not None:
        return context.llm

    return settings.llm


def llm_predictor_from_settings_or_context(
    settings: _Settings, context: Optional["ServiceContext"]
) -> LLMPredictor:
    """Get settings from either settings or context."""
    if context is not None:
        return context.llm_predictor

    return settings.llm_predictor


def embed_model_from_settings_or_context(
    settings: _Settings, context: Optional["ServiceContext"]
) -> BaseEmbedding:
    """Get settings from either settings or context."""
    if context is not None:
        return context.embed_model

    return settings.embed_model


def callback_manager_from_settings_or_context(
    settings: _Settings, context: Optional["ServiceContext"]
) -> CallbackManager:
    """Get settings from either settings or context."""
    if context is not None:
        return context.callback_manager

    return settings.callback_manager
