from typing import Any, Callable, List, Optional

from llama_index.bridge.pydantic import BaseModel, PrivateAttr
from llama_index.callbacks.base import BaseCallbackHandler, CallbackManager
from llama_index.embeddings import BaseEmbedding
from llama_index.embeddings.utils import resolve_embed_model
from llama_index.llm_predictor import LLMPredictor
from llama_index.llms import LLM, ChatMessage
from llama_index.llms.utils import resolve_llm
from llama_index.utils import get_tokenizer, set_global_tokenizer


class Settings(BaseModel):
    """Settings for the Llama Index, lazily initialized."""

    system_prompt: Optional[str] = PrivateAttr(None)
    messages_to_prompt: Optional[Callable[[List[ChatMessage]], str]] = None
    completion_to_prompt: Optional[Callable[[str], str]] = None

    # lazy initialization
    _llm: Optional[LLM] = PrivateAttr(None)
    _llm_predictor: Optional[LLMPredictor] = PrivateAttr(None)
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
    def llm_predictor(self) -> LLMPredictor:
        """Get the LLM predictor."""
        if self._llm_predictor is None:
            self._llm_predictor = LLMPredictor(
                self.llm, system_prompt=self.system_prompt
            )
        return self._llm_predictor

    @llm_predictor.setter
    def llm_predictor(self, llm_predictor: LLMPredictor) -> None:
        """Set the LLM predictor."""
        self._llm_predictor = llm_predictor

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
