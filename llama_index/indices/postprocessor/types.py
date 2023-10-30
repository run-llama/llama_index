from abc import ABC, abstractmethod
from typing import List, Optional

from llama_index.bridge.pydantic import Field
from llama_index.callbacks import CallbackManager
from llama_index.indices.query.schema import QueryBundle
from llama_index.prompts.mixin import PromptDictType, PromptMixinType
from llama_index.schema import BaseComponent, NodeWithScore


class BaseNodePostprocessor(BaseComponent, ABC):
    callback_manager: CallbackManager = Field(
        default_factory=CallbackManager, exclude=True
    )

    class Config:
        arbitrary_types_allowed = True

    def _get_prompts(self) -> PromptDictType:
        """Get prompts."""
        # set by default since most postprocessors don't require prompts
        return {}

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""

    def _get_prompt_modules(self) -> PromptMixinType:
        """Get prompt modules."""
        return {}

    # implement class_name so users don't have to worry about it when extending
    @classmethod
    def class_name(cls) -> str:
        return "BaseNodePostprocessor"

    @abstractmethod
    def postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """Postprocess nodes."""
