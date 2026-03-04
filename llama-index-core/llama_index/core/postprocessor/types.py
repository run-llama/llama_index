import asyncio
from abc import ABC, abstractmethod
from typing import List, Optional

from llama_index.core.bridge.pydantic import Field, ConfigDict
from llama_index.core.callbacks import CallbackManager
from llama_index.core.instrumentation import DispatcherSpanMixin
from llama_index.core.prompts.mixin import PromptDictType, PromptMixinType
from llama_index.core.schema import BaseComponent, NodeWithScore, QueryBundle


class BaseNodePostprocessor(BaseComponent, DispatcherSpanMixin, ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    callback_manager: CallbackManager = Field(
        default_factory=CallbackManager, exclude=True
    )

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

    def postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
        query_str: Optional[str] = None,
    ) -> List[NodeWithScore]:
        """Postprocess nodes."""
        if query_str is not None and query_bundle is not None:
            raise ValueError("Cannot specify both query_str and query_bundle")
        elif query_str is not None:
            query_bundle = QueryBundle(query_str)
        else:
            pass
        return self._postprocess_nodes(nodes, query_bundle)

    @abstractmethod
    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """Postprocess nodes."""

    async def apostprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
        query_str: Optional[str] = None,
    ) -> List[NodeWithScore]:
        """Postprocess nodes (async)."""
        if query_str is not None and query_bundle is not None:
            raise ValueError("Cannot specify both query_str and query_bundle")
        elif query_str is not None:
            query_bundle = QueryBundle(query_str)
        else:
            pass
        return await self._apostprocess_nodes(nodes, query_bundle)

    async def _apostprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """Postprocess nodes (async)."""
        return await asyncio.to_thread(self._postprocess_nodes, nodes, query_bundle)
