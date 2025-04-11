import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from llama_index.core.base.query_pipeline.query import (
    ChainableMixin,
    InputKeys,
    OutputKeys,
    QueryComponent,
    validate_and_convert_stringable,
)
from llama_index.core.bridge.pydantic import Field, SerializeAsAny, ConfigDict
from llama_index.core.callbacks import CallbackManager
from llama_index.core.instrumentation import DispatcherSpanMixin
from llama_index.core.prompts.mixin import PromptDictType, PromptMixinType
from llama_index.core.schema import BaseComponent, NodeWithScore, QueryBundle


class BaseNodePostprocessor(ChainableMixin, BaseComponent, DispatcherSpanMixin, ABC):
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

    def _as_query_component(self, **kwargs: Any) -> QueryComponent:
        """As query component."""
        return PostprocessorComponent(postprocessor=self)


class PostprocessorComponent(QueryComponent):
    """Postprocessor component."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    postprocessor: SerializeAsAny[BaseNodePostprocessor] = Field(
        ..., description="Postprocessor"
    )

    def set_callback_manager(self, callback_manager: CallbackManager) -> None:
        """Set callback manager."""
        self.postprocessor.callback_manager = callback_manager

    def _validate_component_inputs(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Validate component inputs during run_component."""
        # make sure `nodes` is a list of nodes
        if "nodes" not in input:
            raise ValueError("Input must have key 'nodes'")
        nodes = input["nodes"]
        if not isinstance(nodes, list):
            raise ValueError("Input nodes must be a list")
        for node in nodes:
            if not isinstance(node, NodeWithScore):
                raise ValueError("Input nodes must be a list of NodeWithScore")

        # if query_str exists, make sure `query_str` is stringable
        if "query_str" in input:
            input["query_str"] = validate_and_convert_stringable(input["query_str"])

        return input

    def _run_component(self, **kwargs: Any) -> Any:
        """Run component."""
        output = self.postprocessor.postprocess_nodes(
            kwargs["nodes"], query_str=kwargs.get("query_str", None)
        )
        return {"nodes": output}

    async def _arun_component(self, **kwargs: Any) -> Any:
        """Run component (async)."""
        # NOTE: no native async for postprocessor
        return self._run_component(**kwargs)

    @property
    def input_keys(self) -> InputKeys:
        """Input keys."""
        return InputKeys.from_keys({"nodes"}, optional_keys={"query_str"})

    @property
    def output_keys(self) -> OutputKeys:
        """Output keys."""
        return OutputKeys.from_keys({"nodes"})
