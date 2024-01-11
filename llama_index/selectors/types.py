from abc import abstractmethod
from typing import Any, List, Sequence, Union, Dict

from llama_index.bridge.pydantic import BaseModel, Field
from llama_index.prompts.mixin import PromptMixin, PromptMixinType
from llama_index.schema import QueryBundle, QueryType, NodeWithScore
from llama_index.tools.types import ToolMetadata
from llama_index.core.query_pipeline.query_component import QueryComponent, ChainableMixin, validate_and_convert_stringable, InputKeys, OutputKeys, QUERY_COMPONENT_TYPE
from llama_index.callbacks.base import CallbackManager

MetadataType = Union[str, ToolMetadata]


class SingleSelection(BaseModel):
    """A single selection of a choice."""

    index: int
    reason: str


class MultiSelection(BaseModel):
    """A multi-selection of choices."""

    selections: List[SingleSelection]

    @property
    def ind(self) -> int:
        if len(self.selections) != 1:
            raise ValueError(
                f"There are {len(self.selections)} selections, " "please use .inds."
            )
        return self.selections[0].index

    @property
    def reason(self) -> str:
        if len(self.reasons) != 1:
            raise ValueError(
                f"There are {len(self.reasons)} selections, " "please use .reasons."
            )
        return self.selections[0].reason

    @property
    def inds(self) -> List[int]:
        return [x.index for x in self.selections]

    @property
    def reasons(self) -> List[str]:
        return [x.reason for x in self.selections]


# separate name for clarity and to not confuse function calling model
SelectorResult = MultiSelection


def _wrap_choice(choice: MetadataType) -> ToolMetadata:
    if isinstance(choice, ToolMetadata):
        return choice
    elif isinstance(choice, str):
        return ToolMetadata(description=choice)
    else:
        raise ValueError(f"Unexpected type: {type(choice)}")


def _wrap_query(query: QueryType) -> QueryBundle:
    if isinstance(query, QueryBundle):
        return query
    elif isinstance(query, str):
        return QueryBundle(query_str=query)
    else:
        raise ValueError(f"Unexpected type: {type(query)}")


class BaseSelector(PromptMixin, ChainableMixin):
    def _get_prompt_modules(self) -> PromptMixinType:
        """Get prompt sub-modules."""
        return {}

    def select(
        self, choices: Sequence[MetadataType], query: QueryType
    ) -> SelectorResult:
        metadatas = [_wrap_choice(choice) for choice in choices]
        query_bundle = _wrap_query(query)
        return self._select(choices=metadatas, query=query_bundle)

    async def aselect(
        self, choices: Sequence[MetadataType], query: QueryType
    ) -> SelectorResult:
        metadatas = [_wrap_choice(choice) for choice in choices]
        query_bundle = _wrap_query(query)
        return await self._aselect(choices=metadatas, query=query_bundle)

    @abstractmethod
    def _select(
        self, choices: Sequence[ToolMetadata], query: QueryBundle
    ) -> SelectorResult:
        pass

    @abstractmethod
    async def _aselect(
        self, choices: Sequence[ToolMetadata], query: QueryBundle
    ) -> SelectorResult:
        pass

    def _as_query_component(self, **kwargs: Any) -> QueryComponent:
        """As query component."""
        return SelectorComponent(selector=self)


class SelectorComponent(QueryComponent):
    """Selector component."""

    selector: BaseSelector = Field(..., description="Selector")
    
    class Config:
        arbitrary_types_allowed = True

    def set_callback_manager(self, callback_manager: CallbackManager) -> None:
        """Set callback manager."""
        pass

    def _validate_component_inputs(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Validate component inputs during run_component."""
        if "choices" not in input:
            raise ValueError("Input must have key 'choices'")
        if not isinstance(input["choices"], list):
            raise ValueError("Input choices must be a list")

        for idx, choice in enumerate(input["choices"]):
            # make stringable
            input["choices"][idx] = validate_and_convert_stringable(choice)

        # make sure `query` is stringable
        if "query" not in input:
            raise ValueError("Input must have key 'query'")
        input["query"] = validate_and_convert_stringable(input["query"])
            
        return input

    def _run_component(self, **kwargs: Any) -> Any:
        """Run component."""
        output = self.selector.select(
            kwargs["choices"], kwargs["query"]
        )
        return {"nodes": output}

    async def _arun_component(self, **kwargs: Any) -> Any:
        """Run component (async)."""
        # NOTE: no native async for postprocessor
        return self._run_component(**kwargs)

    @property
    def input_keys(self) -> InputKeys:
        """Input keys."""
        return InputKeys.from_keys({"choices", "query"})

    @property
    def output_keys(self) -> OutputKeys:
        """Output keys."""
        return OutputKeys.from_keys({"output"})



class RouterComponent(QueryComponent):
    """Router Component.

    Routes queries to different query components based on a selector.

    Assumes a single query component is selected.
    
    """
    selector: BaseSelector = Field(..., description="Selector")
    choices: List[str] = Field(..., description="Choices (must correspond to components)")
    components: List[QueryComponent] = Field(..., description="Components (must correspond to choices)")
    
    class Config:
        arbitrary_types_allowed = True

    def set_callback_manager(self, callback_manager: CallbackManager) -> None:
        """Set callback manager."""
        pass

    def _validate_component_inputs(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Validate component inputs during run_component."""
        # make sure `query` is stringable
        if "query" not in input:
            raise ValueError("Input must have key 'query'")
        input["query"] = validate_and_convert_stringable(input["query"])
            
        return input

    def _run_component(self, **kwargs: Any) -> Any:
        """Run component."""
        # for the output selection, run the corresponding component, aggregate into list
        sel_output = self.selector.select(
            self.choices, kwargs["query"]
        )
        # assume one selection
        if len(sel_output.selections) != 1:
            raise ValueError("Expected one selection")
        component = self.components[sel_output.ind]
        # run component
        output = component.run_component(**kwargs)

        return {"nodes": output}

    async def _arun_component(self, **kwargs: Any) -> Any:
        """Run component (async)."""
        # for the output selection, run the corresponding component, aggregate into list
        sel_output = await self.selector.aselect(
            self.choices, kwargs["query"]
        )
        # assume one selection
        if len(sel_output.selections) != 1:
            raise ValueError("Expected one selection")
        component = self.components[sel_output.ind]
        # run component
        output = await component.arun_component(**kwargs)

        return {"nodes": output}

    @property
    def input_keys(self) -> InputKeys:
        """Input keys."""
        return InputKeys.from_keys({"query"})

    @property
    def output_keys(self) -> OutputKeys:
        """Output keys."""
        return OutputKeys.from_keys({"output"})