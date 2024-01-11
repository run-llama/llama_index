"""Router components."""


from abc import abstractmethod
from typing import Any, List, Sequence, Union, Dict

from llama_index.bridge.pydantic import BaseModel, Field, PrivateAttr
from llama_index.prompts.mixin import PromptMixin, PromptMixinType
from llama_index.schema import QueryBundle, QueryType, NodeWithScore
from llama_index.tools.types import ToolMetadata
from llama_index.core.query_pipeline.query_component import QueryComponent, ChainableMixin, validate_and_convert_stringable, InputKeys, OutputKeys, QUERY_COMPONENT_TYPE
from llama_index.callbacks.base import CallbackManager
from llama_index.selectors.types import BaseSelector


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
        return {"output": output.selections}

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

    _query_keys: List[str] = PrivateAttr()

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self, 
        selector: BaseSelector,
        choices: List[str],
        components: List[QUERY_COMPONENT_TYPE],
    ) -> None:
        """Init."""
        new_components = []
        query_keys = []
        for component in components:
            if isinstance(component, ChainableMixin):
                new_component = component.as_query_component()
            else:
                new_component = component

            # validate component has one input key
            if len(new_component.free_req_input_keys) != 1:
                raise ValueError("Expected one required input key")
            query_keys.append(list(new_component.free_req_input_keys)[0])
            new_components.append(new_component)

        self._query_keys = query_keys

        super().__init__(
            selector=selector,
            choices=choices,
            components=new_components,
        )
                
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

    def validate_component_outputs(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Validate component inputs during run_component."""
        return input

    def _validate_component_outputs(self, output: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

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
        # run with input_keys of component
        output = component.run_component(
            **{self._query_keys[sel_output.ind]: kwargs["query"]}
        )

        return output

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
        output = await component.arun_component(
            **{self._query_keys[sel_output.ind]: kwargs["query"]}
        )

        return output

    @property
    def input_keys(self) -> InputKeys:
        """Input keys."""
        return InputKeys.from_keys({"query"})

    @property
    def output_keys(self) -> OutputKeys:
        """Output keys."""
        # not used
        return OutputKeys.from_keys(set())