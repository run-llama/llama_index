"""Agent components."""

from inspect import signature
from typing import Any, Callable, Dict, Optional, Set, Tuple, cast

from llama_index.core.base.query_pipeline.query import (
    InputKeys,
    OutputKeys,
    QueryComponent,
)
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.query_pipeline.components.function import FnComponent, get_parameters
# from llama_index.core.query_pipeline.components.input import InputComponent


class BaseStatefulComponent(QueryComponent):
    """Takes in agent inputs and transforms it into desired outputs."""



class StatefulFnComponent(BaseStatefulComponent, FnComponent):
    """Query component that takes in an arbitrary function.

    Stateful version of `FnComponent`. Expects functions to have `state` as the first argument.
    
    """

    def __init__(
        self,
        fn: Callable,
        req_params: Optional[Set[str]] = None,
        opt_params: Optional[Set[str]] = None,
        **kwargs: Any
    ) -> None:
        """Init params."""

        # determine parameters
        default_req_params, default_opt_params = get_parameters(fn)
        # make sure task and step are part of the list, and remove them from the list
        if "state" not in default_req_params:
            raise ValueError(
                "StatefulFnComponent must have 'state' as required parameters"
            )

        default_req_params = default_req_params - {"state"}
        default_opt_params = default_opt_params - {"state"}
    
        super().__init__(fn=fn, req_params=req_params, opt_params=opt_params, **kwargs)
    
    @property
    def input_keys(self) -> InputKeys:
        """Input keys."""
        return InputKeys.from_keys(
            required_keys={"state", *self._req_params},
            optional_keys=self._opt_params,
        )

    @property
    def output_keys(self) -> OutputKeys:
        """Output keys."""
        # output can be anything, overrode validate function
        return OutputKeys.from_keys({self.output_key})