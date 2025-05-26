"""Agent components."""

from typing import Any, Callable, Dict, Optional, Set

from llama_index.core.base.query_pipeline.query import (
    QueryComponent,
)
from llama_index.core.bridge.pydantic import Field
from llama_index.core.query_pipeline.components.function import (
    FnComponent,
    get_parameters,
)

# from llama_index.core.query_pipeline.components.input import InputComponent


class BaseStatefulComponent(QueryComponent):
    """Takes in agent inputs and transforms it into desired outputs."""

    state: Dict[str, Any] = Field(
        default_factory=dict, description="State of the pipeline."
    )

    def reset_state(self) -> None:
        """Reset state."""
        self.state = {}


class StatefulFnComponent(BaseStatefulComponent, FnComponent):
    """
    Query component that takes in an arbitrary function.

    Stateful version of `FnComponent`. Expects functions to have `state` as the first argument.

    """

    def __init__(
        self,
        fn: Callable,
        req_params: Optional[Set[str]] = None,
        opt_params: Optional[Set[str]] = None,
        state: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
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

        if req_params is None:
            req_params = default_req_params
        if opt_params is None:
            opt_params = default_opt_params

        super().__init__(
            fn=fn,
            req_params=req_params,
            opt_params=opt_params,
            state=state or {},
            **kwargs,
        )

    def _run_component(self, **kwargs: Any) -> Dict:
        """Run component."""
        kwargs.update({"state": self.state})
        return super()._run_component(**kwargs)

    async def _arun_component(self, **kwargs: Any) -> Any:
        """Async run component."""
        kwargs.update({"state": self.state})
        return await super()._arun_component(**kwargs)
