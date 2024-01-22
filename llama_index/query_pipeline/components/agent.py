"""Agent components."""

from inspect import signature
from typing import Any, Callable, Dict, Optional, Set, Tuple, cast

from llama_index.bridge.pydantic import Field, PrivateAttr
from llama_index.callbacks.base import CallbackManager
from llama_index.core.query_pipeline.query_component import (
    InputKeys,
    OutputKeys,
    QueryComponent,
)


def get_parameters(fn: Callable) -> Tuple[Set[str], Set[str]]:
    """Get parameters from function.

    Returns:
        Tuple[Set[str], Set[str]]: required and optional parameters

    """
    # please write function below
    params = signature(fn).parameters
    required_params = set()
    optional_params = set()
    for param_name in params:
        param_default = params[param_name].default
        if param_default is params[param_name].empty:
            required_params.add(param_name)
        else:
            optional_params.add(param_name)
    return required_params, optional_params



def default_agent_input_fn(task: Any, state: dict) -> dict:
    """Default agent input function."""
    from llama_index.agent.types import Task
    task = cast(Task, task)

    return {"input": task.input}


class AgentInputComponent(QueryComponent):
    """Takes in agent inputs and transforms it into desired outputs."""

    fn: Callable = Field(..., description="Function to run.")
    async_fn: Optional[Callable] = Field(
        None, description="Async function to run. If not provided, will run `fn`."
    )

    _req_params: Set[str] = PrivateAttr()
    _opt_params: Set[str] = PrivateAttr()

    def __init__(
        self,
        fn: Callable,
        async_fn: Optional[Callable] = None,
        req_params: Optional[Set[str]] = None,
        opt_params: Optional[Set[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize."""
        # determine parameters
        default_req_params, default_opt_params = get_parameters(fn)
        if req_params is None:
            req_params = default_req_params
        if opt_params is None:
            opt_params = default_opt_params

        self._req_params = req_params
        self._opt_params = opt_params
        super().__init__(fn=fn, async_fn=async_fn, **kwargs)

    class Config:
        arbitrary_types_allowed = True

    def set_callback_manager(self, callback_manager: CallbackManager) -> None:
        """Set callback manager."""
        # TODO: implement

    def _validate_component_inputs(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Validate component inputs during run_component."""
        from llama_index.agent.types import Task

        if "task" not in input:
            raise ValueError("Input must have key 'task'")
        if not isinstance(input["task"], Task):
            raise ValueError("Input must have key 'task' of type Task")
        
        if "state" not in input:
            raise ValueError("Input must have key 'state'")
        if not isinstance(input["state"], dict):
            raise ValueError("Input must have key 'state' of type dict")

        return input

    def validate_component_outputs(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """Validate component outputs."""
        # NOTE: we override this to do nothing
        return output

    def _validate_component_outputs(self, input: Dict[str, Any]) -> Dict[str, Any]:
        return input

    def _run_component(self, **kwargs: Any) -> Dict:
        """Run component."""
        output = self.fn(**kwargs)
        if not isinstance(output, dict):
            raise ValueError("Output must be a dictionary")
        
        return output

    async def _arun_component(self, **kwargs: Any) -> Any:
        """Run component (async)."""
        if self.async_fn is None:
            return self._run_component(**kwargs)
        else:
            output = await self.async_fn(**kwargs)
            if not isinstance(output, dict):
                raise ValueError("Output must be a dictionary")
            return output

    @property
    def input_keys(self) -> InputKeys:
        """Input keys."""
        return InputKeys.from_keys(
            required_keys={"task", "state", *self._req_params}, optional_keys=self._opt_params
        )

    @property
    def output_keys(self) -> OutputKeys:
        """Output keys."""
        # output can be anything, overrode validate function
        return OutputKeys.from_keys(set())


class AgentFnComponent(QueryComponent):
    """Function component for agents.

    Designed to let users easily modify state.
    
    """

    fn: Callable = Field(..., description="Function to run.")
    async_fn: Optional[Callable] = Field(
        None, description="Async function to run. If not provided, will run `fn`."
    )

    _req_params: Set[str] = PrivateAttr()
    _opt_params: Set[str] = PrivateAttr()

    def __init__(
        self,
        fn: Callable,
        async_fn: Optional[Callable] = None,
        req_params: Optional[Set[str]] = None,
        opt_params: Optional[Set[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize."""
        # determine parameters
        default_req_params, default_opt_params = get_parameters(fn)
        # make sure task and step are part of the list, and remove them from the list
        if "task" not in default_req_params or "state" not in default_req_params:
            raise ValueError("AgentFnComponent must have 'task' and 'state' as required parameters")

        default_req_params = default_req_params - {"task", "state"}
        default_opt_params = default_opt_params - {"task", "state"}

        if req_params is None:
            req_params = default_req_params
        if opt_params is None:
            opt_params = default_opt_params

        self._req_params = req_params
        self._opt_params = opt_params
        super().__init__(fn=fn, async_fn=async_fn, **kwargs)

    class Config:
        arbitrary_types_allowed = True

    def set_callback_manager(self, callback_manager: CallbackManager) -> None:
        """Set callback manager."""
        # TODO: implement

    def _validate_component_inputs(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Validate component inputs during run_component."""
        from llama_index.agent.types import Task

        if "task" not in input:
            raise ValueError("Input must have key 'task'")
        if not isinstance(input["task"], Task):
            raise ValueError("Input must have key 'task' of type Task")
        
        if "state" not in input:
            raise ValueError("Input must have key 'state'")
        if not isinstance(input["state"], dict):
            raise ValueError("Input must have key 'state' of type dict")

        return input

    def validate_component_outputs(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """Validate component outputs."""
        # NOTE: we override this to do nothing
        return output

    def _validate_component_outputs(self, input: Dict[str, Any]) -> Dict[str, Any]:
        return input

    def _run_component(self, **kwargs: Any) -> Dict:
        """Run component."""
        output = self.fn(**kwargs)
        # if not isinstance(output, dict):
        #     raise ValueError("Output must be a dictionary")
        
        return {"output": output}

    async def _arun_component(self, **kwargs: Any) -> Any:
        """Run component (async)."""
        if self.async_fn is None:
            return self._run_component(**kwargs)
        else:
            output = await self.async_fn(**kwargs)
            # if not isinstance(output, dict):
            #     raise ValueError("Output must be a dictionary")
            return {"output": output}

    @property
    def input_keys(self) -> InputKeys:
        """Input keys."""
        return InputKeys.from_keys(
            required_keys={"task", "state", *self._req_params}, optional_keys=self._opt_params
        )

    @property
    def output_keys(self) -> OutputKeys:
        """Output keys."""
        # output can be anything, overrode validate function
        return OutputKeys.from_keys({"output"})