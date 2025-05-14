"""Agent components."""

from inspect import signature
from typing import Any, Callable, Dict, Optional, Set, Tuple, cast
from typing_extensions import Annotated

from llama_index.core.base.query_pipeline.query import (
    InputKeys,
    OutputKeys,
    QueryComponent,
)
from llama_index.core.bridge.pydantic import (
    Field,
    PrivateAttr,
    ConfigDict,
    WithJsonSchema,
)
from llama_index.core.callbacks.base import CallbackManager


AnnotatedCallable = Annotated[
    Callable,
    WithJsonSchema({"type": "string"}, mode="serialization"),
    WithJsonSchema({"type": "string"}, mode="validation"),
]


def get_parameters(fn: Callable) -> Tuple[Set[str], Set[str]]:
    """
    Get parameters from function.

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
    from llama_index.core.agent.types import Task

    task = cast(Task, task)

    return {"input": task.input}


class AgentInputComponent(QueryComponent):
    """
    Takes in agent inputs and transforms it into desired outputs.

    NOTE: this is now deprecated in favor of using `StatefulFnComponent`.

    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    fn: AnnotatedCallable = Field(..., description="Function to run.")
    async_fn: Optional[AnnotatedCallable] = Field(
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
        super().__init__(fn=fn, async_fn=async_fn, **kwargs)
        default_req_params, default_opt_params = get_parameters(fn)
        if req_params is None:
            req_params = default_req_params
        if opt_params is None:
            opt_params = default_opt_params

        self._req_params = req_params
        self._opt_params = opt_params

    def set_callback_manager(self, callback_manager: CallbackManager) -> None:
        """Set callback manager."""
        # TODO: implement

    def _validate_component_inputs(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Validate component inputs during run_component."""
        from llama_index.core.agent.types import Task

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
            required_keys={"task", "state", *self._req_params},
            optional_keys=self._opt_params,
        )

    @property
    def output_keys(self) -> OutputKeys:
        """Output keys."""
        # output can be anything, overrode validate function
        return OutputKeys.from_keys(set())


class BaseAgentComponent(QueryComponent):
    """
    Agent component.

    Abstract class used for type checking.

    """


class AgentFnComponent(BaseAgentComponent):
    """
    Function component for agents.

    Designed to let users easily modify state.

    NOTE: this is now deprecated in favor of using `StatefulFnComponent`.

    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
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
        super().__init__(fn=fn, async_fn=async_fn, **kwargs)
        default_req_params, default_opt_params = get_parameters(fn)
        # make sure task and step are part of the list, and remove them from the list
        if "task" not in default_req_params or "state" not in default_req_params:
            raise ValueError(
                "AgentFnComponent must have 'task' and 'state' as required parameters"
            )

        default_req_params = default_req_params - {"task", "state"}
        default_opt_params = default_opt_params - {"task", "state"}

        if req_params is None:
            req_params = default_req_params
        if opt_params is None:
            opt_params = default_opt_params

        self._req_params = req_params
        self._opt_params = opt_params

    def set_callback_manager(self, callback_manager: CallbackManager) -> None:
        """Set callback manager."""
        # TODO: implement

    def _validate_component_inputs(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Validate component inputs during run_component."""
        from llama_index.core.agent.types import Task

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
            required_keys={"task", "state", *self._req_params},
            optional_keys=self._opt_params,
        )

    @property
    def output_keys(self) -> OutputKeys:
        """Output keys."""
        # output can be anything, overrode validate function
        return OutputKeys.from_keys({"output"})


class CustomAgentComponent(BaseAgentComponent):
    """
    Custom component for agents.

    Designed to let users easily modify state.

    NOTE: this is now deprecated in favor of using `StatefulFnComponent`.

    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    callback_manager: CallbackManager = Field(
        default_factory=CallbackManager, description="Callback manager"
    )

    def set_callback_manager(self, callback_manager: CallbackManager) -> None:
        """Set callback manager."""
        self.callback_manager = callback_manager
        # TODO: refactor to put this on base class
        for component in self.sub_query_components:
            component.set_callback_manager(callback_manager)

    def _validate_component_inputs(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Validate component inputs during run_component."""
        # NOTE: user can override this method to validate inputs
        # but we do this by default for convenience
        return input

    async def _arun_component(self, **kwargs: Any) -> Any:
        """Run component (async)."""
        raise NotImplementedError("This component does not support async run.")

    @property
    def _input_keys(self) -> Set[str]:
        """Input keys dict."""
        raise NotImplementedError("Not implemented yet. Please override this method.")

    @property
    def _optional_input_keys(self) -> Set[str]:
        """Optional input keys dict."""
        return set()

    @property
    def _output_keys(self) -> Set[str]:
        """Output keys dict."""
        raise NotImplementedError("Not implemented yet. Please override this method.")

    @property
    def input_keys(self) -> InputKeys:
        """Input keys."""
        # NOTE: user can override this too, but we have them implement an
        # abstract method to make sure they do it

        input_keys = self._input_keys.union({"task", "state"})
        return InputKeys.from_keys(
            required_keys=input_keys, optional_keys=self._optional_input_keys
        )

    @property
    def output_keys(self) -> OutputKeys:
        """Output keys."""
        # NOTE: user can override this too, but we have them implement an
        # abstract method to make sure they do it
        return OutputKeys.from_keys(self._output_keys)
