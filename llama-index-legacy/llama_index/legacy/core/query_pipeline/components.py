"""Query pipeline components."""

from inspect import signature
from typing import Any, Callable, Dict, Optional, Set, Tuple

from llama_index.legacy.bridge.pydantic import Field, PrivateAttr
from llama_index.legacy.callbacks.base import CallbackManager
from llama_index.legacy.core.query_pipeline.query_component import (
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


class FnComponent(QueryComponent):
    """Query component that takes in an arbitrary function."""

    fn: Callable = Field(..., description="Function to run.")
    async_fn: Optional[Callable] = Field(
        None, description="Async function to run. If not provided, will run `fn`."
    )
    output_key: str = Field(
        default="output", description="Output key for component output."
    )

    _req_params: Set[str] = PrivateAttr()
    _opt_params: Set[str] = PrivateAttr()

    def __init__(
        self,
        fn: Callable,
        async_fn: Optional[Callable] = None,
        req_params: Optional[Set[str]] = None,
        opt_params: Optional[Set[str]] = None,
        output_key: str = "output",
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
        super().__init__(fn=fn, async_fn=async_fn, output_key=output_key, **kwargs)

    class Config:
        arbitrary_types_allowed = True

    def set_callback_manager(self, callback_manager: CallbackManager) -> None:
        """Set callback manager."""
        # TODO: implement

    def _validate_component_inputs(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Validate component inputs during run_component."""
        # check that all required parameters are present
        missing_params = self._req_params - set(input.keys())
        if missing_params:
            raise ValueError(
                f"Missing required parameters: {missing_params}. "
                f"Input keys: {input.keys()}"
            )

        # check that no extra parameters are present
        extra_params = set(input.keys()) - self._req_params - self._opt_params
        if extra_params:
            raise ValueError(
                f"Extra parameters: {extra_params}. " f"Input keys: {input.keys()}"
            )
        return input

    def _run_component(self, **kwargs: Any) -> Dict:
        """Run component."""
        return {self.output_key: self.fn(**kwargs)}

    async def _arun_component(self, **kwargs: Any) -> Any:
        """Run component (async)."""
        if self.async_fn is None:
            return self._run_component(**kwargs)
        else:
            return {self.output_key: await self.async_fn(**kwargs)}

    @property
    def input_keys(self) -> InputKeys:
        """Input keys."""
        return InputKeys.from_keys(
            required_keys=self._req_params, optional_keys=self._opt_params
        )

    @property
    def output_keys(self) -> OutputKeys:
        """Output keys."""
        return OutputKeys.from_keys({self.output_key})


class InputComponent(QueryComponent):
    """Input component."""

    def _validate_component_inputs(self, input: Dict[str, Any]) -> Dict[str, Any]:
        return input

    def _validate_component_outputs(self, input: Dict[str, Any]) -> Dict[str, Any]:
        return input

    def validate_component_inputs(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Validate component inputs."""
        # NOTE: we override this to do nothing
        return input

    def validate_component_outputs(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """Validate component outputs."""
        # NOTE: we override this to do nothing
        return output

    def set_callback_manager(self, callback_manager: Any) -> None:
        """Set callback manager."""

    def _run_component(self, **kwargs: Any) -> Any:
        """Run component."""
        return kwargs

    async def _arun_component(self, **kwargs: Any) -> Any:
        """Run component (async)."""
        return self._run_component(**kwargs)

    @property
    def input_keys(self) -> InputKeys:
        """Input keys."""
        # NOTE: this shouldn't be used
        return InputKeys.from_keys(set(), optional_keys=set())
        # return InputComponentKeys.from_keys(set(), optional_keys=set())

    @property
    def output_keys(self) -> OutputKeys:
        """Output keys."""
        return OutputKeys.from_keys(set())


class ArgPackComponent(QueryComponent):
    """Arg pack component.

    Packs arbitrary number of args into a list.

    """

    convert_fn: Optional[Callable] = Field(
        default=None, description="Function to convert output."
    )

    def _validate_component_inputs(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Validate component inputs during run_component."""
        raise NotImplementedError

    def validate_component_inputs(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Validate component inputs."""
        return input

    def _validate_component_outputs(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """Validate component outputs."""
        # make sure output value is a list
        if not isinstance(output["output"], list):
            raise ValueError(f"Output is not a list.")
        return output

    def set_callback_manager(self, callback_manager: Any) -> None:
        """Set callback manager."""

    def _run_component(self, **kwargs: Any) -> Any:
        """Run component."""
        # combine all lists into one
        output = []
        for v in kwargs.values():
            if self.convert_fn is not None:
                v = self.convert_fn(v)
            output.append(v)
        return {"output": output}

    async def _arun_component(self, **kwargs: Any) -> Any:
        """Run component (async)."""
        return self._run_component(**kwargs)

    @property
    def input_keys(self) -> InputKeys:
        """Input keys."""
        # NOTE: this shouldn't be used
        return InputKeys.from_keys(set(), optional_keys=set())

    @property
    def output_keys(self) -> OutputKeys:
        """Output keys."""
        return OutputKeys.from_keys({"output"})


class KwargPackComponent(QueryComponent):
    """Kwarg pack component.

    Packs arbitrary number of kwargs into a dict.

    """

    convert_fn: Optional[Callable] = Field(
        default=None, description="Function to convert output."
    )

    def _validate_component_inputs(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Validate component inputs during run_component."""
        raise NotImplementedError

    def validate_component_inputs(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Validate component inputs."""
        return input

    def _validate_component_outputs(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """Validate component outputs."""
        # make sure output value is a list
        if not isinstance(output["output"], dict):
            raise ValueError(f"Output is not a dict.")
        return output

    def set_callback_manager(self, callback_manager: Any) -> None:
        """Set callback manager."""

    def _run_component(self, **kwargs: Any) -> Any:
        """Run component."""
        if self.convert_fn is not None:
            for k, v in kwargs.items():
                kwargs[k] = self.convert_fn(v)
        return {"output": kwargs}

    async def _arun_component(self, **kwargs: Any) -> Any:
        """Run component (async)."""
        return self._run_component(**kwargs)

    @property
    def input_keys(self) -> InputKeys:
        """Input keys."""
        # NOTE: this shouldn't be used
        return InputKeys.from_keys(set(), optional_keys=set())

    @property
    def output_keys(self) -> OutputKeys:
        """Output keys."""
        return OutputKeys.from_keys({"output"})
