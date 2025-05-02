"""Function components."""

from inspect import signature
from typing import Any, Callable, Dict, Optional, Set, Tuple
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


class FnComponent(QueryComponent):
    """Query component that takes in an arbitrary function."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    fn: AnnotatedCallable = Field(..., description="Function to run.")
    async_fn: Optional[AnnotatedCallable] = Field(
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
        super().__init__(fn=fn, async_fn=async_fn, output_key=output_key, **kwargs)
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


# alias
FunctionComponent = FnComponent
