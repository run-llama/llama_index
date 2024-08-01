"""Input components."""

from typing import Any, Dict

from llama_index.core.base.query_pipeline.query import (
    InputKeys,
    OutputKeys,
    QueryComponent,
)


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
