"""Arg pack components."""

from typing import Any, Callable, Dict, Optional

from llama_index.core.base.query_pipeline.query import (
    InputKeys,
    OutputKeys,
    QueryComponent,
)
from llama_index.core.bridge.pydantic import Field


class ArgPackComponent(QueryComponent):
    """
    Arg pack component.

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

            if isinstance(v, list):
                output.extend(v)
            else:
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
    """
    Kwarg pack component.

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
