"""Pipeline schema."""

from llama_index.schema import BaseComponent
from llama_index.bridge.pydantic import Field, PrivateAttr, BaseModel, validator
from abc import abstractmethod
from typing import Any, Dict, Set, Optional, Union
# TODO: fix circular dependency risk
from llama_index.core.llms.types import CompletionResponse, ChatResponse
from llama_index.schema import QueryBundle
from llama_index.core.response.schema import Response

## Define common types used throughout these components
StringableInput = Union[CompletionResponse, ChatResponse, str, QueryBundle, Response]

def validate_and_convert_stringable(input: Any) -> str:
    """Validate and convert stringable input."""
    if not isinstance(input, StringableInput):
        raise ValueError(f"Input {input} is not stringable.")
    return str(input)


class InputKeys(BaseModel):
    """Input keys."""
    required_keys: Set[str] = Field(default_factory=set)
    optional_keys: Set[str] = Field(default_factory=set)

    @classmethod
    def from_keys(
        cls, 
        required_keys: Set[str], 
        optional_keys: Optional[Set[str]] = None
    ) -> "InputKeys":
        """Create InputKeys from tuple."""
        return cls(required_keys=required_keys, optional_keys=optional_keys or set())

    def validate(self, input_keys: Set[str]) -> None:
        """Validate input keys."""
        # check if required keys are present, and that keys all are in required or optional
        if not self.required_keys.issubset(input_keys):
            raise ValueError(
                f"Required keys {self.required_keys} are not present in input keys {input_keys}"
            )
        if not input_keys.issubset(self.required_keys.union(self.optional_keys)):
            raise ValueError(
                f"Input keys {input_keys} contain keys not in required or optional keys {self.required_keys.union(self.optional_keys)}"
            )


class OutputKeys(BaseModel):
    """Output keys."""
    required_keys: Set[str] = Field(default_factory=set)

    @classmethod
    def from_keys(
        cls, 
        required_keys: Set[str], 
    ) -> "InputKeys":
        """Create InputKeys from tuple."""
        return cls(required_keys=required_keys)

    def validate(self, input_keys: Set[str]) -> None:
        """Validate input keys."""
        # validate that input keys exactly match required keys
        if input_keys != self.required_keys:
            raise ValueError(
                f"Input keys {input_keys} do not match required keys {self.required_keys}"
            )

class QueryComponent(BaseComponent):
    """Query component.

    Represents a component that can be run in a `QueryPipeline`.
    
    """

    class Config:
        arbitrary_types_allowed = True

    @abstractmethod
    def _validate_component_inputs(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Validate component inputs during run_component."""

    @abstractmethod
    def _validate_component_outputs(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """Validate component outputs during run_component."""

    def validate_component_inputs(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Validate component inputs."""
        # make sure set of input keys == self.input_keys
        if set(input.keys()) != set(self.input_keys):
            raise ValueError(
                f"Input keys do not match. Expected {self.input_keys}, got {set(input.keys())}"
            )
        return self._validate_component_inputs(input)

    def validate_component_outputs(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """Validate component outputs."""
        # make sure set of output keys == self.output_keys
        if set(output.keys()) != set(self.output_keys):
            raise ValueError(
                f"Output keys do not match. Expected {self.output_keys}, got {set(output.keys())}"
            )
        return self._validate_component_outputs(output)

    def run_component(self, **kwargs: Any) -> Dict[str, Any]:
        """Run component."""
        kwargs = self.validate_component_inputs(kwargs)
        component_outputs = self._run_component(**kwargs)
        return self.validate_component_outputs(component_outputs)

    async def arun_component(self, **kwargs: Any) -> Dict[str, Any]:
        """Run component."""
        kwargs = self.validate_component_inputs(kwargs)
        component_outputs = await self._arun_component(**kwargs)
        return self.validate_component_outputs(component_outputs)

    @abstractmethod
    def _run_component(self, **kwargs: Any) -> Dict:
        """Run component."""

    async def _arun_component(self, **kwargs: Any) -> Any:
        """Run component (async)."""
        raise NotImplementedError("This component does not support async run.")

    @property
    @abstractmethod
    def input_keys(self) -> InputKeys:
        """Input keys."""

    @property
    @abstractmethod
    def output_keys(self) -> OutputKeys:
        """Output keys.""" 
