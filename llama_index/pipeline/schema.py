"""Pipeline schema."""

from llama_index.schema import BaseComponent
from abc import abstractmethod
from typing import Any, Dict


class QueryComponent(BaseComponent):
    """Query component."""

    class Config:
        arbitrary_types_allowed = True

    @abstractmethod
    def _validate_component_inputs(self, **kwargs: Any) -> None:
        """Validate component inputs."""

    @abstractmethod
    def _validate_component_outputs(self, **kwargs: Any) -> None:
        """Validate component outputs."""

    def validate_component_inputs(self, **kwargs: Any) -> None:
        """Validate component inputs."""
        # make sure set of input keys == self.input_keys
        if set(kwargs.keys()) != set(self.input_keys):
            raise ValueError(
                f"Input keys do not match. Expected {self.input_keys}, got {set(kwargs.keys())}"
            )
        self._validate_component_inputs(**kwargs)

    def validate_component_outputs(self, **kwargs: Any) -> None:
        """Validate component outputs."""
        # make sure set of output keys == self.output_keys
        if set(kwargs.keys()) != set(self.output_keys):
            raise ValueError(
                f"Output keys do not match. Expected {self.output_keys}, got {set(kwargs.keys())}"
            )
        self._validate_component_outputs(**kwargs)

    def run_component(self, **kwargs: Any) -> Dict[str, Any]:
        """Run component."""
        self.validate_component_inputs(**kwargs)
        component_outputs = self._run_component(**kwargs)

    @abstractmethod
    def _run_component(self, **kwargs: Any) -> Any:
        """Run component."""

    @property
    @abstractmethod
    def input_keys(self) -> Any:
        """Input keys."""

    @property
    @abstractmethod
    def output_keys(self) -> Any:
        """Output keys.""" 
