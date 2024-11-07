from llama_index.core.base.query_pipeline.query import (
    InputKeys,
    OutputKeys,
    QueryComponent,
)
from llama_index.core.query_pipeline.query import QueryPipeline
from llama_index.core.bridge.pydantic import Field, ConfigDict, WithJsonSchema
from llama_index.core.callbacks.base import CallbackManager
from typing import Any, Dict, Optional, Callable
from typing_extensions import Annotated

AnnotatedCallable = Annotated[
    Callable,
    WithJsonSchema({"type": "string"}, mode="serialization"),
    WithJsonSchema({"type": "string"}, mode="validation"),
]


class LoopComponent(QueryComponent):
    """Loop component."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    pipeline: QueryPipeline = Field(..., description="Query pipeline")
    should_exit_fn: Optional[AnnotatedCallable] = Field(
        ..., description="Should exit function"
    )
    add_output_to_input_fn: Optional[AnnotatedCallable] = Field(
        ...,
        description="Add output to input function. If not provided, will reuse the original input for the next iteration. If provided, will call the function to combine the output into the input for the next iteration.",
    )
    max_iterations: int = Field(default=5, description="Max iterations")

    def __init__(
        self,
        pipeline: QueryPipeline,
        should_exit_fn: Optional[Callable] = None,
        add_output_to_input_fn: Optional[Callable] = None,
        max_iterations: Optional[int] = 5,
    ) -> None:
        """Init params."""
        super().__init__(
            pipeline=pipeline,
            should_exit_fn=should_exit_fn,
            add_output_to_input_fn=add_output_to_input_fn,
            max_iterations=max_iterations,
        )

    def set_callback_manager(self, callback_manager: CallbackManager) -> None:
        """Set callback manager."""
        # TODO: implement

    def _validate_component_inputs(self, input: Dict[str, Any]) -> Dict[str, Any]:
        return input

    def _run_component(self, **kwargs: Any) -> Dict:
        """Run component."""
        current_input = kwargs
        for _ in range(self.max_iterations):
            output = self.pipeline.run_component(**current_input)
            if self.should_exit_fn:
                should_exit = self.should_exit_fn(output)
                if should_exit:
                    break

            if self.add_output_to_input_fn:
                current_input = self.add_output_to_input_fn(current_input, output)

        return output

    async def _arun_component(self, **kwargs: Any) -> Any:
        """Run component (async)."""
        current_input = kwargs
        for _ in range(self.max_iterations):
            output = await self.pipeline.arun_component(**current_input)
            if self.should_exit_fn:
                should_exit = self.should_exit_fn(output)
                if should_exit:
                    break

            if self.add_output_to_input_fn:
                current_input = self.add_output_to_input_fn(current_input, output)

        return output

    @property
    def input_keys(self) -> InputKeys:
        """Input keys."""
        return self.pipeline.input_keys

    @property
    def output_keys(self) -> OutputKeys:
        """Output keys."""
        return self.pipeline.output_keys
