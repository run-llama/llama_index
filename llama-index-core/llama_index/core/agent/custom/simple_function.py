"""Custom function agent worker."""

import uuid
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Tuple,
)

from llama_index.core.agent.types import (
    BaseAgentWorker,
    Task,
    TaskStep,
    TaskStepOutput,
)
from llama_index.core.bridge.pydantic import BaseModel, Field, ConfigDict
from llama_index.core.callbacks import (
    CallbackManager,
    trace_method,
)
from llama_index.core.chat_engine.types import (
    AGENT_CHAT_RESPONSE_TYPE,
    AgentChatResponse,
)
from llama_index.core.query_pipeline.components.function import get_parameters


class FnAgentWorker(BaseModel, BaseAgentWorker):
    """Function Agent Worker.

    Define an agent worker over a stateful function (takes in a `state` variable).
    The stateful function expects a tuple of (`AgentChatResponse`, bool) as the response.

    Subclass this to define your own agent worker.

    Args:
        fn (Callable): The function to use. Must contain a `state` dictionary.
        initial_state (Dict[str, Any]): The initial state

    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    fn: Callable = Field(..., description="Function to run.")
    async_fn: Optional[Callable] = Field(
        None, description="Async function to run. If not provided, will run `fn`."
    )
    initial_state: Dict[str, Any] = Field(
        default_factory=dict, description="Initial state dictionary."
    )
    task_input_key: str = Field(default="__task__", description="Task")
    output_key: str = Field(default="__output__", description="output")

    verbose: bool = Field(False, description="Verbose mode.")

    def __init__(
        self,
        fn: Callable,
        async_fn: Optional[Callable] = None,
        initial_state: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        # determine parameters
        default_req_params, default_opt_params = get_parameters(fn)
        # make sure task and step are part of the list, and remove them from the list
        if "state" not in default_req_params:
            raise ValueError(
                "StatefulFnComponent must have 'state' as required parameters"
            )

        super().__init__(fn=fn, async_fn=async_fn, initial_state=initial_state)

    def initialize_step(self, task: Task, **kwargs: Any) -> TaskStep:
        """Initialize step from task."""
        step_state = {
            **self.initial_state,
            self.task_input_key: task,
        }

        return TaskStep(
            task_id=task.task_id,
            step_id=str(uuid.uuid4()),
            input=task.input,
            step_state=step_state,
        )

    def _get_task_step_response(
        self, agent_response: AGENT_CHAT_RESPONSE_TYPE, step: TaskStep, is_done: bool
    ) -> TaskStepOutput:
        """Get task step response."""
        if is_done:
            new_steps = []
        else:
            new_steps = [
                step.get_next_step(
                    step_id=str(uuid.uuid4()),
                    # NOTE: input is unused
                    input=None,
                )
            ]

        return TaskStepOutput(
            output=agent_response,
            task_step=step,
            is_last=is_done,
            next_steps=new_steps,
        )

    def _run_step(
        self, state: Dict[str, Any], task: Task, input: Optional[str] = None
    ) -> Tuple[AgentChatResponse, bool]:
        """Run step.

        Returns:
            Tuple of (agent_response, is_done)

        """
        current_state, is_done = self.fn(state)
        # TODO: return auxiliary response
        output = state[self.output_key] if self.output_key in state else ""
        return (
            AgentChatResponse(response=output, metadata=current_state),
            is_done,
        )

    async def _arun_step(
        self, state: Dict[str, Any], task: Task, input: Optional[str] = None
    ) -> Tuple[AgentChatResponse, bool]:
        """Run step (async).

        Can override this method if you want to run the step asynchronously.

        Returns:
            Tuple of (agent_response, is_done)

        """
        if self.async_fn is None:
            current_state, is_done = self.fn(state)
        else:
            current_state, is_done = await self.async_fn(state)
        # TODO: return auxiliary response
        return (
            AgentChatResponse(response=state[self.output_key], metadata=current_state),
            is_done,
        )

    @trace_method("run_step")
    def run_step(self, step: TaskStep, task: Task, **kwargs: Any) -> TaskStepOutput:
        """Run step."""
        agent_response, is_done = self._run_step(
            step.step_state, task, input=step.input
        )
        response = self._get_task_step_response(agent_response, step, is_done)
        # sync step state with task state
        task.extra_state.update(step.step_state)
        return response

    @trace_method("run_step")
    async def arun_step(
        self, step: TaskStep, task: Task, **kwargs: Any
    ) -> TaskStepOutput:
        """Run step (async)."""
        agent_response, is_done = await self._arun_step(
            step.step_state, task, input=step.input
        )
        response = self._get_task_step_response(agent_response, step, is_done)
        task.extra_state.update(step.step_state)
        return response

    @trace_method("run_step")
    def stream_step(self, step: TaskStep, task: Task, **kwargs: Any) -> TaskStepOutput:
        """Run step (stream)."""
        raise NotImplementedError("This agent does not support streaming.")

    @trace_method("run_step")
    async def astream_step(
        self, step: TaskStep, task: Task, **kwargs: Any
    ) -> TaskStepOutput:
        """Run step (async stream)."""
        raise NotImplementedError("This agent does not support streaming.")

    def finalize_task(self, task: Task, **kwargs: Any) -> None:
        """Finalize task, after all the steps are completed."""

    def set_callback_manager(self, callback_manager: CallbackManager) -> None:
        """Set callback manager."""
