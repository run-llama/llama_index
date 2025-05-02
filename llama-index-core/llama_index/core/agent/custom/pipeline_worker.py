"""Agent worker that takes in a query pipeline."""

import uuid
from typing import (
    Any,
    List,
    Optional,
    Sequence,
    cast,
)

from llama_index.core.agent.types import (
    BaseAgentWorker,
    Task,
    TaskStep,
    TaskStepOutput,
)
from llama_index.core.base.query_pipeline.query import QueryComponent
from llama_index.core.bridge.pydantic import BaseModel, Field, ConfigDict
from llama_index.core.callbacks import (
    CallbackManager,
    trace_method,
)
from llama_index.core.chat_engine.types import (
    AGENT_CHAT_RESPONSE_TYPE,
)
from llama_index.core.memory.chat_memory_buffer import ChatMemoryBuffer
from llama_index.core.query_pipeline.components.agent import (
    AgentInputComponent,
    BaseAgentComponent,
)
from llama_index.core.query_pipeline.query import QueryPipeline
from llama_index.core.tools import ToolOutput
from deprecated import deprecated

DEFAULT_MODEL_NAME = "gpt-3.5-turbo-0613"


def _get_agent_components(
    query_component: QueryComponent,
) -> Sequence[BaseAgentComponent]:
    """Get agent components."""
    agent_components: List[BaseAgentComponent] = []
    for c in query_component.sub_query_components:
        if isinstance(c, BaseAgentComponent):
            agent_components.append(cast(BaseAgentComponent, c))

        if len(c.sub_query_components) > 0:
            agent_components.extend(_get_agent_components(c))

    return agent_components


@deprecated("Use `FnAgentWorker` instead to build a stateful agent.")
class QueryPipelineAgentWorker(BaseModel, BaseAgentWorker):
    """
    Query Pipeline agent worker.

    NOTE: This is now deprecated. Use `FnAgentWorker` instead to build a stateful agent.

    Barebones agent worker that takes in a query pipeline.

    **Default Workflow**: The default workflow assumes that you compose
    a query pipeline with `StatefulFnComponent` objects. This allows you to store, update
    and retrieve state throughout the executions of the query pipeline by the agent.

    The task and step state of the agent are stored in this `state` variable via a special key.
    Of course you can choose to store other variables in this state as well.

    **Deprecated Workflow**: The deprecated workflow assumes that the first component in the
    query pipeline is an `AgentInputComponent` and last is `AgentFnComponent`.

    Args:
        pipeline (QueryPipeline): Query pipeline

    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    pipeline: QueryPipeline = Field(..., description="Query pipeline")
    callback_manager: CallbackManager = Field(..., exclude=True)
    task_key: str = Field("task", description="Key to store task in state")
    step_state_key: str = Field("step_state", description="Key to store step in state")

    def __init__(
        self,
        pipeline: QueryPipeline,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize."""
        if callback_manager is not None:
            # set query pipeline callback
            pipeline.set_callback_manager(callback_manager)
        else:
            callback_manager = pipeline.callback_manager
        super().__init__(
            pipeline=pipeline,
            callback_manager=callback_manager,
            **kwargs,
        )
        # validate query pipeline
        # self.agent_input_component
        self.agent_components

    @property
    def agent_input_component(self) -> AgentInputComponent:
        """
        Get agent input component.

        NOTE: This is deprecated and will be removed in the future.

        """
        root_key = self.pipeline.get_root_keys()[0]
        if not isinstance(self.pipeline.module_dict[root_key], AgentInputComponent):
            raise ValueError(
                "Query pipeline first component must be AgentInputComponent, got "
                f"{self.pipeline.module_dict[root_key]}"
            )

        return cast(AgentInputComponent, self.pipeline.module_dict[root_key])

    @property
    def agent_components(self) -> Sequence[BaseAgentComponent]:
        """Get agent output component."""
        return _get_agent_components(self.pipeline)

    def preprocess(self, task: Task, step: TaskStep) -> None:
        """
        Preprocessing flow.

        This runs preprocessing to propagate the task and step as variables
        to relevant components in the query pipeline.

        Contains deprecated flow of updating agent components.
        But also contains main flow of updating StatefulFnComponent components.

        """
        # NOTE: this is deprecated
        # partial agent output component with task and step
        for agent_fn_component in self.agent_components:
            agent_fn_component.partial(task=task, state=step.step_state)

        # update stateful components
        self.pipeline.update_state(
            {self.task_key: task, self.step_state_key: step.step_state}
        )

    def initialize_step(self, task: Task, **kwargs: Any) -> TaskStep:
        """Initialize step from task."""
        sources: List[ToolOutput] = []
        # temporary memory for new messages
        new_memory = ChatMemoryBuffer.from_defaults()

        # initialize initial state
        initial_state = {
            "sources": sources,
            "memory": new_memory,
        }

        return TaskStep(
            task_id=task.task_id,
            step_id=str(uuid.uuid4()),
            input=task.input,
            step_state=initial_state,
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

    @trace_method("run_step")
    def run_step(self, step: TaskStep, task: Task, **kwargs: Any) -> TaskStepOutput:
        """Run step."""
        self.preprocess(task, step)

        # HACK: do a try/except for now. Fine since old agent components are deprecated
        try:
            self.agent_input_component
            uses_deprecated = True
        except ValueError:
            uses_deprecated = False

        if uses_deprecated:
            agent_response, is_done = self.pipeline.run(
                state=step.step_state, task=task
            )
        else:
            agent_response, is_done = self.pipeline.run()
        response = self._get_task_step_response(agent_response, step, is_done)
        # sync step state with task state
        task.extra_state.update(step.step_state)
        return response

    @trace_method("run_step")
    async def arun_step(
        self, step: TaskStep, task: Task, **kwargs: Any
    ) -> TaskStepOutput:
        """Run step (async)."""
        self.preprocess(task, step)

        # HACK: do a try/except for now. Fine since old agent components are deprecated
        try:
            self.agent_input_component
            uses_deprecated = True
        except ValueError:
            uses_deprecated = False

        if uses_deprecated:
            agent_response, is_done = await self.pipeline.arun(
                state=step.step_state, task=task
            )
        else:
            agent_response, is_done = await self.pipeline.arun()
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
        # add new messages to memory
        task.memory.set(task.memory.get() + task.extra_state["memory"].get_all())
        # reset new memory
        task.extra_state["memory"].reset()

    def set_callback_manager(self, callback_manager: CallbackManager) -> None:
        """Set callback manager."""
        # TODO: make this abstractmethod (right now will break some agent impls)
        self.callback_manager = callback_manager
        self.pipeline.set_callback_manager(callback_manager)
