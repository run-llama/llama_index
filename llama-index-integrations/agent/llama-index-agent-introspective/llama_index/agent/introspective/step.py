"""Introspective agent worker."""

import logging
import uuid
from typing import Any, List, Optional
from llama_index.core.agent.types import (
    BaseAgentWorker,
    Task,
    TaskStep,
    TaskStepOutput,
)
from llama_index.core.agent.utils import add_user_step_to_memory
from llama_index.core.callbacks import (
    CallbackManager,
    trace_method,
)
from llama_index.core.chat_engine.types import (
    AgentChatResponse,
)
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.memory import ChatMemoryBuffer

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class IntrospectiveAgentWorker(BaseAgentWorker):
    """
    Introspective Agent Worker.

    This agent worker implements the Reflection AI agentic pattern. It does
    so by merely delegating the work to two other agents in a purely
    deterministic fashion.

    The task this agent performs (again via delegation) is to generate a response
    to a query and perform reflection and correction on the response. This
    agent delegates the task to (optionally) first a `main_agent_worker` that
    generates the initial response to the query. This initial response is then
    passed to the `reflective_agent_worker` to perform the reflection and
    correction of the initial response. Optionally, the `main_agent_worker`
    can be skipped if none is provided. In this case, the users input query
    will be assumed to contain the original response that needs to go thru
    reflection and correction.

    Attributes:
        reflective_agent_worker (BaseAgentWorker): Reflective agent responsible for
            performing reflection and correction of the initial response.
        main_agent_worker (Optional[BaseAgentWorker], optional): Main agent responsible
            for generating an initial response to the user query. Defaults to None.
            If None, the user input is assumed as the initial response.
        verbose (bool, optional): Whether execution should be verbose. Defaults to False.
        callback_manager (Optional[CallbackManager], optional): Callback manager. Defaults to None.

    """

    def __init__(
        self,
        reflective_agent_worker: BaseAgentWorker,
        main_agent_worker: Optional[BaseAgentWorker] = None,
        verbose: bool = False,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        """Init params."""
        self._verbose = verbose
        self._main_agent_worker = main_agent_worker
        self._reflective_agent_worker = reflective_agent_worker
        self.callback_manager = callback_manager or CallbackManager([])

    @classmethod
    def from_defaults(
        cls,
        reflective_agent_worker: BaseAgentWorker,
        main_agent_worker: Optional[BaseAgentWorker] = None,
        verbose: bool = False,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ) -> "IntrospectiveAgentWorker":
        """
        Create an IntrospectiveAgentWorker from args.

        Similar to `from_defaults` in other classes, this method will
        infer defaults for a variety of parameters, including the LLM,
        if they are not specified.
        """
        return cls(
            main_agent_worker=main_agent_worker,
            reflective_agent_worker=reflective_agent_worker,
            verbose=verbose,
            callback_manager=callback_manager,
            **kwargs,
        )

    def initialize_step(self, task: Task, **kwargs: Any) -> TaskStep:
        """Initialize step from task."""
        # temporary memory for new messages
        main_memory = ChatMemoryBuffer.from_defaults()
        reflective_memory = ChatMemoryBuffer.from_defaults()

        # put current history in new memory
        messages = task.memory.get(input=task.input)
        for message in messages:
            main_memory.put(message)

        # initialize task state
        task_state = {
            "main": {
                "memory": main_memory,
                "sources": [],
            },
            "reflection": {"memory": reflective_memory, "sources": []},
        }
        task.extra_state.update(task_state)

        return TaskStep(
            task_id=task.task_id,
            step_id=str(uuid.uuid4()),
            input=task.input,
        )

    def get_all_messages(self, task: Task) -> List[ChatMessage]:
        return (
            +task.memory.get(input=task.input)
            + task.extra_state["main"]["memory"].get_all()
            + task.extra_state["reflection"]["memory"].get_all()
        )

    @trace_method("run_step")
    def run_step(self, step: TaskStep, task: Task, **kwargs: Any) -> TaskStepOutput:
        """Run step."""
        # run main agent
        if self._main_agent_worker is not None:
            main_agent_messages = task.extra_state["main"]["memory"].get()
            main_agent = self._main_agent_worker.as_agent(
                chat_history=main_agent_messages
            )
            main_agent_response = main_agent.chat(task.input)
            original_response = main_agent_response.response
            task.extra_state["main"]["sources"] = main_agent_response.sources
            task.extra_state["main"]["memory"] = main_agent.memory
        else:
            pass

        # run reflective agent
        reflective_agent_messages = task.extra_state["main"]["memory"].get()
        reflective_agent = self._reflective_agent_worker.as_agent(
            chat_history=reflective_agent_messages
        )
        # NOTE: atm you *need* to pass an input string to `chat`, even if the memory is already
        # preloaded. Input will be concatenated on top of chat history from memory
        # which will be used to generate the response.
        # TODO: make agent interface more flexible
        reflective_agent_response = reflective_agent.chat(original_response)
        task.extra_state["reflection"]["sources"] = reflective_agent_response.sources
        task.extra_state["reflection"]["memory"] = reflective_agent.memory

        agent_response = AgentChatResponse(
            response=str(reflective_agent_response.response),
            sources=task.extra_state["main"]["sources"]
            + task.extra_state["reflection"]["sources"],
        )

        return TaskStepOutput(
            output=agent_response,
            task_step=step,
            is_last=True,
            next_steps=[],
        )

    @trace_method("run_step")
    async def arun_step(
        self, step: TaskStep, task: Task, **kwargs: Any
    ) -> TaskStepOutput:
        """Run step (async)."""
        # run main agent if one is supplied otherwise assume user input
        # is the original response to be reflected on and subsequently corrected
        if self._main_agent_worker is not None:
            main_agent_messages = task.extra_state["main"]["memory"].get()
            main_agent = self._main_agent_worker.as_agent(
                chat_history=main_agent_messages, verbose=self._verbose
            )
            main_agent_response = await main_agent.achat(task.input)
            original_response = main_agent_response.response
            task.extra_state["main"]["sources"] = main_agent_response.sources
            task.extra_state["main"]["memory"] = main_agent.memory
        else:
            add_user_step_to_memory(
                step, task.extra_state["main"]["memory"], verbose=self._verbose
            )
            original_response = step.input
            # fictitious agent's initial response (to get reflection/correction cycle started)
            task.extra_state["main"]["memory"].put(
                ChatMessage(content=original_response, role="assistant")
            )

        # run reflective agent
        reflective_agent_messages = task.extra_state["main"]["memory"].get()
        reflective_agent = self._reflective_agent_worker.as_agent(
            chat_history=reflective_agent_messages, verbose=self._verbose
        )
        reflective_agent_response = await reflective_agent.achat(original_response)
        task.extra_state["reflection"]["sources"] = reflective_agent_response.sources
        task.extra_state["reflection"]["memory"] = reflective_agent.memory

        agent_response = AgentChatResponse(
            response=str(reflective_agent_response.response),
            sources=task.extra_state["main"]["sources"]
            + task.extra_state["reflection"]["sources"],
        )

        return TaskStepOutput(
            output=agent_response,
            task_step=step,
            is_last=True,
            next_steps=[],
        )

    @trace_method("run_step")
    def stream_step(self, step: TaskStep, task: Task, **kwargs: Any) -> TaskStepOutput:
        """Run step (stream)."""
        raise NotImplementedError("Stream not supported for introspective agent")

    @trace_method("run_step")
    async def astream_step(
        self, step: TaskStep, task: Task, **kwargs: Any
    ) -> TaskStepOutput:
        """Run step (async stream)."""
        raise NotImplementedError("Stream not supported for introspective agent")

    def finalize_task(self, task: Task, **kwargs: Any) -> None:
        """Finalize task, after all the steps are completed."""
        # add new messages to memory
        main_memory = task.extra_state["main"][
            "memory"
        ].get_all()  # contains initial response as final message
        final_corrected_message = task.extra_state["reflection"]["memory"].get_all()[-1]
        # swap main workers response with the reflected/corrected one
        finalized_task_memory = main_memory[:-1] + [final_corrected_message]
        task.memory.set(finalized_task_memory)
