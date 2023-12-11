from llama_index.chat_engine.types import BaseChatEngine
from llama_index.core import BaseQueryEngine
from llama_index.agent.v1.schema import (
    TaskStep, 
    TaskStepOutput,
    TaskState,
    Task,
    AgentState,
    BaseAgentStepEngine
)
from llama_index.memory.types import BaseMemory
from llama_index.tools.types import BaseTool
from typing import List, Optional, Deque, Union, Any
from pydantic import BaseModel
from llama_index.schema import QueryBundle
from llama_index.response.schema import RESPONSE_TYPE, Response
from llama_index.llms.base import LLM, ChatMessage, ChatResponse, MessageRole
from llama_index.memory import BaseMemory, ChatMemoryBuffer
from llama_index.callbacks import (
    CallbackManager,
)


# def _add_initial_step(step_queue: Deque[TaskStep], task: Task):
#     """Add initial step."""
#     step_queue.append(
#         TaskStep(
#             task_id=task.task_id,
#             step_id=0,
#             input=task.input,
#             memory=task.memory,
#         )
#     )


class AgentEngine(BaseModel, BaseQueryEngine, BaseChatEngine):
    """Agent engine."""

    # TODO: implement this
    step_executor: BaseAgentStepEngine
	# stores tasks and their states
    state: AgentState
	# stores conversation history 
	# (currently assume agent engine handles a single message thread)
    memory: BaseMemory

    def __init__(
        self,
        step_executor: BaseAgentStepEngine,
        chat_history: Optional[List[ChatMessage]] = None,
        state: Optional[AgentState] = None,
        memory: Optional[BaseMemory] = None,
        llm: Optional[LLM] = None,
        callback_manager: Optional[CallbackManager] = None,
        system_prompt: Optional[str] = None,
        prefix_messages: Optional[List[ChatMessage]] = None,
        verbose: bool = False
    ) -> None:
        """Initialize."""
        memory = memory or ChatMemoryBuffer.from_defaults(chat_history, llm=llm)
        state = state or AgentState()
        callback_manager = callback_manager or CallbackManager([])
        super().__init__(
            step_executor=step_executor,
            state=state,
            memory=memory,
            callback_manager=callback_manager,
        )

    def create_task(
		self, 
	    input: str,
        **kwargs
	) -> Task:
        """Create task."""
        task = Task(
            input=input,
            step_queue=[],
            completed_steps=[],
            memory=self.memory,
            **kwargs
        )
        # add it to state
        self.state.task_dict[task.task_id] = task
        return task

    def run_step(
        self,
        task_id: str,
        step: Optional[TaskStep] = None,
        **kwargs: Any
    ) -> TaskStepOutput:
        """Execute step."""
        task = self.state.get_task(task_id)
        step = step or task.step_queue.popleft()

		# TODO: figure out if you can dynamically swap in different step executors
		# not clear when you would do that by theoretically possible		

        cur_step_output = self.step_executor.run_step(
			step, 
			task,
            **kwargs
		)
        # append cur_step_output next steps to queue
        next_steps = cur_step_output.next_steps
        task.step_queue.extend(next_steps)
        return cur_step_output

    def _query(self, query: QueryBundle) -> Response:
        """Run an e2e execution of a query."""
        task = self.create_task(query)

        initial_step = self.step_executor.initialize_step(task)
        task.step_queue.append(initial_step)

        result_output = None
        while True:
            # pass step queue in as argument, assume step executor is stateless
            cur_step_output = self.run_step(task.task_id)
            
            if cur_step_output.is_last:
                result_output = cur_step_output
                break
        return Response(response=str(result_output))
        
    def _chat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
        tool_choice: Union[str, dict] = "auto",
        mode: ChatResponseMode = ChatResponseMode.WAIT,
    ) -> AGENT_CHAT_RESPONSE_TYPE:
        """Chat."""
        raise NotImplementedError("chat not implemented")

    def undo_step(self, task_id: str) -> None:
        """Undo previous step."""
        raise NotImplementedError("undo_step not implemented")