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
from typing import List, Optional, Deque, Union
from pydantic import BaseModel
from llama_index.schema import QueryBundle
from llama_index.response.schema import RESPONSE_TYPE, Response
from llama_index.llms.base import LLM, ChatMessage, ChatResponse, MessageRole


def _add_initial_step(step_queue: Deque[TaskStep], task: Task):
    """Add initial step."""
    step_queue.append(
        TaskStep(
            task_id=task.task_id,
            step_id=0,
            input=task.input,
            memory=task.memory,
        )
    )


class AgentEngine(BaseModel, BaseQueryEngine, BaseChatEngine):
    """Agent engine."""

    # TODO: implement this
    step_executor: BaseAgentStepEngine
	# stores tasks and their states
    state: AgentState
	# stores conversation history 
	# (currently assume agent engine handles a single message thread)
    memory: BaseMemory
	# other kwargs (tools, etc.)
    tools: List[BaseTool] 

    def create_task(
		self, 
	    input: str,
        **kwargs
	) -> Task:
        """Create task."""
        return Task(
            input=input,
            **kwargs
        )

    def run_step(
        self,
        task_id: str,
        step: Optional[TaskStep] = None,
        input: Optional[str] = None,
    ) -> TaskStepOutput:
        """Execute step."""
        step = step or self.state.get_step_queue(task_id)[0]

		# TODO: figure out if you can dynamically swap in different step executors
		# not clear when you would do that by theoretically possible		

        cur_step_output = self.step_executor.run(
			step, 
			self.state.get_step_queue(task_id)
		)
        return cur_step_output

    def _query(self, query: QueryBundle) -> Response:
        """Run an e2e execution of a query."""
        task = self.create_task(query)

        _add_initial_step(self.state.step_queue, task)

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