from typing import Any, List, Optional, Sequence
from llama_index.core.agent.types import (
    BaseAgentWorker,
    Task,
    TaskStep,
)
from llama_index.core.memory.chat_memory_buffer import ChatMemoryBuffer
from llama_index.core.tools.types import BaseTool, ToolOutput
from llama_index.core.llms import LLM
from llama_index.core.callbacks import CallbackManager
import llama_index.core.instrumentation as instrument
import uuid

dispatcher = instrument.get_dispatcher(__name__)


class CriticAgentWorker(BaseAgentWorker):
    """CRITIC Agent Worker."""

    def __init__(
        self,
        tools: Sequence[BaseTool],
        llm: LLM,
        critique_prompt_template: str,
        critique_few_shot_examples: List[str],
        correct_prompt_template: str,
        critique_tools: Optional[Sequence[BaseTool]] = None,
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = False,
    ) -> None:
        self.llm = llm
        self.callback_manager = callback_manager or llm.callback_manager
        self.tools = tools
        self.critique_prompt_template = critique_prompt_template
        self.critique_few_shot_examples = critique_few_shot_examples
        self.correct_prompt_template = correct_prompt_template
        self.critique_tools = critique_tools
        self.verbose = verbose

    def initialize_step(self, task: Task, **kwargs: Any) -> TaskStep:
        """Initialize step from task."""
        sources: List[ToolOutput] = []
        # temporary memory for new messages
        new_memory = ChatMemoryBuffer.from_defaults()

        # put current history in new memory
        messages = task.memory.get()
        for message in messages:
            new_memory.put(message)

        # initialize task state
        task_state = {
            "sources": sources,
            "new_memory": new_memory,
        }
        task.extra_state.update(task_state)

        return TaskStep(
            task_id=task.task_id,
            step_id=str(uuid.uuid4()),
            input=task.input,
            step_state={"prev_reasoning": ""},
        )
