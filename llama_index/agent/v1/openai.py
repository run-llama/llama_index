from llama_index.agent.v1.schema import BaseAgentStepEngine, TaskStep, TaskStepOutput
from pydantic import BaseModel, Field
from llama_index.llms.openai import OpenAI
from llama_index.llms.base import LLM
from typing import List, Dict, Any, Union

class OpenAIAgentStepEngine(BaseAgentStepEngine):
    """OpenAI Agent step engine."""

    def __init__(
        self,
        llm: OpenAI,
        memory: BaseMemory,
        prefix_messages: List[ChatMessage],
        verbose: bool,
        max_function_calls: int,
        callback_manager: Optional[CallbackManager],
    ):
        self._llm = llm
        self._verbose = verbose
        self._max_function_calls = max_function_calls
        self.prefix_messages = prefix_messages
        self.memory = memory
        self.callback_manager = callback_manager or self._llm.callback_manager
        self.sources: List[ToolOutput] = []

    def _get_llm_chat_kwargs(
        self, openai_tools: List[dict], tool_choice: Union[str, dict] = "auto"
    ) -> Dict[str, Any]:
        llm_chat_kwargs: dict = {"messages": self.all_messages}
        if openai_tools:
            llm_chat_kwargs.update(
                tools=openai_tools, tool_choice=resolve_tool_choice(tool_choice)
            )
        return llm_chat_kwargs

    def _run_step(self, task_step: TaskStep) -> TaskStepOutput:
        """Run step."""

        llm_chat_kwargs = self._get_llm_chat_kwargs(
            openai_tools, current_tool_choice
        )
