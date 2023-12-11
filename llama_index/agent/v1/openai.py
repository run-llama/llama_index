from llama_index.agent.v1.schema import BaseAgentStepEngine, TaskStep, TaskStepOutput
from pydantic import BaseModel, Field
from llama_index.llms.openai import OpenAI
from llama_index.llms.base import LLM
from typing import List, Dict, Any, Union
from llama_index.chat_engine.types import (
    AGENT_CHAT_RESPONSE_TYPE,
    AgentChatResponse,
    ChatResponseMode,
    StreamingAgentChatResponse,
)
from llama_index.tools import BaseTool
from llama_index.llms.openai_utils import OpenAIToolCall

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

    def _get_agent_response(
        self, mode: ChatResponseMode, **llm_chat_kwargs: Any
    ) -> AGENT_CHAT_RESPONSE_TYPE:
        if mode == ChatResponseMode.WAIT:
            chat_response: ChatResponse = self._llm.chat(**llm_chat_kwargs)
            return self._process_message(chat_response)
        elif mode == ChatResponseMode.STREAM:
            return self._get_stream_ai_response(**llm_chat_kwargs)
        else:
            raise NotImplementedError

    def _call_function(self, tools: List[BaseTool], tool_call: OpenAIToolCall) -> None:
        function_call = tool_call.function
        # validations to get passed mypy
        assert function_call is not None
        assert function_call.name is not None
        assert function_call.arguments is not None

        with self.callback_manager.event(
            CBEventType.FUNCTION_CALL,
            payload={
                EventPayload.FUNCTION_CALL: function_call.arguments,
                EventPayload.TOOL: get_function_by_name(
                    tools, function_call.name
                ).metadata,
            },
        ) as event:
            function_message, tool_output = call_function(
                tools, tool_call, verbose=self._verbose
            )
            event.on_end(payload={EventPayload.FUNCTION_OUTPUT: str(tool_output)})
        self.sources.append(tool_output)
        self.memory.put(function_message)
        
    # def _should_continue(
    #     self, tool_calls: Optional[List[OpenAIToolCall]], n_function_calls: int
    # ) -> bool:
    #     if n_function_calls > self._max_function_calls:
    #         return False
    #     if not tool_calls:
    #         return False
    #     return True

    def _run_step(self, task_step: TaskStep) -> TaskStepOutput:
        """Run step."""

        llm_chat_kwargs = self._get_llm_chat_kwargs(
            openai_tools, current_tool_choice
        )
        agent_chat_response = self._get_agent_response(mode=mode, **llm_chat_kwargs)

        # TODO: implement _should_continue
        if not self._should_continue(self.latest_tool_calls, n_function_calls):
            is_done = True
            # TODO: return response
        else:
            is_done = False
            for tool_call in self.latest_tool_calls:
                # Some validation
                if not isinstance(tool_call, get_args(OpenAIToolCall)):
                    raise ValueError("Invalid tool_call object")

                if tool_call.type != "function":
                    raise ValueError("Invalid tool type. Unsupported by OpenAI")
                # TODO: maybe execute this with multi-threading
                self._call_function(tools, tool_call)
                # change function call to the default value, if a custom function was given
                # as an argument (none and auto are predefined by OpenAI)
                if current_tool_choice not in ("auto", "none"):
                    current_tool_choice = "auto"
                n_function_calls += 1