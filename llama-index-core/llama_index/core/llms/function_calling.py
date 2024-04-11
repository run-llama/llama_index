from typing import (
    Any,
    List,
    Optional,
    Union,
    TYPE_CHECKING,
)
import asyncio

from llama_index.core.base.llms.types import (
    ChatMessage,
)
from llama_index.core.llms.llm import LLM

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
)
from llama_index.core.llms.llm import ToolSelection

if TYPE_CHECKING:
    from llama_index.core.chat_engine.types import AgentChatResponse
    from llama_index.core.tools.types import BaseTool


class FunctionCallingLLM(LLM):
    """
    Function calling LLMs are LLMs that support function calling.
    They support an expanded range of capabilities.

    """

    def chat_with_tools(
        self,
        tools: List["BaseTool"],
        user_msg: Optional[Union[str, ChatMessage]] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        **kwargs: Any,
    ) -> ChatResponse:
        """Predict and call the tool."""
        raise NotImplementedError("chat_with_tools is not supported by default.")

    async def achat_with_tools(
        self,
        tools: List["BaseTool"],
        user_msg: Optional[Union[str, ChatMessage]] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        **kwargs: Any,
    ) -> ChatResponse:
        """Predict and call the tool."""
        raise NotImplementedError("achat_with_tools is not supported by default.")

    def get_tool_calls_from_response(
        self,
        response: "AgentChatResponse",
        error_on_no_tool_call: bool = True,
        **kwargs: Any,
    ) -> List[ToolSelection]:
        """Predict and call the tool."""
        raise NotImplementedError(
            "get_tool_calls_from_response is not supported by default."
        )

    def predict_and_call(
        self,
        tools: List["BaseTool"],
        user_msg: Optional[Union[str, ChatMessage]] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        **kwargs: Any,
    ) -> "AgentChatResponse":
        """Predict and call the tool."""
        from llama_index.core.chat_engine.types import AgentChatResponse
        from llama_index.core.tools.calling import (
            call_tool_with_selection,
        )

        if not self.metadata.is_function_calling_model:
            return super().predict_and_call(
                tools,
                user_msg=user_msg,
                chat_history=chat_history,
                verbose=verbose,
                **kwargs,
            )

        response = self.chat_with_tools(
            tools,
            user_msg,
            chat_history=chat_history,
            verbose=verbose,
            allow_parallel_tool_calls=allow_parallel_tool_calls,
            **kwargs,
        )
        tool_calls = self.get_tool_calls_from_response(response)
        tool_outputs = [
            call_tool_with_selection(tool_call, tools, verbose=verbose)
            for tool_call in tool_calls
        ]
        if allow_parallel_tool_calls:
            output_text = "\n\n".join(
                [tool_output.content for tool_output in tool_outputs]
            )
            return AgentChatResponse(response=output_text, sources=tool_outputs)
        else:
            if len(tool_outputs) > 1:
                raise ValueError("Invalid")
            return AgentChatResponse(
                response=tool_outputs[0].content, sources=tool_outputs
            )

    async def apredict_and_call(
        self,
        tools: List["BaseTool"],
        user_msg: Optional[Union[str, ChatMessage]] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        **kwargs: Any,
    ) -> "AgentChatResponse":
        """Predict and call the tool."""
        from llama_index.core.tools.calling import (
            acall_tool_with_selection,
        )
        from llama_index.core.chat_engine.types import AgentChatResponse

        if not self.metadata.is_function_calling_model:
            return await super().apredict_and_call(
                tools,
                user_msg=user_msg,
                chat_history=chat_history,
                verbose=verbose,
                **kwargs,
            )

        response = await self.achat_with_tools(
            tools,
            user_msg,
            chat_history=chat_history,
            verbose=verbose,
            allow_parallel_tool_calls=allow_parallel_tool_calls,
            **kwargs,
        )
        tool_calls = self.get_tool_calls_from_response(response)
        tool_tasks = [
            acall_tool_with_selection(tool_call, tools, verbose=verbose)
            for tool_call in tool_calls
        ]
        tool_outputs = await asyncio.gather(*tool_tasks)
        if allow_parallel_tool_calls:
            output_text = "\n\n".join(
                [tool_output.content for tool_output in tool_outputs]
            )
            return AgentChatResponse(response=output_text, sources=tool_outputs)
        else:
            if len(tool_outputs) > 1:
                raise ValueError("Invalid")
            return AgentChatResponse(
                response=tool_outputs[0].content, sources=tool_outputs
            )
