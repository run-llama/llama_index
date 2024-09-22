from typing import Any, Dict, List, Optional, Sequence, Union, TYPE_CHECKING
from abc import abstractmethod
import asyncio

from llama_index.core.base.llms.types import (
    ChatMessage,
)
from llama_index.core.llms.llm import LLM

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    ChatResponseAsyncGen,
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
        tools: Sequence["BaseTool"],
        user_msg: Optional[Union[str, ChatMessage]] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        **kwargs: Any,
    ) -> ChatResponse:
        """Chat with function calling."""
        chat_kwargs = self._prepare_chat_with_tools(
            tools,
            user_msg=user_msg,
            chat_history=chat_history,
            verbose=verbose,
            allow_parallel_tool_calls=allow_parallel_tool_calls,
            **kwargs,
        )
        response = self.chat(**chat_kwargs)
        return self._validate_chat_with_tools_response(
            response,
            tools,
            allow_parallel_tool_calls=allow_parallel_tool_calls,
            **kwargs,
        )

    async def achat_with_tools(
        self,
        tools: Sequence["BaseTool"],
        user_msg: Optional[Union[str, ChatMessage]] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        **kwargs: Any,
    ) -> ChatResponse:
        """Async chat with function calling."""
        chat_kwargs = self._prepare_chat_with_tools(
            tools,
            user_msg=user_msg,
            chat_history=chat_history,
            verbose=verbose,
            allow_parallel_tool_calls=allow_parallel_tool_calls,
            **kwargs,
        )
        response = await self.achat(**chat_kwargs)
        return self._validate_chat_with_tools_response(
            response,
            tools,
            allow_parallel_tool_calls=allow_parallel_tool_calls,
            **kwargs,
        )

    def stream_chat_with_tools(
        self,
        tools: Sequence["BaseTool"],
        user_msg: Optional[Union[str, ChatMessage]] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        **kwargs: Any,
    ) -> ChatResponseGen:
        """Stream chat with function calling."""
        chat_kwargs = self._prepare_chat_with_tools(
            tools,
            user_msg=user_msg,
            chat_history=chat_history,
            verbose=verbose,
            allow_parallel_tool_calls=allow_parallel_tool_calls,
            **kwargs,
        )
        # TODO: no validation for streaming outputs
        return self.stream_chat(**chat_kwargs)

    async def astream_chat_with_tools(
        self,
        tools: Sequence["BaseTool"],
        user_msg: Optional[Union[str, ChatMessage]] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        **kwargs: Any,
    ) -> ChatResponseAsyncGen:
        """Async stream chat with function calling."""
        chat_kwargs = self._prepare_chat_with_tools(
            tools,
            user_msg=user_msg,
            chat_history=chat_history,
            verbose=verbose,
            allow_parallel_tool_calls=allow_parallel_tool_calls,
            **kwargs,
        )
        # TODO: no validation for streaming outputs
        return await self.astream_chat(**chat_kwargs)

    @abstractmethod
    def _prepare_chat_with_tools(
        self,
        tools: Sequence["BaseTool"],
        user_msg: Optional[Union[str, ChatMessage]] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Prepare the arguments needed to let the LLM chat with tools."""

    def _validate_chat_with_tools_response(
        self,
        response: ChatResponse,
        tools: Sequence["BaseTool"],
        allow_parallel_tool_calls: bool = False,
        **kwargs: Any,
    ) -> ChatResponse:
        """Validate the response from chat_with_tools."""
        return response

    def get_tool_calls_from_response(
        self,
        response: ChatResponse,
        error_on_no_tool_call: bool = True,
        **kwargs: Any,
    ) -> List[ToolSelection]:
        """Predict and call the tool."""
        raise NotImplementedError(
            "get_tool_calls_from_response is not supported by default."
        )

    def predict_and_call(
        self,
        tools: Sequence["BaseTool"],
        user_msg: Optional[Union[str, ChatMessage]] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        error_on_no_tool_call: bool = True,
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
            user_msg=user_msg,
            chat_history=chat_history,
            verbose=verbose,
            allow_parallel_tool_calls=allow_parallel_tool_calls,
            **kwargs,
        )
        tool_calls = self.get_tool_calls_from_response(
            response, error_on_no_tool_call=error_on_no_tool_call
        )
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
            elif len(tool_outputs) == 0:
                return AgentChatResponse(
                    response=response.message.content or "", sources=tool_outputs
                )

            return AgentChatResponse(
                response=tool_outputs[0].content, sources=tool_outputs
            )

    async def apredict_and_call(
        self,
        tools: Sequence["BaseTool"],
        user_msg: Optional[Union[str, ChatMessage]] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        error_on_no_tool_call: bool = True,
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
            user_msg=user_msg,
            chat_history=chat_history,
            verbose=verbose,
            allow_parallel_tool_calls=allow_parallel_tool_calls,
            **kwargs,
        )

        tool_calls = self.get_tool_calls_from_response(
            response, error_on_no_tool_call=error_on_no_tool_call
        )
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
            elif len(tool_outputs) == 0:
                return AgentChatResponse(
                    response=response.message.content or "", sources=tool_outputs
                )

            return AgentChatResponse(
                response=tool_outputs[0].content, sources=tool_outputs
            )
