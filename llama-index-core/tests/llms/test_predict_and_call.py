from llama_index.core.llms.mock import MockLLM
from llama_index.core.base.llms.types import ChatMessage, MessageRole, ChatResponse
from llama_index.core.program.function_program import FunctionTool
from llama_index.core.agent.react.formatter import ReActChatFormatter
from llama_index.core.agent.react.output_parser import ReActOutputParser


def tool(*args, **kwargs) -> str:
    return "hello!!"


class _ReActDrivingLLM(MockLLM):
    async def achat(
        self, messages: list[ChatMessage], **kwargs: object
    ) -> ChatResponse:
        return ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT,
                content="Thought: do it\nAction: tool\nAction Input: {}\n",
            ),
            raw={"content": "react"},
        )


def test_predict_and_call_via_react_agent() -> None:
    """Ensure tool is called via ReAct-style action parsing."""
    llm = _ReActDrivingLLM()
    response = llm.predict_and_call(
        tools=[FunctionTool.from_defaults(fn=tool)],
        react_chat_formatter=ReActChatFormatter.from_defaults(),
        output_parser=ReActOutputParser(),
        user_msg=ChatMessage(role=MessageRole.USER, content="run tool"),
        chat_history=[],
    )
    assert response.response == "hello!!"
    assert len(response.sources) == 1
    assert response.sources[0].content == "hello!!"
