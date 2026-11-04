from types import SimpleNamespace

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.utilities.token_counting import TokenCounter


def _counter() -> TokenCounter:
    return TokenCounter(tokenizer=lambda text: list(str(text)))


def test_estimate_tokens_accepts_none_tool_arguments() -> None:
    tool_call = SimpleNamespace(
        function=SimpleNamespace(name="search", arguments=None)
    )
    message = ChatMessage(
        role=MessageRole.ASSISTANT,
        content="calling tool",
        additional_kwargs={"tool_calls": [tool_call]},
    )

    tokens = _counter().estimate_tokens_in_messages([message])
    assert tokens > 0


def test_estimate_tokens_accepts_dict_function_arguments() -> None:
    message = ChatMessage(
        role=MessageRole.ASSISTANT,
        content="calling fn",
        additional_kwargs={
            "function_call": {"name": "search", "arguments": {"q": "hello"}}
        },
    )

    tokens = _counter().estimate_tokens_in_messages([message])
    assert tokens > 0


def test_estimate_tokens_skips_non_mapping_function_call() -> None:
    message = ChatMessage(
        role=MessageRole.ASSISTANT,
        content="calling fn",
        additional_kwargs={"function_call": "search()"},
    )

    tokens = _counter().estimate_tokens_in_messages([message])
    assert tokens > 0
