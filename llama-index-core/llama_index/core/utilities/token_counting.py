# Modified from:
# https://github.com/nyno-ai/openai-token-counter

from typing import Any, Callable, Dict, List, Optional

from llama_index.core.base.llms.types import ChatMessage, MessageRole, ToolCallBlock
from llama_index.core.utils import get_tokenizer


def _has_tool_call_blocks(message: ChatMessage) -> bool:
    """Whether the message already carries its tool calls as ToolCallBlocks."""
    return any(isinstance(block, ToolCallBlock) for block in message.blocks)


class TokenCounter:
    """
    Token counter class.

    Attributes:
        model (Optional[str]): The model to use for token counting.

    """

    def __init__(self, tokenizer: Optional[Callable[[str], list]] = None) -> None:
        self.tokenizer = tokenizer or get_tokenizer()

    def get_string_tokens(self, string: str) -> int:
        """
        Get the token count for a string.

        Args:
            string (str): The string to count.

        Returns:
            int: The token count.

        """
        return len(self.tokenizer(string))

    def estimate_tokens_in_messages(self, messages: List[ChatMessage]) -> int:
        """
        Estimate token count for a single message.

        Args:
            message (OpenAIMessage): The message to estimate the token count for.

        Returns:
            int: The estimated token count.

        """
        tokens = 0

        for message in messages:
            if message.role:
                tokens += self.get_string_tokens(message.role)

            tokens += message.estimate_tokens()

            additional_kwargs = {**message.additional_kwargs}

            # backward compatibility
            if "function_call" in additional_kwargs:
                function_call = additional_kwargs.pop("function_call")
                if function_call.get("name", None) is not None:
                    tokens += self.get_string_tokens(function_call["name"])

                if function_call.get("arguments", None) is not None:
                    tokens += self.get_string_tokens(function_call["arguments"])

                tokens += 3  # Additional tokens for function call

            # Only count tool calls from additional_kwargs when they are not also
            # represented as ToolCallBlocks. Integrations such as OpenAI store each
            # tool call both as a ToolCallBlock (already counted by estimate_tokens
            # above) and in additional_kwargs for backward compatibility, which
            # would otherwise double-count them.
            if "tool_calls" in additional_kwargs and not _has_tool_call_blocks(message):
                tool_calls = additional_kwargs.get("tool_calls", []) or []
                for tool_call in tool_calls:
                    if (
                        hasattr(tool_call, "function")
                        and tool_call.function is not None
                    ):
                        tokens += self.get_string_tokens(tool_call.function.name)
                        tokens += self.get_string_tokens(tool_call.function.arguments)

                        tokens += 3  # Additional tokens for tool call

            tokens += 3  # Add three per message

            if message.role == MessageRole.FUNCTION or message.role == MessageRole.TOOL:
                tokens -= 2  # Subtract 2 if role is "function"

        return tokens

    async def aestimate_tokens_in_messages(self, messages: List[ChatMessage]) -> int:
        """
        Async estimate token count for a single message.

        Args:
            message (OpenAIMessage): The message to estimate the token count for.

        Returns:
            int: The estimated token count.

        """
        tokens = 0

        for message in messages:
            if message.role:
                tokens += self.get_string_tokens(message.role)

            tokens += await message.aestimate_tokens()

            additional_kwargs = {**message.additional_kwargs}

            # backward compatibility
            if "function_call" in additional_kwargs:
                function_call = additional_kwargs.pop("function_call")
                if function_call.get("name", None) is not None:
                    tokens += self.get_string_tokens(function_call["name"])

                if function_call.get("arguments", None) is not None:
                    tokens += self.get_string_tokens(function_call["arguments"])

                tokens += 3  # Additional tokens for function call

            # See the note in estimate_tokens_in_messages: skip tool calls that are
            # already represented as ToolCallBlocks to avoid double-counting.
            if "tool_calls" in additional_kwargs and not _has_tool_call_blocks(message):
                tool_calls = additional_kwargs.get("tool_calls", []) or []
                for tool_call in tool_calls:
                    if (
                        hasattr(tool_call, "function")
                        and tool_call.function is not None
                    ):
                        tokens += self.get_string_tokens(tool_call.function.name)
                        tokens += self.get_string_tokens(tool_call.function.arguments)

                        tokens += 3  # Additional tokens for tool call

            tokens += 3  # Add three per message

            if message.role == MessageRole.FUNCTION or message.role == MessageRole.TOOL:
                tokens -= 2  # Subtract 2 if role is "function"

        return tokens

    def estimate_tokens_in_tools(self, tools: List[Dict[str, Any]]) -> int:
        """
        Estimate token count for the tools.

        We take here a list of tools created using the `to_openai_tool()` function (or similar).

        Args:
            tools (list[Dict[str, Any]]): The tools to estimate the token count for.

        Returns:
            int: The estimated token count.

        """
        if not tools:
            return 0

        return self.get_string_tokens(str(tools))
