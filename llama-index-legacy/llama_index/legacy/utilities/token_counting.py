# Modified from:
# https://github.com/nyno-ai/openai-token-counter

from typing import Any, Callable, Dict, List, Optional

from llama_index.legacy.llms import ChatMessage, MessageRole
from llama_index.legacy.utils import get_tokenizer


class TokenCounter:
    """Token counter class.

    Attributes:
        model (Optional[str]): The model to use for token counting.
    """

    def __init__(self, tokenizer: Optional[Callable[[str], list]] = None) -> None:
        self.tokenizer = tokenizer or get_tokenizer()

    def get_string_tokens(self, string: str) -> int:
        """Get the token count for a string.

        Args:
            string (str): The string to count.

        Returns:
            int: The token count.
        """
        return len(self.tokenizer(string))

    def estimate_tokens_in_messages(self, messages: List[ChatMessage]) -> int:
        """Estimate token count for a single message.

        Args:
            message (OpenAIMessage): The message to estimate the token count for.

        Returns:
            int: The estimated token count.
        """
        tokens = 0

        for message in messages:
            if message.role:
                tokens += self.get_string_tokens(message.role)

            if message.content:
                tokens += self.get_string_tokens(message.content)

            additional_kwargs = {**message.additional_kwargs}

            if "function_call" in additional_kwargs:
                function_call = additional_kwargs.pop("function_call")
                if function_call.get("name", None) is not None:
                    tokens += self.get_string_tokens(function_call["name"])

                if function_call.get("arguments", None) is not None:
                    tokens += self.get_string_tokens(function_call["arguments"])

                tokens += 3  # Additional tokens for function call

            tokens += 3  # Add three per message

            if message.role == MessageRole.FUNCTION:
                tokens -= 2  # Subtract 2 if role is "function"

        return tokens

    def estimate_tokens_in_functions(self, functions: List[Dict[str, Any]]) -> int:
        """Estimate token count for the functions.

        We take here a list of functions created using the `to_openai_spec` function (or similar).

        Args:
            function (list[Dict[str, Any]]): The functions to estimate the token count for.

        Returns:
            int: The estimated token count.
        """
        prompt_definition = str(functions)
        tokens = self.get_string_tokens(prompt_definition)
        tokens += 9  # Additional tokens for function definition
        return tokens
