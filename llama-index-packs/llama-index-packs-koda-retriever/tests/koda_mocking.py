"""
This module contains a class (KVMockLLM) that slightly extends the Llama Index MockLLM
class further by allowing the user to specify a dictionary of prompt-response pairs.
This allows the user to specify the response that the mock LLM should return based on the prompt.

This should really only be used for testing purposes to simulate the behavior of a real LLM.
Maybe this would be useful in the LlamaIndex repo itself?

AUTHOR: no_dice
"""

from llama_index.core.llms.mock import MockLLM
from llama_index.core.base.llms.types import CompletionResponse
import random


PROMPT_RESPONSES = {
    "What are LLMs good at?": "concept seeking query",
    "How many Jurassic Park movies are there?": "fact seeking query",
}


class KVMockLLM(MockLLM):
    """Simple mock LLM that returns a response based on the prompt."""

    prompt_responses: dict = PROMPT_RESPONSES
    strict: bool = False
    default_response: str = "concept seeking query"

    @classmethod
    def class_name(cls) -> str:
        return "KVMockLLM"

    def random_prompt(self) -> str:
        """Returns a random prompt from the prompt_responses dictionary."""
        return random.choice(list(self.prompt_responses.keys()))

    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        """Returns a response that was matched from the given prompt."""
        if self.strict:
            if prompt not in self.prompt_responses:
                err = f"Prompt '{prompt}' not found in prompt_responses. Please recreate this MockLLM with the expected prompts and responses."
                raise ValueError(err)

        response = self.prompt_responses.get(prompt, self.default_response)

        return CompletionResponse(text=response)
