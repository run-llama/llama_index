"""ChatGPT Wrapper."""

from typing import Any, List, Optional

import openai

from gpt_index.langchain_helpers.chain_wrapper import LLMPredictor
from gpt_index.prompts.base import Prompt

DEFAULT_MESSAGE_PREPEND = [
    {"role": "system", "content": "You are a helpful assistant."},
]


class ChatGPTLLMPredictor(LLMPredictor):
    """LLM predictor class.

    Args:
        retry_on_throttling (bool): Whether to retry on rate limit errors.
            Defaults to true.
        prepend_messages (Optional[List[str]]): Messages to prepend to the
            ChatGPT API.
        openai_kwargs (Any): Additional kwargs to pass to the OpenAI API.
            https://platform.openai.com/docs/api-reference/chat/create.

    """

    def __init__(
        self,
        prepend_messages: Optional[List[str]] = None,
        include_role_in_response: bool = False,
        **openai_kwargs: Any
    ) -> None:
        """Initialize params."""
        self._prepend_messages = prepend_messages or DEFAULT_MESSAGE_PREPEND
        self._include_role_in_response = include_role_in_response
        # set openAI kwargs
        if "temperature" not in openai_kwargs:
            openai_kwargs["temperature"] = 0
        self._openai_kwargs = openai_kwargs
        self._total_tokens_used = 0
        self.flag = True
        self._last_token_usage: Optional[int] = None

    def _predict(self, prompt: Prompt, **prompt_args: Any) -> str:
        """Inner predict function.

        If retry_on_throttling is true, we will retry on rate limit errors.

        """
        prompt_str = prompt.format(**prompt_args)
        messages = self._prepend_messages + [{"role": "user", "content": prompt_str}]

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages, **self._openai_kwargs
        )

        # hacky
        message_obj = response["choices"][0]["message"]
        response = ""
        if self._include_role_in_response:
            response += "role: " + message_obj["role"] + "\n"
        response += message_obj["content"]
        return response

    async def _apredict(self, prompt: Prompt, **prompt_args: Any) -> str:
        """Async inner predict function.

        If retry_on_throttling is true, we will retry on rate limit errors.

        """
        prompt_str = prompt.format(**prompt_args)
        messages = self._prepend_messages + [{"role": "user", "content": prompt_str}]

        response = await openai.ChatCompletion.acreate(
            model="gpt-3.5-turbo", messages=messages, **self._openai_kwargs
        )

        # hacky
        message_obj = response["choices"][0]["message"]
        response = ""
        if self._include_role_in_response:
            response += "role: " + message_obj["role"] + "\n"
        response += message_obj["content"]
        return response
