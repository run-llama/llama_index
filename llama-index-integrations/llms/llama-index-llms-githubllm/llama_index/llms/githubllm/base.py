import os
import logging
import time
from typing import Any, Dict, List, Sequence, Generator

import requests
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from llama_index.core.llms.custom import CustomLLM
from llama_index.core.bridge.pydantic import Field, PrivateAttr, validator

logger = logging.getLogger(__name__)


class GithubLLM(CustomLLM):
    """GitHub LLM with Azure Fallback.

    This module allows using LLMs hosted on GitHub's inference endpoint with automatic fallback to Azure when rate limits are reached.

    To use this module, you must:
    * Export your GitHub token as the environment variable `GITHUB_TOKEN`
    * Export your Azure API key as the environment variable `AZURE_API_KEY` (for fallback)
    * Specify the model name you want to use

    Example:
        .. code-block:: python

        from llama_index.llms.github import GithubLLM

        # Make sure GITHUB_TOKEN and AZURE_API_KEY are set in your environment variables
        llm = GithubLLM(model="gpt-4o", system_prompt="You are a knowledgeable history teacher.", use_azure_fallback=True)

        # Single turn completion
        response = llm.complete("What is the capital of France?")
        print(response)

        # Multi-turn chat
        messages = [
            ChatMessage(role="user", content="Tell me about the French Revolution."),
            ChatMessage(role="assistant", content="The French Revolution was a period of major social and political upheaval in France that began in 1789 with the Storming of the Bastille and ended in the late 1790s with the ascent of Napoleon Bonaparte. It was partially carried forward by Napoleon during the later expansion of the French Empire. The Revolution overthrew the monarchy, established a republic, catalyzed violent periods of political turmoil, and fundamentally altered French history."),
            ChatMessage(role="user", content="What were the main causes?")
        ]

        response = llm.chat(messages)
        print(response)

        # Streaming chat
        for chunk in llm.stream_chat([ChatMessage(role="user", content="Can you elaborate on the Reign of Terror?")]):
            print(chunk.message.content, end='', flush=True)
    """

    github_endpoint_url: str = Field(
        default="https://models.inference.ai.azure.com/chat/completions"
    )
    model: str = Field(description="The model to use for inference.")
    system_prompt: str = Field(default="You are a helpful assistant.")
    use_azure_fallback: bool = Field(default=True)
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=None)

    _rate_limit_reset_time: float = PrivateAttr(default=0)
    _request_count: int = PrivateAttr(default=0)
    _max_requests_per_minute: int = PrivateAttr(default=15)  # Adjust based on your tier
    _max_requests_per_day: int = PrivateAttr(default=150)  # Adjust based on your tier

    SUPPORTED_MODELS = [
        "AI21-Jamba-Instruct",
        "cohere-command-r",
        "cohere-command-r-plus",
        "cohere-embed-v3-english",
        "cohere-embed-v3-multilingual",
        "meta-llama-3-70b-instruct",
        "meta-llama-3-8b-instruct",
        "meta-llama-3.1-405b-instruct",
        "meta-llama-3.1-70b-instruct",
        "meta-llama-3.1-8b-instruct",
        "mistral-large",
        "mistral-large-2407",
        "mistral-nemo",
        "mistral-small",
        "gpt-4o",
        "gpt-4o-mini",
        "phi-3-medium-instruct-128k",
        "phi-3-medium-instruct-4k",
        "phi-3-mini-instruct-128k",
        "phi-3-mini-instruct-4k",
        "phi-3-small-instruct-128k",
        "phi-3-small-instruct-8k",
    ]

    @validator("model")
    def validate_model(cls, v):
        if v.lower() not in [model.lower() for model in cls.SUPPORTED_MODELS]:
            raise ValueError(
                f"Model {v} is not supported. Please choose from {cls.SUPPORTED_MODELS}"
            )
        return v

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=4096,  # Assuming a default context window, adjust as needed
            num_output=self.max_tokens or 256,  # Adjust default as needed
            model_name=self.model,
        )

    def _check_rate_limit(self) -> bool:
        """Check if the rate limit has been reached."""
        current_time = time.time()
        if current_time < self._rate_limit_reset_time:
            return False
        if self._request_count >= self._max_requests_per_minute:
            self._rate_limit_reset_time = current_time + 60
            self._request_count = 0
            return False
        return True

    def _increment_request_count(self):
        """Increment the request count."""
        self._request_count += 1

    def _call_api(
        self,
        endpoint_url: str,
        headers: Dict[str, str],
        data: Dict[str, Any],
        stream: bool = False,
    ) -> Any:
        """Make an API call to either GitHub or Azure."""
        if stream:
            response = requests.post(
                endpoint_url, headers=headers, json=data, stream=True
            )
        else:
            response = requests.post(endpoint_url, headers=headers, json=data)
        response.raise_for_status()
        return response

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """Generate a completion."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        response_content = self._call_llm(messages, **kwargs)
        return CompletionResponse(text=response_content)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        """Stream a completion."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        for chunk in self._stream_llm(messages, **kwargs):
            yield CompletionResponse(text=chunk, delta=chunk)

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        """Generate a chat response."""
        formatted_messages = [{"role": "system", "content": self.system_prompt}]
        formatted_messages.extend(
            [{"role": m.role, "content": m.content} for m in messages]
        )

        response_content = self._call_llm(formatted_messages, **kwargs)
        return ChatResponse(
            message=ChatMessage(role="assistant", content=response_content)
        )

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        """Stream a chat response."""
        formatted_messages = [{"role": "system", "content": self.system_prompt}]
        formatted_messages.extend(
            [{"role": m.role, "content": m.content} for m in messages]
        )

        for chunk in self._stream_llm(formatted_messages, **kwargs):
            yield ChatResponse(
                message=ChatMessage(role="assistant", content=chunk), delta=chunk
            )

    def _call_llm(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        """Call the LLM with fallback to Azure if rate limited."""
        data = {
            "messages": messages,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **kwargs,
        }

        if self._check_rate_limit():
            try:
                github_token = os.environ.get("GITHUB_TOKEN")
                if not github_token:
                    raise ValueError("GITHUB_TOKEN environment variable is not set.")

                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {github_token}",
                }

                response = self._call_api(self.github_endpoint_url, headers, data)
                self._increment_request_count()
                return response.json()["choices"][0]["message"]["content"]
            except (requests.exceptions.RequestException, ValueError) as e:
                logger.warning(f"GitHub API call failed: {e!s}. Falling back to Azure.")

        if self.use_azure_fallback:
            azure_api_key = os.environ.get("AZURE_API_KEY")
            if not azure_api_key:
                raise ValueError("AZURE_API_KEY environment variable is not set.")

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {azure_api_key}",
            }

            response = self._call_api(self.github_endpoint_url, headers, data)
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise ValueError("Rate limit reached and Azure fallback is disabled.")

    def _stream_llm(
        self, messages: List[Dict[str, str]], **kwargs: Any
    ) -> Generator[str, None, None]:
        """Stream from the LLM with fallback to Azure if rate limited."""
        data = {
            "messages": messages,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
            **kwargs,
        }

        if self._check_rate_limit():
            try:
                github_token = os.environ.get("GITHUB_TOKEN")
                if not github_token:
                    raise ValueError("GITHUB_TOKEN environment variable is not set.")

                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {github_token}",
                }

                response = self._call_api(
                    self.github_endpoint_url, headers, data, stream=True
                )
                self._increment_request_count()
                for line in response.iter_lines():
                    if line:
                        yield line.decode("utf-8")
                return
            except (requests.exceptions.RequestException, ValueError) as e:
                logger.warning(f"GitHub API call failed: {e!s}. Falling back to Azure.")

        if self.use_azure_fallback:
            azure_api_key = os.environ.get("AZURE_API_KEY")
            if not azure_api_key:
                raise ValueError("AZURE_API_KEY environment variable is not set.")

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {azure_api_key}",
            }

            response = self._call_api(
                self.github_endpoint_url, headers, data, stream=True
            )
            for line in response.iter_lines():
                if line:
                    yield line.decode("utf-8")
        else:
            raise ValueError("Rate limit reached and Azure fallback is disabled.")

    @classmethod
    def class_name(cls) -> str:
        """Get the name of the class."""
        return "GithubLLM"
