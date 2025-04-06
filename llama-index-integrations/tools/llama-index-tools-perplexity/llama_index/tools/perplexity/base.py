from llama_index.core.tools.tool_spec.base import BaseToolSpec
import requests


class PerplexityToolSpec(BaseToolSpec):
    """Perplexity API tool spec."""

    spec_functions = [
        "chat_completion",
    ]

    def __init__(self, api_key: str) -> None:
        """Initialize the Perplexity API tool spec with the given API key."""
        self.api_key = api_key

    def chat_completion(
        self,
        query: str,
        model: str = "sonar-pro",
        max_tokens: int = 256,
        temperature: float = 0.2,
        top_p: float = 0.9,
        search_domain_filter: list = None,
        return_images: bool = False,
        return_related_questions: bool = False,
        search_recency_filter: str = "",
        top_k: int = 0,
        stream: bool = False,
        presence_penalty: float = 0,
        frequency_penalty: float = 1,
        response_format: dict = None,
        web_search_options: dict = None,
    ) -> str:
        """
        Call the Perplexity API to generate a chat completion.
        All parameters have defaults aligned with the Sonar API's default parameters and can be overridden.
        Please refer to the official documentation for additional details:
        https://docs.perplexity.ai/api-reference/chat-completions

        Returns:
            str: The API response as a string.
        """
        # Set defaults for mutable parameters if not provided.
        # if search_domain_filter is None:
        #     search_domain_filter = []
        # if response_format is None:
        #     response_format = {}
        # if web_search_options is None:
        #     web_search_options = {}

        messages = [
            {"role": "system", "content": "Be precise and concise."},
            {"role": "user", "content": query},
        ]

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "search_domain_filter": search_domain_filter,
            "return_images": return_images,
            "return_related_questions": return_related_questions,
            "search_recency_filter": search_recency_filter,
            "top_k": top_k,
            "stream": stream,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "response_format": response_format,
            "web_search_options": web_search_options,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        url = "https://api.perplexity.ai/chat/completions"
        response = requests.request("POST", url, json=payload, headers=headers)
        return response.text
