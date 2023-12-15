import json
from typing import Any, Optional

from llama_index.bridge.pydantic import Field
from llama_index.callbacks import CallbackManager
from llama_index.llms import CustomLLM, CompletionResponseGen, CompletionResponse, LLMMetadata


class CwLM(CustomLLM):
    url: str = Field(description="The Cloudwalk LM request url.")
    model_name: str = Field(description="The Cloudwalk LM model to use.")
    timeout: float = Field(description="Timeout in seconds for the request.", default=60.0)
    temperature: float = Field(description="What sampling temperature to use, between 0 and 2. "
                                           "Higher values like 0.8 will make the output more random, "
                                           "while lower values like 0.2 will make it more focused and deterministic.",
                               default=1.0)

    def __init__(
            self,
            url: str,
            model_name: str,
            timeout: float = 60.0,
            temperature: float = 1.0,
            callback_manager: Optional[CallbackManager] = None,
            **kwargs: Any,
    ) -> None:
        kwargs = kwargs or {}
        super().__init__(
            url=url,
            model_name=model_name,
            timeout=timeout,
            temperature=temperature,
            callback_manager=callback_manager,
            **kwargs,
        )

    def _call_api(self, prompt: str) -> dict:
        import httpx

        headers = {"Content-Type": "application/json"}
        json_data = {
            "model": self.model_name,
            "messages": [{"content": prompt, "role": "user"}],
            "stream": False,
            "temperature": self.temperature,
        }

        with httpx.Client() as client:
            response = client.post(
                self.url,
                headers=headers,
                json=json_data,
                timeout=self.timeout,
            )

        return response.json()

    @classmethod
    def class_name(cls) -> str:
        return "CwLM"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            model_name=self.model_name,
            url=self.url,
        )

    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        response = self._call_api(prompt)
        text = json.dumps(response)
        return CompletionResponse(text=text, raw=response)

    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        raise (ValueError("Not Implemented"))
