import os
from typing import Any, Dict, Generator, Literal

import requests
import sseclient
from llama_index.core.llms import (
    CompletionResponse,
    CompletionResponseGen,
    CustomLLM,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback
from pydantic import Field

SMART_ENDPOINT = "https://chat-api.you.com/smart"
RESEARCH_ENDPOINT = "https://chat-api.you.com/research"


def _request(base_url: str, api_key: str, **kwargs) -> Dict[str, Any]:
    """
    This function can be replaced by a OpenAPI-generated Python SDK in the future,
    for better input/output typing support.
    """
    headers = {"x-api-key": api_key}
    response = requests.post(base_url, headers=headers, json=kwargs)
    response.raise_for_status()
    return response.json()


def _request_stream(
    base_url: str, api_key: str, **kwargs
) -> Generator[str, None, None]:
    headers = {"x-api-key": api_key}
    params = dict(**kwargs, stream=True)
    response = requests.post(base_url, headers=headers, stream=True, json=params)
    response.raise_for_status()

    client = sseclient.SSEClient(response)
    for event in client.events():
        if event.event in ("search_results", "done"):
            pass
        elif event.event == "token":
            yield event.data
        elif event.event == "error":
            raise ValueError(f"Error in response: {event.data}")
        else:
            raise NotImplementedError(f"Unknown event type {event.event}")


class YouLM(CustomLLM):
    # TODO: DOCME

    mode: Literal["smart", "research"] = Field("smart", description="# TODO[DOCME]")

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            model_name=f"you.com-{self.mode}",
            is_chat_model=True,
            is_function_calling_model=False,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        response = _request(
            self.endpoint,
            api_key=self.api_key,
            query=prompt,
        )
        return CompletionResponse(text=response["answer"], raw=response)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        response = _request_stream(
            self.endpoint,
            api_key=self.api_key,
            query=prompt,
        )

        completion = ""
        for token in response:
            completion += token
            yield CompletionResponse(text=completion, delta=token)

    @property
    def endpoint(self) -> str:
        if self.mode == "smart":
            return SMART_ENDPOINT
        return RESEARCH_ENDPOINT

    @property
    def api_key(self) -> str:
        return os.environ["YDC_API_KEY"]
