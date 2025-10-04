import json
from typing import Iterable, List

import requests


def get_response(response: requests.Response) -> List[str]:
    """Extract text from SGLang API response."""
    data = json.loads(response.content)
    # SGLang typically returns text in a 'text' field
    if isinstance(data, dict) and "text" in data:
        text = data["text"]
        # Handle both single string and list of strings
        if isinstance(text, str):
            return [text]
        return text
    return []


def post_http_request(
    api_url: str, sampling_params: dict = {}, stream: bool = False
) -> requests.Response:
    """Post HTTP request to SGLang server."""
    headers = {
        "User-Agent": "LlamaIndex SGLang Client",
        "Content-Type": "application/json",
    }
    sampling_params["stream"] = stream

    return requests.post(
        api_url,
        headers=headers,
        json=sampling_params,
        stream=stream,
    )


def get_streaming_response(response: requests.Response) -> Iterable[List[str]]:
    """Get streaming response from SGLang server."""
    for chunk in response.iter_lines(
        chunk_size=8192, decode_unicode=False, delimiter=b"\n"
    ):
        if chunk:
            chunk_str = chunk.decode("utf-8")
            # Handle SSE format
            if chunk_str.startswith("data: "):
                chunk_str = chunk_str[6:]
            
            if chunk_str.strip() == "[DONE]":
                break
            
            try:
                data = json.loads(chunk_str)
                if "text" in data:
                    text = data["text"]
                    if isinstance(text, str):
                        yield [text]
                    else:
                        yield text
            except json.JSONDecodeError:
                continue