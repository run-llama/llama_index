import json
from typing import Any, Iterable, List

import requests


def extract_text_list(data: Any) -> List[str]:
    """Normalize vLLM completion payloads into a list of strings."""
    if not isinstance(data, dict):
        return []

    if "choices" in data and isinstance(data["choices"], list):
        texts: List[str] = []
        for choice in data["choices"]:
            if isinstance(choice, dict):
                if "text" in choice and choice["text"] is not None:
                    texts.append(choice["text"])
                elif (
                    "message" in choice
                    and isinstance(choice["message"], dict)
                    and "content" in choice["message"]
                ):
                    texts.append(choice["message"]["content"])
        if texts:
            return texts

    if "text" in data:
        text_field = data["text"]
        if isinstance(text_field, list):
            return [text for text in text_field if isinstance(text, str)]
        if isinstance(text_field, str):
            return [text_field]

    return []


def get_response(response: requests.Response) -> List[str]:
    data = json.loads(response.content)
    return extract_text_list(data)


def post_http_request(
    api_url: str, sampling_params: dict = {}, stream: bool = False
) -> requests.Response:
    headers = {"User-Agent": "Test Client"}
    sampling_params["stream"] = stream

    return requests.post(api_url, headers=headers, json=sampling_params, stream=stream)


def get_streaming_response(response: requests.Response) -> Iterable[List[str]]:
    for chunk in response.iter_lines(
        chunk_size=8192, decode_unicode=False, delimiter=b"\0"
    ):
        if chunk:
            data = json.loads(chunk.decode("utf-8"))
            texts = extract_text_list(data)
            if texts:
                yield texts


__all__ = [
    "get_response",
    "post_http_request",
    "get_streaming_response",
    "extract_text_list",
]
