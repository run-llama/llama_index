import json
from typing import Dict, Iterable, List, Optional

import requests


def get_response(response: requests.Response) -> List[str]:
    data = json.loads(response.content)
    return data["text"]


def post_http_request(
    api_url: str,
    sampling_params: dict = {},
    stream: bool = False,
    headers: Optional[Dict[str, str]] = None,
    timeout: Optional[float] = None,
) -> requests.Response:
    base_headers = {"User-Agent": "LlamaIndex-VLLMClient"}
    if headers:
        base_headers.update(headers)
    sampling_params["stream"] = stream

    response = requests.post(
        api_url,
        headers=base_headers,
        json=sampling_params,
        stream=stream,
        timeout=timeout,
    )
    response.raise_for_status()
    return response


def get_streaming_response(response: requests.Response) -> Iterable[List[str]]:
    for chunk in response.iter_lines(
        chunk_size=8192, decode_unicode=False, delimiter=b"\0"
    ):
        if chunk:
            data = json.loads(chunk.decode("utf-8"))
            yield data["text"]


# OpenAI-like helpers
def post_openai_chat_request(
    api_url: str,
    payload: dict,
    headers: Optional[Dict[str, str]] = None,
    timeout: Optional[float] = None,
    stream: bool = False,
) -> requests.Response:
    base_headers = {"User-Agent": "LlamaIndex-VLLMOpenAI"}
    if headers:
        base_headers.update(headers)
    payload.setdefault("stream", stream)
    response = requests.post(
        api_url, headers=base_headers, json=payload, stream=stream, timeout=timeout
    )
    response.raise_for_status()
    return response


def get_openai_chat_response(response: requests.Response) -> str:
    data = response.json()
    choices = data.get("choices", [])
    if not choices:
        return ""
    choice = choices[0]
    msg = choice.get("message") or choice.get("delta") or {}
    content = msg.get("content") if isinstance(msg, dict) else None
    if content:
        return content
    # fallback
    text = data.get("text")
    if isinstance(text, list) and text:
        return text[0]
    if isinstance(text, str):
        return text
    return ""


def get_openai_streaming_deltas(response: requests.Response) -> Iterable[str]:
    for raw_line in response.iter_lines(decode_unicode=False):
        if not raw_line:
            continue
        line = raw_line.strip()
        if line.startswith(b"data:"):
            line = line[len(b"data:") :].strip()
        if line in (b"[DONE]", b""):
            continue
        data = json.loads(line.decode("utf-8"))
        choices = data.get("choices", [])
        if not choices:
            continue
        delta = choices[0].get("delta") or choices[0].get("message") or {}
        content = delta.get("content") if isinstance(delta, dict) else None
        if content:
            yield content
