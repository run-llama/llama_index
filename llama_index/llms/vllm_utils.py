import json
from typing import Iterable, List

import requests


def get_response(response: requests.Response) -> List[str]:
    data = json.loads(response.content)
    return data["text"]


def post_http_request(
    api_url: str, sampling_params: dict = {}, stream: bool = False
) -> requests.Response:
    headers = {"User-Agent": "Test Client"}
    sampling_params["stream"] = stream

    return requests.post(api_url, headers=headers, json=sampling_params, stream=True)


def get_streaming_response(response: requests.Response) -> Iterable[List[str]]:
    for chunk in response.iter_lines(
        chunk_size=8192, decode_unicode=False, delimiter=b"\0"
    ):
        if chunk:
            data = json.loads(chunk.decode("utf-8"))
            yield data["text"]
