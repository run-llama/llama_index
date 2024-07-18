import json
import logging
import urllib.parse
from typing import Dict, Mapping, Any, Sequence, Union, Tuple, Iterable, AsyncIterable

import httpx

from llama_index.utils.qianfan.authorization import encode_authorization


QueryParamTypes = Union[
    Mapping[str, Union[Any, Sequence[Any]]],
    Sequence[Tuple[str, Any]],
]


class Error(Exception):
    """
    Error message returned by Baidu QIANFAN LLM Platform.
    """

    error_code: int
    error_msg: str


logger = logging.getLogger(__name__)


def _rebuild_url(
    url: str, params: QueryParamTypes = None
) -> Tuple[str, str, QueryParamTypes]:
    """
    Rebuild url and return the full URL, the URL without the query, and the query parameters.
    """
    parsed_url = urllib.parse.urlparse(url)
    query_items = urllib.parse.parse_qsl(parsed_url.query)

    query = httpx.QueryParams(query_items)
    if params:
        query = query.merge(params)

    full_url = urllib.parse.ParseResult(
        parsed_url.scheme,
        parsed_url.netloc,
        parsed_url.path,
        params="",
        query=str(query),
        fragment=parsed_url.fragment,
    )
    url_without_query = urllib.parse.ParseResult(
        parsed_url.scheme,
        parsed_url.netloc,
        parsed_url.path,
        params="",
        query="",
        fragment=parsed_url.fragment,
    )
    return full_url.geturl(), url_without_query.geturl(), query.multi_items()


class Client:
    """
    The access client for Baidu's Qianfan LLM Platform.
    """

    def __init__(self, access_key: str, secret_key: str):
        """
        Initialize a Client instance.

        :param access_key: The Access Key obtained from the Security Authentication Center of Baidu Intelligent Cloud Console.
        :param secret_key: The Secret Key paired with the Access Key.
        """
        self._access_key = access_key
        self._secret_key = secret_key

    def _get_headers(self, method: str, url: str) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
        }
        authorization = encode_authorization(
            method.upper(), url, headers, self._access_key, self._secret_key
        )
        headers["Authorization"] = authorization

        return headers

    def _preprocess(
        self, method: str, url: str, params: QueryParamTypes = None, json: Any = None
    ) -> httpx.Request:
        full_url, url_without_query, params = _rebuild_url(url, params)

        if logger.level <= logging.DEBUG:
            logging.debug(f"{method} {url_without_query}, request body: {json}")

        headers = self._get_headers(method, full_url)
        return httpx.Request(
            method=method,
            url=url_without_query,
            params=params,
            headers=headers,
            json=json,
        )

    def _postprocess(self, r: httpx.Response) -> Dict:
        if logger.level <= logging.DEBUG:
            logger.debug(f"{r.request.method} {r.url} response body: {r.text}")
        resp_dict = r.json()

        error_code = resp_dict.get("error_code", 0)
        if error_code != 0:
            raise Error(error_code, resp_dict.get("error_msg"))

        return resp_dict

    def _postprocess_stream_part(self, line: str) -> Iterable[Dict]:
        if line == "":
            return

        if line.startswith("{") and line.endswith("}"):  # error
            resp_dict = json.loads(line)
            error_code = resp_dict.get("error_code", 0)
            if error_code != 0:
                raise Error(error_code, resp_dict.get("error_msg"))

        if line.startswith("data: "):
            line = line[len("data: ") :]
            resp_dict = json.loads(line)
            yield resp_dict

    def post(self, url: str, params: QueryParamTypes = None, json: Any = None) -> Dict:
        """
        Make an Request with POST Method.
        """
        request = self._preprocess("POST", url=url, params=params, json=json)
        with httpx.Client() as client:
            r = client.send(request=request)
        r.raise_for_status()
        return self._postprocess(r)

    async def apost(
        self, url: str, params: QueryParamTypes = None, json: Any = None
    ) -> Dict:
        """
        Make an Asynchronous Request with POST Method.
        """
        response = self._preprocess("POST", url=url, params=params, json=json)
        async with httpx.AsyncClient() as aclient:
            r = await aclient.send(request=response)
        r.raise_for_status()
        return self._postprocess(r)

    def post_reply_stream(
        self, url: str, params: QueryParamTypes = None, json: Any = None
    ) -> Iterable[Dict]:
        """
        Make an Request with POST Method and the response is returned in a stream.
        """
        request = self._preprocess("POST", url=url, params=params, json=json)

        with httpx.Client() as client:
            r = client.send(request=request, stream=True)
            r.raise_for_status()
            for line in r.iter_lines():
                yield from self._postprocess_stream_part(line)

    async def apost_reply_stream(
        self, url: str, params: QueryParamTypes = None, json: Any = None
    ) -> AsyncIterable[Dict]:
        """
        Make an Asynchronous Request with POST Method and the response is returned in a stream.
        """
        request = self._preprocess("POST", url=url, params=params, json=json)

        async with httpx.AsyncClient() as aclient:
            r = await aclient.send(request=request, stream=True)
            r.raise_for_status()
            async for line in r.aiter_lines():
                for part in self._postprocess_stream_part(line):
                    yield part
