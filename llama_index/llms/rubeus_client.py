import json
from types import TracebackType
from typing import Dict, Any, Union, Mapping, cast, List, Optional, Type
import httpx
from llama_index.llms.rubeus_utils import (
    remove_empty_values,
    DEFAULT_MAX_RETRIES,
    Body,
    Options,
    Config,
    ProviderOptions,
    RubeusResponse,
)
from pydantic import BaseModel
from llama_index.llms import rubeus_exceptions as exceptions


class APIClient:
    _client: httpx.Client

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        timeout: Union[float, None],
        max_retries: int = DEFAULT_MAX_RETRIES,
        custom_headers: Optional[Mapping[str, str]] = None,
        custom_query: Optional[Mapping[str, object]],
        custom_params: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.api_key = api_key
        self.max_retries = max_retries
        self._custom_headers = custom_headers
        self._custom_query = custom_query or None
        self._custom_params = custom_params
        self._client = httpx.Client(
            base_url=base_url,
            timeout=timeout,
            headers={"Accept": "application/json"},
        )

    @property
    def custom_auth(self) -> Optional[httpx.Auth]:
        return None

    def post(
        self,
        path: str,
        *,
        body: List[Body],
        stream: bool,
        mode: str,
        cast_to: Type[RubeusResponse],
    ) -> RubeusResponse:
        body = cast(List[Body], body)
        opts = self._construct(method="post", url=path, body=body, mode=mode)
        return self._request(options=opts, stream=stream, cast_to=cast_to)

    def _construct(
        self, *, method: str, url: str, body: List[Body], mode: str
    ) -> Options:
        opts = Options.construct()
        opts.method = method
        opts.url = url
        json_body = {
            "config": self._config(mode, body).dict(),
            "params": self._custom_params,
        }
        opts.json_body = remove_empty_values(json_body)
        opts.headers = self._custom_headers or None
        return opts

    def _config(self, mode: str, body: List[Body]) -> Config:
        config = Config(mode=mode, options=[])
        for i in body:
            item = i.dict()
            options = ProviderOptions(
                provider=item.get("provider"),
                apiKey=item.get("model_api_key"),
                weight=item.get("weight"),
                retry=item.get("retry"),
                override_params=item.get("override_params"),
            )
            config.options.append(options)
        return config

    @property
    def _default_headers(self) -> dict[str, str]:
        return {"Content-Type": "application/json", "x-portkey-api-key": self.api_key}

    def _build_headers(self, options: Options) -> httpx.Headers:
        custom_headers = options.headers or {}
        headers_dict = self._merge_mappings(self._default_headers, custom_headers)

        headers = httpx.Headers(headers_dict)
        return headers

    def _merge_mappings(
        self,
        obj1: Mapping[str, Any],
        obj2: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Merge two mappings of the given type
        In cases with duplicate keys the second mapping takes precedence.
        """
        return {**obj1, **obj2}

    def is_closed(self) -> bool:
        return self._client.is_closed

    def close(self) -> None:
        """Close the underlying HTTPX client.

        The client will *not* be usable after this.
        """
        self._client.close()

    def __enter__(self: Any) -> Any:
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.close()

    def _build_request(self, options: Options) -> httpx.Request:
        headers = self._build_headers(options)
        params = options.params
        json_body = options.json_body
        request = self._client.build_request(
            method=options.method,
            url=options.url,
            headers=headers,
            params=params,
            json=json_body,
            timeout=options.timeout,
        )
        return request

    def _request(
        self, *, options: Options, stream: bool, cast_to: Type[RubeusResponse]
    ) -> RubeusResponse:
        request = self._build_request(options)
        try:
            res = self._client.send(request, auth=self.custom_auth, stream=stream)
            res.raise_for_status()
        except httpx.HTTPStatusError as err:  # 4xx and 5xx errors
            # If the response is streamed then we need to explicitly read the response
            # to completion before attempting to access the response text.
            err.response.read()
            raise self._make_status_error_from_response(request, err.response) from None
        except httpx.TimeoutException as err:
            raise exceptions.APITimeoutError(request=request) from err
        except Exception as err:
            raise exceptions.APIConnectionError(request=request) from err
        response = cast(
            RubeusResponse, BaseModel.construct(**res.json(), raw_body=res.json())
        )
        return response

    def _make_status_error_from_response(
        self,
        request: httpx.Request,
        response: httpx.Response,
    ) -> exceptions.APIStatusError:
        err_text = response.text.strip()
        body = err_text

        try:
            body = json.loads(err_text)["error"]["message"]
            err_msg = f"Error code: {response.status_code} - {body}"
        except Exception:
            err_msg = err_text or f"Error code: {response.status_code}"

        return self._make_status_error(
            err_msg, body=body, request=request, response=response
        )

    def _make_status_error(
        self,
        err_msg: str,
        *,
        body: object,
        request: httpx.Request,
        response: httpx.Response,
    ) -> exceptions.APIStatusError:
        if response.status_code == 400:
            return exceptions.BadRequestError(
                err_msg, request=request, response=response, body=body
            )
        if response.status_code == 401:
            return exceptions.AuthenticationError(
                err_msg, request=request, response=response, body=body
            )
        if response.status_code == 403:
            return exceptions.PermissionDeniedError(
                err_msg, request=request, response=response, body=body
            )
        if response.status_code == 404:
            return exceptions.NotFoundError(
                err_msg, request=request, response=response, body=body
            )
        if response.status_code == 409:
            return exceptions.ConflictError(
                err_msg, request=request, response=response, body=body
            )
        if response.status_code == 422:
            return exceptions.UnprocessableEntityError(
                err_msg, request=request, response=response, body=body
            )
        if response.status_code == 429:
            return exceptions.RateLimitError(
                err_msg, request=request, response=response, body=body
            )
        if response.status_code >= 500:
            return exceptions.InternalServerError(
                err_msg, request=request, response=response, body=body
            )
        return exceptions.APIStatusError(
            err_msg, request=request, response=response, body=body
        )
