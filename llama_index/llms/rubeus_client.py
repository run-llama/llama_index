from types import TracebackType
from typing import Dict, Any, Union, Mapping, cast, List, Optional
import httpx
from httpx import Timeout
from llama_index.llms.rubeus_utils import (
    remove_empty_values,
    Params,
    DEFAULT_MAX_RETRIES,
    Body,
    Options,
    Config,
    ProviderOptions,
)


class APIClient:
    _client: httpx.Client

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        timeout: Union[float, Timeout, None],
        max_retries: int = DEFAULT_MAX_RETRIES,
        custom_headers: Mapping[str, str] | None = None,
        custom_query: Optional[Mapping[str, object]],
        custom_params: Optional[Params],
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
    def custom_auth(self) -> httpx.Auth | None:
        return None

    def post(
        self, path: str, *, body: List[Body], stream: bool, mode: str
    ) -> httpx.Response:
        body = cast(List[Body], body)
        opts = self._construct(method="post", url=path, body=body, mode=mode)
        return self._request(options=opts, stream=stream)

    def _construct(
        self, *, method: str, url: str, body: List[Body], mode: str
    ) -> Options:
        opts = Options.construct()
        opts.method = method
        opts.url = url
        opts.json_body = {
            "config": self._config(mode, body),
            "params": self._custom_params,
        }
        opts.headers = self._custom_headers or None
        return opts

    def _config(self, mode: str, body: List[Body]) -> Config:
        config: Config = {"mode": mode, "options": []}
        for i in body:
            options: ProviderOptions = {
                "messages": i.get("messages"),
                "prompt": i.get("prompt"),
                "provider": i.get("provider"),
                "apiKey": i.get("model_api_key"),
                "weight": i.get("weight"),
                # "retry": {"attempts": i.get("max_retries"), "on_status_codes": []},
                "override_params": {"model": i.get("model")},
            }
            config["options"].append(options)
        cleaned_config = cast(Config, remove_empty_values(config))
        return cleaned_config

    @property
    def default_headers(self) -> dict[str, str]:
        return {"Content-Type": "application/json", "x-portkey-api-key": self.api_key}

    def _build_headers(self, options: Options) -> httpx.Headers:
        custom_headers = options.headers or {}
        headers_dict = self._merge_mappings(self.default_headers, custom_headers)

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
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType | None,
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
        import json

        print("options: ", json.dumps(options.json_body))
        return request

    def _request(self, *, options: Options, stream: bool) -> httpx.Response:
        request = self._build_request(options)
        return self._client.send(request, auth=self.custom_auth, stream=stream)
