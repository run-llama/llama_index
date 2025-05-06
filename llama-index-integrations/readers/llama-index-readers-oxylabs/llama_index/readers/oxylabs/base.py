import abc
from platform import architecture, python_version
from typing import Any, Optional
from importlib.metadata import version

from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document

from llama_index.readers.oxylabs.utils import json_to_markdown
from oxylabs import RealtimeClient, AsyncClient
from oxylabs.sources.response import Response


class OxylabsBaseReader(BasePydanticReader, abc.ABC):
    """
    Oxylabs Scraper base class.

    https://developers.oxylabs.io/scraper-apis/web-scraper-api
    """

    top_level_header: Optional[str] = None

    timeout_s: int = 100
    oxylabs_scraper_url: str = "https://realtime.oxyserps-dev.fun/v1/queries"
    oxylabs_api: RealtimeClient
    async_oxylabs_api: AsyncClient

    def __init__(self, username: str, password: str, **data) -> None:
        bits, _ = architecture()
        sdk_type = (
            f"oxylabs-llama-index-oxy-sdk-python/"
            f"{version('llama-index-readers-oxylabs')} "
            f"({python_version()}; {bits})"
        )

        data["oxylabs_api"] = RealtimeClient(username, password, sdk_type=sdk_type)
        data["async_oxylabs_api"] = AsyncClient(username, password, sdk_type=sdk_type)
        super().__init__(**data)

    def _get_document_from_response(
        self, response: list[dict] | list[list[dict]]
    ) -> Document:
        processed_content = json_to_markdown(response, 0, self.top_level_header)
        return Document(text=processed_content)

    def load_data(self, payload: dict[str, Any]) -> list[Document]:
        response = self.get_response(payload)
        validated_responses = self._validate_response(response)

        return [self._get_document_from_response(validated_responses)]

    async def aload_data(self, payload: dict[str, Any]) -> list[Document]:
        response = await self.aget_response(payload)
        validated_responses = self._validate_response(response)

        return [self._get_document_from_response(validated_responses)]

    def get_response(self, payload: dict[str, Any]) -> Response:
        raise NotImplementedError

    async def aget_response(self, payload: dict[str, Any]) -> Response:
        raise NotImplementedError

    @staticmethod
    def _validate_response(
        response: Any,
    ) -> list[dict[Any, Any]] | list[list[dict[Any, Any]]]:
        """
        Validate Oxylabs response format and unpack data.
        """
        validated_results = []
        try:
            result_pages = response.raw["results"]
            if not isinstance(result_pages, list) or not result_pages:
                raise ValueError("No results returned!")

            for result_page in result_pages:
                result_page = dict(result_page)
                content = result_page["content"]

                if isinstance(content, list):
                    validated_results.append(content)
                    continue

                if not isinstance(content, dict):
                    raise ValueError(
                        "Result `content` format error,"
                        " try setting parameter `parse` to True"
                    )

                if "results" in content:
                    result = content["results"]
                    if isinstance(result, list):
                        validated_results.append(result)
                    elif isinstance(result, dict):
                        validated_results.append(result)
                    else:
                        raise ValueError("Response format Error!")
                else:
                    validated_results.append(content)

            return validated_results

        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Response Validation Error: {exc!s}") from exc
