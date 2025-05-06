"""Oxylabs Web Reader."""

import asyncio
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from platform import architecture, python_version
from importlib.metadata import version

from llama_index.core.bridge.pydantic import Field
from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document
from markdownify import markdownify

from llama_index.readers.web.oxylabs_web.utils import strip_html, json_to_markdown

if TYPE_CHECKING:
    from oxylabs.internal.api import AsyncAPI, RealtimeAPI


def get_default_config() -> dict[str, Any]:
    from oxylabs.utils.utils import prepare_config

    return prepare_config(async_integration=True)


class OxylabsWebReader(BasePydanticReader):
    """
    Scrape any website with Oxylabs Scraper.

    Oxylabs API documentation:
    https://developers.oxylabs.io/scraper-apis/web-scraper-api/other-websites

    Args:
        username: Oxylabs username.
        password: Oxylabs password.

    Example:
        .. code-block:: python
            from llama_index.readers.web.oxylabs_web.base import OxylabsWebReader

            reader = OxylabsWebReader(
                username=os.environ["OXYLABS_USERNAME"], password=os.environ["OXYLABS_PASSWORD"]
            )

            docs = reader.load_data(
                [
                    "https://sandbox.oxylabs.io/products/1",
                    "https://sandbox.oxylabs.io/products/2"
                ],
                {
                    "parse": True,
                }
            )

            print(docs[0].text)

    """

    timeout_s: int = 100
    oxylabs_scraper_url: str = "https://realtime.oxylabs.io/v1/queries"
    api: "RealtimeAPI"
    async_api: "AsyncAPI"
    default_config: dict[str, Any] = Field(default_factory=get_default_config)

    def __init__(self, username: str, password: str, **kwargs) -> None:
        from oxylabs.internal.api import AsyncAPI, APICredentials, RealtimeAPI

        credentials = APICredentials(username=username, password=password)

        bits, _ = architecture()
        sdk_type = (
            f"oxylabs-llama-index-web-sdk-python/"
            f"{version('llama-index-readers-web')} "
            f"({python_version()}; {bits})"
        )

        api = RealtimeAPI(credentials, sdk_type=sdk_type)
        async_api = AsyncAPI(credentials, sdk_type=sdk_type)

        super().__init__(api=api, async_api=async_api, **kwargs)

    @classmethod
    def class_name(cls) -> str:
        return "OxylabsWebReader"

    def _get_document_from_response(self, response: dict[str, Any]) -> Document:
        content = response["results"][0]["content"]

        if isinstance(content, (dict, list)):
            text = json_to_markdown(content)
        else:
            striped_html = strip_html(str(content))
            text = markdownify(striped_html)

        return Document(
            metadata={"oxylabs_job": response["job"]},
            text=text,
        )

    async def aload_data(
        self,
        urls: list[str],
        additional_params: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Asynchronously load data from urls.

        Args:
            urls: List of URLs to load.
            additional_params: Dictionary with the scraper parameters. Accepts the values from
                the additional parameters described here:
                https://developers.oxylabs.io/scraper-apis/web-scraper-api/targets/generic-target#additional

        """
        if additional_params is None:
            additional_params = {}

        responses = await asyncio.gather(
            *[
                self.async_api.get_response(
                    {**additional_params, "url": url},
                    self.default_config,
                )
                for url in urls
            ]
        )

        return [
            self._get_document_from_response(response)
            for response in responses
            if response
        ]

    def load_data(
        self,
        urls: list[str],
        additional_params: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Load data from urls.

        Args:
            urls: List of URLs to load.
            additional_params: Dictionary with the scraper parameters. Accepts the values from
                the additional parameters described here:
                https://developers.oxylabs.io/scraper-apis/web-scraper-api/targets/generic-target#additional

        """
        if additional_params is None:
            additional_params = {}

        responses = [
            self.api.get_response(
                {**additional_params, "url": url},
                self.default_config,
            )
            for url in urls
        ]

        return [
            self._get_document_from_response(response)
            for response in responses
            if response
        ]
