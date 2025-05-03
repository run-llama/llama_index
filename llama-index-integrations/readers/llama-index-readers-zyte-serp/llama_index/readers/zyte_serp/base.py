from typing import Optional

from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document

from zyte_api import ZyteAPI
from zyte_api.utils import USER_AGENT as PYTHON_ZYTE_API_USER_AGENT


class ZyteSerpReader(BasePydanticReader):
    """
    Get google search results URLs for a search query.

    Args:
        api_key: Zyte API key.
        extract_from: Determines the mode while extracting the search results.
            It can take one of the following values: 'httpResponseBody', 'browserHtml'

    Example:
        .. code-block:: python

            from llama_index.readers.zyte_serp import ZyteSerpReader

            reader = ZyteSerpReader(
               api_key="ZYTE_API_KEY",
            )
            docs = reader.load_data(
                "search query",
            )

    Zyte-API reference:
        https://docs.zyte.com/zyte-api/get-started.html

    """

    client: ZyteAPI
    api_key: str
    extract_from: Optional[str]

    def __init__(
        self,
        api_key: str,
        extract_from: Optional[str] = None,
    ) -> None:
        """Initialize with file path."""
        user_agent = f"llama-index-zyte-api/{PYTHON_ZYTE_API_USER_AGENT}"

        client = ZyteAPI(
            api_key=api_key,
            user_agent=user_agent,
        )

        super().__init__(
            api_key=api_key,
            extract_from=extract_from,
            client=client,
        )

    def _serp_url(self, query: str):
        from urllib.parse import quote_plus

        base_url = "https://www.google.com/search?q="
        return base_url + quote_plus(query)

    def load_data(self, query: str):
        serp_url = self._serp_url(query)
        serp_request = {
            "url": serp_url,
            "serp": True,
        }
        if self.extract_from:
            serp_request.update({"serpOptions": {"extractFrom": self.extract_from}})
        results = self.client.get(serp_request)
        docs = []
        for result in results["serp"]["organicResults"]:
            doc = Document(
                text=result["url"],
                metadata={"name": result["name"], "rank": result["rank"]},
            )
            docs.append(doc)
        return docs
