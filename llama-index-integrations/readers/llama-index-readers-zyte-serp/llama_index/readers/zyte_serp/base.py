from typing import Literal, Optional
from pydantic import Field

from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document


class ZyteSerpReader(BasePydanticReader):
    """Get google search results URLs for a search query.

    Args:
        api_key: Zyte API key.
        extract_from: Determines the mode while extracting the .
            It can take one of the following values: 'html', 'html-text', 'article'

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
        https://www.zyte.com/zyte-api/

    """

    client: Optional[object] = Field(None)
    api_key: str
    extract_from: str

    def __init__(
        self,
        api_key: str,
        extract_from: Literal["httpResponseBody", "browserHtml"] = "httpResponseBody",
    ) -> None:
        """Initialize with file path."""
        super().__init__(
            api_key=api_key,
            extract_from=extract_from,
        )
        try:
            from zyte_api import ZyteAPI
            from zyte_api.utils import USER_AGENT as PYTHON_ZYTE_API_USER_AGENT

        except ImportError:
            raise ImportError(
                "zyte-api package not found, please install it with "
                "`pip install zyte-api`"
            )

        user_agent = f"llama-index-zyte-api/{PYTHON_ZYTE_API_USER_AGENT}"
        self.client = ZyteAPI(
            api_key=api_key,
            user_agent=user_agent,
        )

    def _serp_url(self, query: str):
        base_url = "https://www.google.com/search?q="
        return base_url + query.replace(" ", "+")

    def load_data(self, query: str):
        serp_url = self._serp_url(query)
        serp_request = {
            "url": serp_url,
            "serp": True,
            "serpOptions": {"extractFrom": self.extract_from},
        }
        results = self.client.get(serp_request)
        docs = []
        for result in results["serp"]["organicResults"]:
            doc = Document(
                text=result["url"],
                metadata={"name": result["name"], "rank": result["rank"]},
            )
            docs.append(doc)
        return docs
