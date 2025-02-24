"""Zyte Web Reader."""
import asyncio
import logging
from base64 import b64decode
from typing import Any, Dict, List, Literal, Optional
from pydantic import Field

from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document

logger = logging.getLogger(__name__)


class ZyteWebReader(BasePydanticReader):
    """Load text from URLs using `Zyte api`.

    Args:
        api_key: Zyte API key.
        mode: Determines how the text is extracted for the page content.
            It can take one of the following values: 'html', 'html-text', 'article'
        n_conn: It is the maximum number of concurrent requests to use.
        **download_kwargs: Any additional download arguments to pass for download.
            See: https://docs.zyte.com/zyte-api/usage/reference.html

    Example:
        .. code-block:: python

            from llama_index.readers.web import ZyteWebReader

            reader = ZyteWebReader(
               api_key="ZYTE_API_KEY",
            )
            docs = reader.load_data(
                urls=["<url-1>", "<url-2>"],
            )

    Zyte-API reference:
        https://www.zyte.com/zyte-api/

    """

    client_async: Optional[object] = Field(None)
    api_key: str
    mode: str
    n_conn: int
    download_kwargs: Optional[dict]
    continue_on_failure: bool

    def __init__(
        self,
        api_key: str,
        mode: Literal["article", "html", "html-text"] = "article",
        n_conn: int = 15,
        download_kwargs: Optional[Dict[str, Any]] = None,
        continue_on_failure: bool = True,
    ) -> None:
        """Initialize with file path."""
        super().__init__(
            api_key=api_key,
            mode=mode,
            n_conn=n_conn,
            download_kwargs=download_kwargs,
            continue_on_failure=continue_on_failure,
        )
        try:
            from zyte_api import AsyncZyteAPI
            from zyte_api.utils import USER_AGENT as PYTHON_ZYTE_API_USER_AGENT

        except ImportError:
            raise ImportError(
                "zyte-api package not found, please install it with "
                "`pip install zyte-api`"
            )
        if mode not in ("article", "html", "html-text"):
            raise ValueError(
                f"Unrecognized mode '{mode}'. Expected one of "
                f"'article', 'html', 'html-text'."
            )

        user_agent = f"llama-index-zyte-api/{PYTHON_ZYTE_API_USER_AGENT}"
        self.client_async = AsyncZyteAPI(
            api_key=api_key, user_agent=user_agent, n_conn=n_conn
        )

    @classmethod
    def class_name(cls) -> str:
        return "ZyteWebReader"

    def _zyte_html_option(self) -> str:
        if self.download_kwargs and "browserHtml" in self.download_kwargs:
            return "browserHtml"
        return "httpResponseBody"

    def _get_article(self, page: Dict) -> str:
        headline = page["article"].get("headline", "")
        article_body = page["article"].get("articleBody", "")
        return headline + "\n\n" + article_body

    def _zyte_request_params(self, url: str) -> dict:
        request_params: Dict[str, Any] = {"url": url}
        if self.mode == "article":
            request_params.update({"article": True})

        if self.mode in ("html", "html-text"):
            request_params.update({self._zyte_html_option(): True})

        if self.download_kwargs:
            request_params.update(self.download_kwargs)
        return request_params

    async def fetch_items(self, urls) -> List:
        results = []
        queries = [self._zyte_request_params(url) for url in urls]
        async with self.client_async.session() as session:
            for i, future in enumerate(session.iter(queries)):
                try:
                    result = await future
                    results.append(result)
                except Exception as e:
                    url = queries[i]["url"]
                    if self.continue_on_failure:
                        logger.warning(
                            f"Error {e} while fetching url {url}, "
                            f"skipping because continue_on_failure is True"
                        )
                        continue
                    else:
                        logger.exception(
                            f"Error fetching {url} and aborting, use "
                            f"continue_on_failure=True to continue loading "
                            f"urls after encountering an error."
                        )
                        raise
        return results

    def _get_content(self, response: Dict) -> str:
        if self.mode == "html-text":
            try:
                from html2text import html2text

            except ImportError:
                raise ImportError(
                    "html2text package not found, please install it with "
                    "`pip install html2text`"
                )
        if self.mode in ("html", "html-text"):
            content = response[self._zyte_html_option()]

            if self._zyte_html_option() == "httpResponseBody":
                content = b64decode(content).decode()

            if self.mode == "html-text":
                content = html2text(content)
        elif self.mode == "article":
            content = self._get_article(response)
        return content

    def load_data(self, urls) -> List[Document]:
        docs = []
        responses = asyncio.run(self.fetch_items(urls))
        for response in responses:
            content = self._get_content(response)
            doc = Document(text=content, metadata={"url": response["url"]})
            docs.append(doc)
        return docs
