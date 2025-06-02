import asyncio
import logging
from typing import List

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

logger = logging.getLogger(__name__)


class AsyncWebPageReader(BaseReader):
    """
    Asynchronous web page reader.

    Reads pages from the web asynchronously.

    Args:
        html_to_text (bool): Whether to convert HTML to text.
            Requires `html2text` package.
        limit (int): Maximum number of concurrent requests.
        dedupe (bool): to deduplicate urls if there is exact-match within given list
        fail_on_error (bool): if requested url does not return status code 200 the routine will raise an ValueError

    """

    def __init__(
        self,
        html_to_text: bool = False,
        limit: int = 10,
        dedupe: bool = True,
        fail_on_error: bool = False,
    ) -> None:
        """Initialize with parameters."""
        try:
            import html2text  # noqa: F401
        except ImportError:
            raise ImportError(
                "`html2text` package not found, please run `pip install html2text`"
            )
        try:
            import aiohttp  # noqa: F401
        except ImportError:
            raise ImportError(
                "`aiohttp` package not found, please run `pip install aiohttp`"
            )
        self._limit = limit
        self._html_to_text = html_to_text
        self._dedupe = dedupe
        self._fail_on_error = fail_on_error

    async def aload_data(self, urls: List[str]) -> List[Document]:
        """
        Load data from the input urls.

        Args:
            urls (List[str]): List of URLs to scrape.

        Returns:
            List[Document]: List of documents.

        """
        if self._dedupe:
            urls = list(dict.fromkeys(urls))

        import aiohttp

        def chunked_http_client(limit: int):
            semaphore = asyncio.Semaphore(limit)

            async def http_get(url: str, session: aiohttp.ClientSession):
                async with semaphore:
                    async with session.get(url) as response:
                        return response, await response.text()

            return http_get

        async def fetch_urls(urls: List[str]):
            http_client = chunked_http_client(self._limit)
            async with aiohttp.ClientSession() as session:
                tasks = [http_client(url, session) for url in urls]
                return await asyncio.gather(*tasks, return_exceptions=True)

        if not isinstance(urls, list):
            raise ValueError("urls must be a list of strings.")

        documents = []
        responses = await fetch_urls(urls)

        for i, response_tuple in enumerate(responses):
            if not isinstance(response_tuple, tuple):
                raise ValueError(f"One of the inputs is not a valid url: {urls[i]}")

            response, raw_page = response_tuple

            if response.status != 200:
                logger.warning(f"error fetching page from {urls[i]}")
                logger.info(response)

                if self._fail_on_error:
                    raise ValueError(
                        f"error fetching page from {urls[i]}. server returned status:"
                        f" {response.status} and response {raw_page}"
                    )

                continue

            if self._html_to_text:
                import html2text

                response_text = html2text.html2text(raw_page)
            else:
                response_text = raw_page

            documents.append(
                Document(text=response_text, extra_info={"Source": str(response.url)})
            )

        return documents

    def load_data(self, urls: List[str]) -> List[Document]:
        """
        Load data from the input urls.

        Args:
            urls (List[str]): List of URLs to scrape.

        Returns:
            List[Document]: List of documents.

        """
        return asyncio.run(self.aload_data(urls))
