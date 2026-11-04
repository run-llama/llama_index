import logging
from typing import Iterator, Optional, Sequence

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

logger = logging.getLogger(__name__)


class AntibrowWebReader(BaseReader):
    """
    AntibrowWebReader.

    Load pages in a persistent AntiBrow browser profile running on your own machine.
    The profile keeps cookies, storage, an engine-level fingerprint and its own proxy
    between runs, so a page behind a login loads already signed in instead of starting
    over in a fresh browser.
    Depends on `antibrow` package.
    Get your API key from https://antibrow.com
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        profile: str = "llamaindex",
        proxy: Optional[str] = None,
        temporary: bool = False,
        headless: bool = False,
    ) -> None:
        try:
            from antibrow import launch  # noqa: F401
        except ImportError:
            raise ImportError(
                "`antibrow` package not found, please run `pip install antibrow`"
            )

        self.api_key = api_key
        self.profile = profile
        self.proxy = proxy
        self.temporary = temporary
        self.headless = headless

    def lazy_load_data(
        self,
        urls: Sequence[str],
        selector: Optional[str] = None,
    ) -> Iterator[Document]:
        """
        Load pages from URLs.

        Args:
            urls: The URLs to load, in order.
            selector: Optional CSS selector to read instead of the whole page.

        """
        from antibrow import launch

        browser = launch(
            self.profile,
            api_key=self.api_key,
            proxy=self.proxy,
            temporary=self.temporary,
            headless=self.headless,
            focus_window=False,
        )
        try:
            page = browser.new_page()
            for url in urls:
                response = page.goto(url, wait_until="load")
                yield Document(
                    text=page.locator(selector or "body").first.inner_text(),
                    metadata={
                        "url": page.url,
                        "title": page.title(),
                        "status": response.status if response is not None else None,
                        "profile": self.profile,
                    },
                )
        finally:
            browser.close()


if __name__ == "__main__":
    reader = AntibrowWebReader(temporary=True)
    logger.info(reader.load_data(urls=["https://example.com"]))
