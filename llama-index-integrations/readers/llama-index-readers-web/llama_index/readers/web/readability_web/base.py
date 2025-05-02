import asyncio
import unicodedata
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional

from llama_index.core.node_parser.interface import TextSplitter
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from playwright.async_api._generated import Browser

path = Path(__file__).parent / "Readability.js"


def nfkc_normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def async_to_sync(awaitable):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(awaitable)


class ReadabilityWebPageReader(BaseReader):
    """Readability Webpage Loader.

    Extracting relevant information from a fully rendered web page.
    During the processing, it is always assumed that web pages used as data sources contain textual content.

    1. Load the page and wait for it rendered. (playwright)
    2. Inject Readability.js to extract the main content.

    Args:
        proxy (Optional[str], optional): Proxy server. Defaults to None.
        wait_until (Optional[Literal["commit", "domcontentloaded", "load", "networkidle"]], optional): Wait until the page is loaded. Defaults to "domcontentloaded".
        text_splitter (TextSplitter, optional): Text splitter. Defaults to None.
        normalizer (Optional[Callable[[str], str]], optional): Text normalizer. Defaults to nfkc_normalize.
    """

    def __init__(
        self,
        proxy: Optional[str] = None,
        wait_until: Optional[
            Literal["commit", "domcontentloaded", "load", "networkidle"]
        ] = "domcontentloaded",
        text_splitter: Optional[TextSplitter] = None,
        normalize: Optional[Callable[[str], str]] = nfkc_normalize,
    ) -> None:
        self._launch_options = {
            "headless": True,
        }
        self._wait_until = wait_until
        if proxy:
            self._launch_options["proxy"] = {
                "server": proxy,
            }
        self._text_splitter = text_splitter
        self._normalize = normalize
        self._readability_js = None

    async def async_load_data(self, url: str) -> List[Document]:
        """Render and load data content from url.

        Args:
            url (str): URL to scrape.

        Returns:
            List[Document]: List of documents.

        """
        from playwright.async_api import async_playwright

        async with async_playwright() as async_playwright:
            browser = await async_playwright.chromium.launch(**self._launch_options)

            article = await self.scrape_page(
                browser,
                url,
            )
            extra_info = {
                key: article[key]
                for key in [
                    "title",
                    "length",
                    "excerpt",
                    "byline",
                    "dir",
                    "lang",
                    "siteName",
                ]
            }

            if self._normalize is not None:
                article["textContent"] = self._normalize(article["textContent"])
            texts = []
            if self._text_splitter is not None:
                texts = self._text_splitter.split_text(article["textContent"])
            else:
                texts = [article["textContent"]]

            await browser.close()

            return [Document(text=x, extra_info=extra_info) for x in texts]

    def load_data(self, url: str) -> List[Document]:
        return async_to_sync(self.async_load_data(url))

    async def scrape_page(
        self,
        browser: Browser,
        url: str,
    ) -> Dict[str, str]:
        """Scrape a single article url.

        Args:
            browser (Any): a Playwright Chromium browser.
            url (str): URL of the article to scrape.

        Returns:
            Ref: https://github.com/mozilla/readability
            title: article title;
            content: HTML string of processed article content;
            textContent: text content of the article, with all the HTML tags removed;
            length: length of an article, in characters;
            excerpt: article description, or short excerpt from the content;
            byline: author metadata;
            dir: content direction;
            siteName: name of the site.
            lang: content language

        """
        if self._readability_js is None:
            with open(path) as f:
                self._readability_js = f.read()

        inject_readability = f"""
            (function(){{
            {self._readability_js}
            function executor() {{
                return new Readability({{}}, document).parse();
            }}
            return executor();
            }}())
        """

        # browser = cast(Browser, browser)
        page = await browser.new_page(ignore_https_errors=True)
        page.set_default_timeout(60000)
        await page.goto(url, wait_until=self._wait_until)

        r = await page.evaluate(inject_readability)

        await page.close()
        print("scraped:", url)

        return r
