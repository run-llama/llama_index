import logging
import os
from typing import Optional, Iterable, Sequence, AsyncIterable, Any, List
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


logger = logging.getLogger(__name__)
API_BASE_URL = "https://api.tzafon.ai"

class TzafonWebReader(BaseReader):
    def __init__(self, api_key: Optional[str] = None) -> None:
        try:
            from tzafon import Computer
        except ImportError:
            raise ImportError(
                "`tzafon` package not found, please run `pip install tzafon`"
            )

        self.api_key = api_key or os.getenv("TZAFON_API_KEY")

        if not self.api_key:
            raise ValueError("TZAFON_API_KEY is not set. Get your API key from https://tzafon.ai/dashboard")

        self.tzafon = Computer(api_key=self.api_key)
        logger.info("TzafonWebReader initialized")

    def load_data(self, *args: Any, **load_kwargs: Any) -> List[Document]:
        """Load data from the input directory."""
        return [doc for doc in self.lazy_load_data(*args, **load_kwargs)]

    async def aload_data(self, *args: Any, **load_kwargs: Any) -> List[Document]:
        """Load data from the input directory."""
        return [doc async for doc in self.alazy_load_data(*args, **load_kwargs)]

    def lazy_load_data(
        self,
        urls: Sequence[str],
        text_content: bool = True,
    ) -> Iterable[Document]:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            raise ImportError(
                "`playwright` package not found, please run `pip install playwright`"
            )

        computer = self.tzafon.create(kind="browser")
        computer_id = computer.id
        cdp_url = f"{API_BASE_URL}/computers/{computer_id}/cdp?token={self.api_key}"

        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.connect_over_cdp(cdp_url)
                
                context = browser.contexts[0] if browser.contexts else browser.new_context()

                for url in urls:
                    page = context.new_page()
                    
                    try:
                        page.goto(url)
                        if text_content:
                            content = str(page.inner_text("body"))
                        else:
                            content = str(page.content())
                        
                        yield Document(
                            text=content,
                            metadata={"url": url},
                        )
                    finally:
                        page.close()

                browser.close()

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
        finally:
            computer.terminate()

    async def alazy_load_data(
        self,
        urls: Sequence[str],
        text_content: bool = True,
    ) -> AsyncIterable[Document]:
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            raise ImportError(
                "`playwright` package not found, please run `pip install playwright`"
            )
        
        computer = self.tzafon.create(kind="browser")
        computer_id = computer.id
        cdp_url = f"{API_BASE_URL}/computers/{computer_id}/cdp?token={self.api_key}"

        try:
            async with async_playwright() as playwright:
                browser = await playwright.chromium.connect_over_cdp(cdp_url)
                
                context = browser.contexts[0] if browser.contexts else await browser.new_context()

                for url in urls:
                    page = await context.new_page()
                    
                    try:
                        await page.goto(url)
                        if text_content:
                            content = str(await page.inner_text("body"))
                        else:
                            content = str(await page.content())

                        yield Document(
                            text=content,
                            metadata={"url": url},
                        )
                    finally:
                        await page.close()
                
                await browser.close()

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
        finally:
            computer.terminate()
        
        