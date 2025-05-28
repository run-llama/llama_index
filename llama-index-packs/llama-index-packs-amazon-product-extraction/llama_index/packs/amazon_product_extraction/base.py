"""Product extraction pack."""

import asyncio
from typing import Any, Dict

from llama_index.core import SimpleDirectoryReader
from llama_index.core.llama_pack.base import BaseLlamaPack
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.program.multi_modal_llm_program import (
    MultiModalLLMCompletionProgram,
)
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from pydantic import BaseModel, Field


async def _screenshot_page(
    url: str, out_path: str, width: int = 1200, height: int = 800
) -> None:
    from pyppeteer import launch

    browser = await launch()
    page = await browser.newPage()
    await page.setViewport({"width": 1200, "height": 800})
    await page.goto(url, {"waitUntil": "domcontentloaded"})
    await page.screenshot({"path": out_path})
    await browser.close()


class Product(BaseModel):
    """Data model for an Amazon Product."""

    title: str = Field(..., description="Title of product")
    category: str = Field(..., description="Category of product")
    discount: float = Field(..., description="Discount of product")
    price: float = Field(..., description="Price of product")
    rating: float = Field(..., description="Rating of product")
    description: str = Field(..., description="Description of product")
    img_description: str = Field(..., description="Description of product image")
    inventory: str = Field(..., description="Inventory of product")


DEFAULT_PROMPT_TEMPLATE_STR = """\
Can you extract the following fields from this product, in JSON format?
"""


class AmazonProductExtractionPack(BaseLlamaPack):
    """
    Product extraction pack.

    Given a website url of a product (e.g. Amazon page), screenshot it,
    and use GPT-4V to extract structured outputs.

    """

    def __init__(
        self,
        website_url: str,
        tmp_file_path: str = "./tmp.png",
        screenshot_width: int = 1200,
        screenshot_height: int = 800,
        prompt_template_str: str = DEFAULT_PROMPT_TEMPLATE_STR,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        self.website_url = website_url
        # download image to temporary file
        asyncio.get_event_loop().run_until_complete(
            _screenshot_page(
                website_url,
                tmp_file_path,
                width=screenshot_width,
                height=screenshot_height,
            )
        )

        # put your local directory here
        self.image_documents = SimpleDirectoryReader(
            input_files=[tmp_file_path]
        ).load_data()

        # initialize openai pydantic program
        self.openai_mm_llm = OpenAIMultiModal(
            model="gpt-4-vision-preview", max_new_tokens=1000
        )
        self.openai_program = MultiModalLLMCompletionProgram.from_defaults(
            output_parser=PydanticOutputParser(Product),
            image_documents=self.image_documents,
            prompt_template_str=prompt_template_str,
            llm=self.openai_mm_llm,
            verbose=True,
        )

    def get_modules(self) -> Dict[str, Any]:
        """Get modules."""
        return {
            "openai_program": self.openai_program,
            "openai_mm_llm": self.openai_mm_llm,
            "image_documents": self.image_documents,
        }

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Run the pipeline."""
        return self.openai_program(*args, **kwargs)
