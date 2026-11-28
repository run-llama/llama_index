import os
import requests
from llama_index.core.tools.tool_spec.base import BaseToolSpec

class X402ScraperToolSpec(BaseToolSpec):
    """LlamaIndex tool specification for x402 paywalled scraping service."""
    
    spec_functions = ["scrape_page"]

    def __init__(self, api_url: str = None, evm_private_key: str = None):
        self.api_url = api_url or "https://x402-scraper-api-production-67a4.up.railway.app/api/scrape"
        self.evm_private_key = evm_private_key or os.getenv("EVM_PRIVATE_KEY")

    def scrape_page(self, url: str) -> str:
        """
        Scrapes a target web page and returns its content as Markdown.

        Args:
            url (str): The web page URL to scrape.
        """
        res = requests.post(self.api_url, json={"url": url})

        if res.status_code == 200:
            return res.json().get("markdown", "")
        elif res.status_code == 402:
            spec = res.json().get("accepts", {})
            return f"HTTP 402: Payment of {spec.get('price')} USDC required to target address {spec.get('payTo')} on Base."

        return f"Error: Received status {res.status_code} from API."
