from typing import Any, Optional
import requests
from llama_index.core.tools.tool_spec.base import BaseToolSpec

class X402ScraperToolSpec(BaseToolSpec):
    """x402 Scraper Tool.
    
    A token-efficient markdown extraction tool designed for RAG pipelines that is 
    inherently protected from rate limits using the emerging x402 machine-to-machine payment standard.
    """
    spec_functions = ["scrape_url"]

    def __init__(self, endpoint_url: str = "https://x402-api-middleware.onrender.com/scrape") -> None:
        """Initialize with the x402 scraper endpoint."""
        self.endpoint_url = endpoint_url

    def scrape_url(self, url: str) -> str:
        """
        Extract clean markdown from a URL.
        
        Args:
            url (str): The URL to scrape.
            
        Returns:
            str: The extracted markdown content.
        """
        response = requests.get(self.endpoint_url, params={"url": url})
        
        if response.status_code == 402:
            return "Error 402: Payment Required. x402 negotiation required via @coinbase/cdp-sdk."
        elif response.status_code == 200:
            return response.text
        else:
            return f"Error {response.status_code}: {response.text}"
