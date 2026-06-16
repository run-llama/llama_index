"""BuyWhere product catalog tool spec."""

from typing import Dict, List, Optional

import requests
from llama_index.core.tools.tool_spec.base import BaseToolSpec


ENDPOINT_BASE_URL = "https://api.buywhere.ai/v1"


class BuyWhereToolSpec(BaseToolSpec):
    """BuyWhere product catalog tool spec.

    Search products, compare prices, and generate affiliate links across
    1.5M+ products from Shopee, Lazada, Amazon, Walmart, and 20+ retailers
    in Southeast Asia and the US.
    """

    spec_functions = [
        "search_products",
        "get_product",
        "compare_prices",
        "get_affiliate_link",
        "get_catalog",
    ]

    def __init__(
        self,
        api_key: str,
        marketplace: Optional[str] = None,
        country: Optional[str] = "US",
    ) -> None:
        """Initialize with BuyWhere API key.

        Args:
            api_key: BuyWhere API key (sign up at https://buywhere.ai).
            marketplace: Optional marketplace filter (e.g. ``"amazon"``,
                ``"shopee"``). ``None`` searches all marketplaces.
            country: Default country for results (ISO 3166-1 alpha-2).
                Defaults to ``"US"``.
        """
        self.api_key = api_key
        self.marketplace = marketplace
        self.country = country

    def _request(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
    ) -> Dict:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": "llama-index-tools-buywhere/0.1.0",
        }
        response = requests.get(
            f"{ENDPOINT_BASE_URL}{endpoint}",
            headers=headers,
            params=params or {},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def search_products(self, query: str, limit: Optional[int] = 5) -> List[Dict]:
        """Search the BuyWhere product catalog for a query.

        Use this for any product discovery task. Returns the top results
        ranked by relevance, with title, price, merchant, and URL.

        Args:
            query: Natural-language search query (e.g. ``"best gaming laptop under $1500"``).
            limit: Maximum number of products to return (1-50, default 5).

        """
        params: Dict = {"q": query, "limit": min(max(limit or 5, 1), 50), "country": self.country}
        if self.marketplace:
            params["marketplace"] = self.marketplace
        data = self._request("/products/search", params=params)
        return data.get("results", [])

    def get_product(self, product_id: str) -> Dict:
        """Fetch a single product's full details by BuyWhere product id.

        Args:
            product_id: The BuyWhere product id (e.g. ``"prod_abc123"``).

        """
        return self._request(f"/products/{product_id}")

    def compare_prices(self, product_id: str) -> List[Dict]:
        """Compare live prices for a product across every supported merchant.

        Args:
            product_id: The BuyWhere product id.

        """
        data = self._request(f"/products/{product_id}/prices")
        return data.get("offers", [])

    def get_affiliate_link(
        self,
        product_id: str,
        merchant: Optional[str] = None,
    ) -> Dict:
        """Generate an affiliate-tagged deep link to a product.

        Args:
            product_id: The BuyWhere product id.
            merchant: Optional merchant slug to deep-link to (defaults to the
                merchant with the lowest current price).

        """
        params: Dict = {}
        if merchant:
            params["merchant"] = merchant
        return self._request(
            f"/products/{product_id}/affiliate",
            params=params,
        )

    def get_catalog(self, category: str, limit: Optional[int] = 10) -> List[Dict]:
        """Browse the BuyWhere catalog for a category.

        Use this when the user wants to explore a category rather than search
        a specific query (e.g. "show me budget laptops").

        Args:
            category: Category slug or display name (e.g. ``"laptops"``,
                ``"smartphones"``, ``"running-shoes"``).
            limit: Maximum number of products to return (1-50, default 10).

        """
        params = {
            "category": category,
            "limit": min(max(limit or 10, 1), 50),
            "country": self.country,
        }
        data = self._request("/catalog", params=params)
        return data.get("results", [])
