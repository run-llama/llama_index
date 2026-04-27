"""ComparEdge Data Reader for LlamaIndex."""

from typing import List, Optional

import requests

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

_PAGE_SIZE = 100  # max records per request


class ComparEdgeReader(BaseReader):
    """Read SaaS product data from ComparEdge API.

    331 products across 28 categories. Pricing, ratings, features.
    No API key needed.

    Args:
        category: Optional category slug (e.g. "project-management", "crm").
                  When omitted, loads all 331 products across all categories.

    Examples:
        Load everything::

            from comparedge_reader import ComparEdgeReader
            reader = ComparEdgeReader()
            docs = reader.load_data()

        Load a single category::

            reader = ComparEdgeReader(category="crm")
            docs = reader.load_data()
    """

    BASE_URL = "https://comparedge-api.up.railway.app/api/v1"

    def __init__(self, category: Optional[str] = None) -> None:
        self.category = category

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_page(self, params: dict) -> dict:
        r = requests.get(f"{self.BASE_URL}/products", params=params, timeout=30)
        r.raise_for_status()
        return r.json()

    def _fetch_all_products(self) -> List[dict]:
        """Paginate through all products, respecting the API's limit/offset."""
        params: dict = {"limit": _PAGE_SIZE, "offset": 0}
        if self.category:
            params["category"] = self.category

        products: List[dict] = []
        while True:
            data = self._fetch_page(params)
            batch = data.get("products", [])
            products.extend(batch)
            total = data.get("total") or 0
            if len(products) >= total or not batch:
                break
            params["offset"] = len(products)

        return products

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_data(self) -> List[Document]:
        """Fetch products from ComparEdge and return as LlamaIndex Documents.

        Each Document contains a plain-text summary with name, description,
        and pricing plans. Metadata includes slug, category, G2 rating,
        free-tier flag, and canonical URL.

        Returns:
            List of Document objects, one per product.

        Raises:
            requests.HTTPError: If the API returns a non-2xx status.
        """
        products = self._fetch_all_products()

        documents = []
        for p in products:
            text_parts = [
                f"{p['name']} ({p.get('category', '')})",
                p.get("description", ""),
            ]

            pricing = p.get("pricing", {}) or {}
            for plan in pricing.get("plans", []):
                price = plan.get("price")
                plan_name = plan.get("name", "")
                period = plan.get("period", "mo")
                if price and price > 0:
                    text_parts.append(f"{plan_name}: ${price}/{period}")
                elif price == 0:
                    text_parts.append(f"{plan_name}: Free")
                # price is None → Enterprise/custom pricing; skip

            ratings = p.get("rating", p.get("ratings", {})) or {}
            metadata = {
                "slug": p.get("slug", ""),
                "category": p.get("category", ""),
                "g2_rating": ratings.get("g2"),
                "has_free_tier": pricing.get("free", False),
                "url": f"https://comparedge.com/tools/{p.get('slug', '')}",
            }

            documents.append(
                Document(text="\n".join(filter(None, text_parts)), metadata=metadata)
            )

        return documents
