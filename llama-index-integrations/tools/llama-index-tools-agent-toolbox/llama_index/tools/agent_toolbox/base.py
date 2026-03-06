"""Agent Toolbox tool spec for LlamaIndex.

Provides 13 production-ready tools for AI agents:
search, extract, screenshot, weather, finance, validate_email,
translate, geoip, news, whois, dns, pdf_extract, qr_generate.

Get a free API key at https://api.sendtoclaw.com/v1/auth/register
"""

from typing import Any, Dict, List, Optional

import requests
from llama_index.core.schema import Document
from llama_index.core.tools.tool_spec.base import BaseToolSpec

DEFAULT_BASE_URL = "https://api.sendtoclaw.com"


class AgentToolboxToolSpec(BaseToolSpec):
    """Agent Toolbox tool spec — 13 tools for AI agents.

    Provides web search, content extraction, screenshots, weather, finance,
    email validation, translation, GeoIP, news, WHOIS, DNS, PDF extraction,
    and QR code generation through a single API.
    """

    spec_functions = [
        "search",
        "extract",
        "screenshot",
        "weather",
        "finance",
        "validate_email",
        "translate",
        "geoip",
        "news",
        "whois",
        "dns",
        "pdf_extract",
        "qr_generate",
    ]

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 30.0,
    ) -> None:
        """Initialize with API key.

        Args:
            api_key: Agent Toolbox API key (starts with atb_).
            base_url: API base URL.
            timeout: Request timeout in seconds.
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _post(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make authenticated POST request."""
        resp = requests.post(
            f"{self.base_url}{endpoint}",
            json=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def search(self, query: str, count: int = 5) -> List[Document]:
        """Search the web using DuckDuckGo.

        Args:
            query: Search query string.
            count: Number of results (1-10).

        Returns:
            List of Documents with search results.
        """
        result = self._post("/v1/search", {"query": query, "count": count})
        data = result.get("data", [])
        docs = []
        for item in data:
            text = f"Title: {item.get('title', '')}\nURL: {item.get('url', '')}\nSnippet: {item.get('snippet', '')}"
            docs.append(Document(text=text, metadata={"url": item.get("url", "")}))
        return docs if docs else [Document(text=str(data))]

    def extract(self, url: str, format: str = "markdown") -> List[Document]:
        """Extract readable content from a web page.

        Args:
            url: URL to extract content from.
            format: Output format (markdown, text, json).

        Returns:
            List of Documents with extracted content.
        """
        result = self._post("/v1/extract", {"url": url, "format": format})
        data = result.get("data", {})
        content = data.get("content", str(data))
        return [Document(text=content, metadata={"url": url, "title": data.get("metadata", {}).get("title", "")})]

    def screenshot(self, url: str, width: int = 1280, height: int = 720) -> List[Document]:
        """Capture a screenshot of a web page.

        Args:
            url: URL to screenshot.
            width: Viewport width in pixels.
            height: Viewport height in pixels.

        Returns:
            List of Documents with base64-encoded screenshot data.
        """
        result = self._post("/v1/screenshot", {"url": url, "width": width, "height": height})
        data = result.get("data", {})
        return [Document(text=str(data), metadata={"url": url})]

    def weather(self, location: str) -> List[Document]:
        """Get current weather and forecast for a location.

        Args:
            location: City name or location string.

        Returns:
            List of Documents with weather data.
        """
        result = self._post("/v1/weather", {"location": location})
        data = result.get("data", {})
        return [Document(text=str(data), metadata={"location": location})]

    def finance(
        self,
        symbol: Optional[str] = None,
        type: str = "quote",
        from_currency: Optional[str] = None,
        to_currency: Optional[str] = None,
        amount: Optional[float] = None,
    ) -> List[Document]:
        """Get stock quotes or currency exchange rates.

        Args:
            symbol: Stock ticker (e.g. AAPL) for quotes.
            type: Request type (quote or exchange).
            from_currency: Source currency code for exchange.
            to_currency: Target currency code for exchange.
            amount: Amount to convert for exchange.

        Returns:
            List of Documents with financial data.
        """
        payload: Dict[str, Any] = {}
        if symbol:
            payload = {"symbol": symbol, "type": type}
        elif from_currency and to_currency:
            payload = {"from": from_currency, "to": to_currency, "amount": amount or 1}
        result = self._post("/v1/finance", payload)
        return [Document(text=str(result.get("data", {})))]

    def validate_email(self, email: str) -> List[Document]:
        """Validate an email address (syntax, MX, SMTP, disposable check).

        Args:
            email: Email address to validate.

        Returns:
            List of Documents with validation results.
        """
        result = self._post("/v1/validate-email", {"email": email})
        return [Document(text=str(result.get("data", {})), metadata={"email": email})]

    def translate(
        self, text: str, target: str, source: str = "auto"
    ) -> List[Document]:
        """Translate text between 100+ languages.

        Args:
            text: Text to translate.
            target: Target language code (e.g. zh, ja, es).
            source: Source language code or 'auto'.

        Returns:
            List of Documents with translation.
        """
        result = self._post("/v1/translate", {"text": text, "target": target, "source": source})
        data = result.get("data", {})
        translation = data.get("translation", str(data))
        return [Document(text=translation, metadata={"source": source, "target": target})]

    def geoip(self, ip: str) -> List[Document]:
        """Look up geolocation for an IP address.

        Args:
            ip: IP address to geolocate.

        Returns:
            List of Documents with geolocation data.
        """
        result = self._post("/v1/geoip", {"ip": ip})
        return [Document(text=str(result.get("data", {})), metadata={"ip": ip})]

    def news(
        self,
        query: str,
        category: Optional[str] = None,
        language: str = "en",
        limit: int = 5,
    ) -> List[Document]:
        """Search for recent news articles.

        Args:
            query: News search query.
            category: Category filter (business, technology, science, etc).
            language: Language code (default: en).
            limit: Number of articles (1-20).

        Returns:
            List of Documents with news articles.
        """
        payload: Dict[str, Any] = {"query": query, "language": language, "limit": limit}
        if category:
            payload["category"] = category
        result = self._post("/v1/news", payload)
        data = result.get("data", {})
        articles = data.get("results", [])
        docs = []
        for a in articles:
            text = f"Title: {a.get('title', '')}\nSource: {a.get('source', '')}\nURL: {a.get('url', '')}"
            docs.append(Document(text=text, metadata={"url": a.get("url", "")}))
        return docs if docs else [Document(text=str(data))]

    def whois(self, domain: str) -> List[Document]:
        """Look up WHOIS information for a domain.

        Args:
            domain: Domain name (e.g. google.com).

        Returns:
            List of Documents with WHOIS data.
        """
        result = self._post("/v1/whois", {"domain": domain})
        return [Document(text=str(result.get("data", {})), metadata={"domain": domain})]

    def dns(self, domain: str, type: str = "A") -> List[Document]:
        """Query DNS records for a domain.

        Args:
            domain: Domain name to query.
            type: Record type (A, AAAA, CNAME, MX, NS, TXT, SOA, SRV, CAA).

        Returns:
            List of Documents with DNS records.
        """
        result = self._post("/v1/dns", {"domain": domain, "type": type})
        return [Document(text=str(result.get("data", {})), metadata={"domain": domain, "type": type})]

    def pdf_extract(self, url: str, max_pages: Optional[int] = None) -> List[Document]:
        """Extract text content from a PDF file.

        Args:
            url: URL of the PDF to extract.
            max_pages: Maximum pages to extract.

        Returns:
            List of Documents with extracted text.
        """
        payload: Dict[str, Any] = {"url": url}
        if max_pages is not None:
            payload["maxPages"] = max_pages
        result = self._post("/v1/pdf-extract", payload)
        data = result.get("data", {})
        text = data.get("text", str(data))
        return [Document(text=text, metadata={"url": url})]

    def qr_generate(
        self, text: str, format: str = "png", width: int = 300
    ) -> List[Document]:
        """Generate a QR code image.

        Args:
            text: Text or URL to encode.
            format: Output format (png or svg).
            width: Image width in pixels.

        Returns:
            List of Documents with QR code data.
        """
        result = self._post("/v1/qr", {"text": text, "format": format, "width": width})
        return [Document(text=str(result.get("data", {})), metadata={"text": text})]
