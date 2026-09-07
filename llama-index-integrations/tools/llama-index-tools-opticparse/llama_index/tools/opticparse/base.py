from typing import Optional, List, Dict, Any
import os
import requests

try:
    from llama_index.core.schema import Document
    from llama_index.core.tools.tool_spec.base import BaseToolSpec
except ImportError:
    class Document:  # type: ignore[no-redef]
        def __init__(self, text: str, extra_info: Optional[Dict[str, Any]] = None):
            self.text = text
            self.extra_info = extra_info or {}

        def __repr__(self) -> str:
            return f"Document(text={self.text[:50]!r}..., extra_info={self.extra_info})"

    class BaseToolSpec:  # type: ignore[no-redef]
        spec_functions: List[str] = []

        def to_tool_list(self) -> List[Any]:
            tools = []
            for func_name in self.spec_functions:
                func = getattr(self, func_name, None)
                if func:
                    tools.append(func)
            return tools


class OpticParseToolSpec(BaseToolSpec):
    """
    OpticParse & PhishVision Tool Spec for LlamaIndex Agents.

    Enables autonomous visual web extraction resilient against anti-bot challenges,
    and real-time zero-day cybersecurity & Web3 wallet-drainer detection.
    """

    spec_functions = [
        "extract",
        "inspect_threat",
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
    ) -> None:
        """Initialize OpticParse LlamaIndex ToolSpec."""
        self.api_key = api_key or os.getenv("OPTICPARSE_API_KEY", "op_live_llamaindex_agent")
        self.endpoint = endpoint or os.getenv(
            "OPTICPARSE_ENDPOINT",
            "https://opticparse-mcp-portal.parastejpal987.workers.dev"
        )

    def extract(
        self,
        url: str,
        query: str = "Extract all main content, pricing, specifications, and structured data.",
    ) -> List[Document]:
        """
        Extract structured web data using OpticParse Vision Web Scraper.

        Args:
            url (str): The target webpage URL to visually scrape.
            query (str): Natural language instruction describing data to extract.

        Returns:
            List[Document]: Extracted content wrapped in LlamaIndex Document format.
        """
        target_endpoint = f"{self.endpoint}/mcp/tools/opticparse_extract"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": "LlamaIndex-OpticParse/1.0",
        }
        payload = {"url": url, "query": query}

        try:
            res = requests.post(target_endpoint, json=payload, headers=headers, timeout=30)
            res.raise_for_status()
            data = res.json()
            text_content = data.get("content") or data.get("markdown") or str(data)
            return [Document(text=str(text_content), extra_info={"url": url, "status": "success"})]
        except Exception as e:
            return [
                Document(
                    text=f"OpticParse extraction error: {str(e)}",
                    extra_info={"url": url, "status": "error", "error": str(e)},
                )
            ]

    def inspect_threat(
        self,
        url: str,
    ) -> List[Document]:
        """
        Scan a target domain or URL for zero-day phishing kits, crypto wallet drainers, and malicious traps.

        Args:
            url (str): Target URL or domain to audit.

        Returns:
            List[Document]: Threat intelligence scan results wrapped in LlamaIndex Document format.
        """
        target_endpoint = f"{self.endpoint}/phishvision/scan"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": "LlamaIndex-PhishVision/1.0",
        }
        payload = {"url": url}

        try:
            res = requests.post(target_endpoint, json=payload, headers=headers, timeout=15)
            res.raise_for_status()
            data = res.json()
            return [Document(text=str(data), extra_info={"url": url, "scan_status": "complete"})]
        except Exception as e:
            return [
                Document(
                    text=f"PhishVision threat detection error: {str(e)}",
                    extra_info={"url": url, "scan_status": "error", "error": str(e)},
                )
            ]


__all__ = ["OpticParseToolSpec", "Document", "BaseToolSpec"]
