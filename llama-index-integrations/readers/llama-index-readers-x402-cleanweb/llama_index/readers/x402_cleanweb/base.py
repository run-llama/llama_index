"""
LlamaIndex Official Data Reader Specification for x402 CleanWeb Agent
--------------------------------------------------------------------------------
Enables autonomous AI agents and RAG pipelines to ingest clean, noise-free web
documents with 87% token savings, anti-prompt injection defense, and x402 micropayments.

Complies with LlamaIndex BaseReader interface returning List[Document].
"""

from __future__ import annotations

import os
import secrets
import time
from typing import Any, Dict, List, Optional
import requests

try:
    from llama_index.core.readers.base import BaseReader
    from llama_index.core.schema import Document
except ImportError:
    try:
        from llama_index.readers.base import BaseReader
        from llama_index.readers.schema.base import Document
    except ImportError:
        # Lightweight standalone fallback if llama_index is not installed
        class BaseReader:
            pass

        class Document:
            def __init__(self, text: str = "", metadata: Optional[Dict[str, Any]] = None):
                self.text = text
                self.metadata = metadata or {}

            def __repr__(self):
                return f"Document(metadata={self.metadata}, text_length={len(self.text)})"


class X402CleanWebReader(BaseReader):
    """
    LlamaIndex Document Reader for CleanWeb Studio.
    Fetches web pages, strips 87% boilerplate/ads/injections, and returns
    structured LlamaIndex Documents with complete source provenance and metadata.
    """

    def __init__(
        self,
        gateway_url: Optional[str] = None,
        vault_key: Optional[str] = None,
        timeout: int = 25
    ):
        self.gateway_url = (
            gateway_url
            or os.getenv("X402_GATEWAY_URL", "https://x402-cleanweb-agent-7qxtp3324q-du.a.run.app")
        ).rstrip("/")
        self.vault_key = vault_key or os.getenv("AGENT_VAULT_KEY")
        self.timeout = timeout

    def _build_headers(self) -> Dict[str, str]:
        session_nonce = f"llamaindex_{secrets.token_hex(4)}_{int(time.time())}"
        headers = {
            "User-Agent": "LlamaIndex-Community-x402/1.0",
            "X-Agent-Nonce": session_nonce,
            "Accept": "application/json"
        }
        if self.vault_key:
            headers["X-Agent-Pass"] = self.vault_key
            headers["X-Vault-Key"] = self.vault_key
        return headers

    def load_data(
        self,
        urls: List[str] | str,
        preserve_alt_text: bool = True,
        dynamic_render: bool = False,
        scroll_depth: int = 0
    ) -> List[Document]:
        """
        Loads and purifies a list of URLs (or a single URL) into LlamaIndex Documents.
        """
        if isinstance(urls, str):
            urls = [urls]

        documents: List[Document] = []
        headers = self._build_headers()

        for target_url in urls:
            try:
                resp = requests.get(
                    f"{self.gateway_url}/api/v1/clean-web",
                    params={
                        "url": target_url,
                        "preserve_alt_text": preserve_alt_text,
                        "dynamic_render": dynamic_render,
                        "scroll_depth": scroll_depth
                    },
                    headers=headers,
                    timeout=self.timeout
                )

                if resp.status_code == 200:
                    data = resp.json()
                    content = data.get("markdown_content") or data.get("content", "")
                    sec_audit = data.get("security_audit", {})
                    token_analytics = data.get("token_analytics", {})
                    spa_info = data.get("spa_hydration", {})

                    metadata = {
                        "source": data.get("url", target_url),
                        "title": data.get("title", "Untitled Document"),
                        "content_hash": sec_audit.get("content_hash", ""),
                        "is_safe": sec_audit.get("is_safe", True),
                        "word_count": data.get("word_count", 0),
                        "estimated_reading_time_sec": data.get("estimated_reading_time_sec", 0),
                        "token_reduction_percent": token_analytics.get("token_reduction_percent", 87.0),
                        "spa_framework": spa_info.get("framework"),
                        "ingestion_engine": "x402_cleanweb_studio"
                    }

                    documents.append(Document(text=content, metadata=metadata))
                elif resp.status_code == 402:
                    error_doc = Document(
                        text=f"[x402 Payment Required]: Free trial exhausted for URL: {target_url}",
                        metadata={"source": target_url, "error": "402_payment_required"}
                    )
                    documents.append(error_doc)
                else:
                    error_doc = Document(
                        text=f"[CleanWeb Error {resp.status_code}]: {resp.text[:200]}",
                        metadata={"source": target_url, "error": f"http_{resp.status_code}"}
                    )
                    documents.append(error_doc)

            except Exception as e:
                error_doc = Document(
                    text=f"[CleanWeb Exception]: {str(e)}",
                    metadata={"source": target_url, "error": str(e)}
                )
                documents.append(error_doc)

        return documents

