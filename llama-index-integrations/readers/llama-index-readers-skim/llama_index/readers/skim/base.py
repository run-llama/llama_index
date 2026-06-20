"""LlamaIndex reader for Skim — the x402-native clean reader API for AI agents.

Exposes :class:`SkimReader`, a LlamaIndex ``BaseReader`` that fetches any URL and
returns a :class:`~llama_index.core.schema.Document` of clean, agent-ready
Markdown plus structured metadata. Each call is paid automatically over the x402
protocol ($0.002 in USDC on Base) using a wallet you control. The private key
never leaves your machine — it only signs an EIP-3009 USDC authorization locally.
"""

from __future__ import annotations

import os
from typing import Any, List, Optional, Union

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

DEFAULT_BASE_URL = "https://skim402.com"

_SIMPLE_TYPES = (str, int, float, bool)


class SkimReader(BaseReader):
    """Read any URL as clean Markdown via Skim, paying per call over x402.

    The reader lazily builds a payment-aware HTTP session the first time it runs,
    using your Base wallet's private key to sign USDC authorizations on demand.

    Args:
        private_key: Hex private key (with or without ``0x``) for the Base wallet
            that pays for reads. Falls back to the ``SKIM_WALLET_PRIVATE_KEY``
            environment variable. Use a dedicated wallet, never your personal one.
        base_url: Skim API base URL. Defaults to ``https://skim402.com``.
        max_price_usd: Hard per-call price cap in USD. The wallet refuses to sign
            for anything above this. Defaults to ``0.01`` (Skim is ``$0.002``).
        include_metadata: When ``True`` (default), populate each ``Document``'s
            ``metadata`` with the page metadata (title, byline, published date,
            language, excerpt) returned by Skim.
        timeout: Per-request timeout in seconds. Defaults to ``60``.

    Example:
        .. code-block:: python

            from llama_index.readers.skim import SkimReader

            reader = SkimReader()  # reads SKIM_WALLET_PRIVATE_KEY from the env
            docs = reader.load_data(urls=["https://en.wikipedia.org/wiki/HTTP_402"])
    """

    def __init__(
        self,
        private_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        max_price_usd: float = 0.01,
        include_metadata: bool = True,
        timeout: float = 60.0,
    ) -> None:
        super().__init__()
        self.private_key = private_key
        self.base_url = base_url
        self.max_price_usd = max_price_usd
        self.include_metadata = include_metadata
        self.timeout = timeout
        self._session: Any = None

    def _get_session(self) -> Any:
        """Build (and cache) a requests Session that auto-pays 402 responses."""
        if self._session is not None:
            return self._session

        try:
            import requests
            from eth_account import Account
            from x402 import x402ClientSync
            from x402.client import max_amount
            from x402.http.clients.requests import wrapRequestsWithPayment
            from x402.mechanisms.evm.exact.register import register_exact_evm_client
            from x402.mechanisms.evm.signers import EthAccountSigner
        except ImportError as exc:  # pragma: no cover - import-guard
            raise ImportError(
                "llama-index-readers-skim needs the x402 client with EVM support. "
                "Install it with:  pip install llama-index-readers-skim  (which pulls "
                "x402[evm]). If you installed manually, run:  pip install 'x402[evm]' "
                "requests eth-account"
            ) from exc

        key = (
            self.private_key
            if self.private_key is not None
            else os.environ.get("SKIM_WALLET_PRIVATE_KEY")
        )
        if not key:
            raise ValueError(
                "Skim requires payment via x402. Provide a Base wallet funded with "
                "USDC by setting the SKIM_WALLET_PRIVATE_KEY environment variable, or "
                "by passing private_key=... to SkimReader(). The key never leaves your "
                "machine — it only signs payment authorizations locally."
            )

        normalized = key[2:] if key.startswith("0x") else key
        if len(normalized) != 64 or any(
            c not in "0123456789abcdefABCDEF" for c in normalized
        ):
            raise ValueError(
                "SKIM_WALLET_PRIVATE_KEY must be a 64-character hex string (with or "
                "without a 0x prefix)."
            )

        account = Account.from_key("0x" + normalized)
        cap_atomic = int(round(self.max_price_usd * 1_000_000))  # USDC has 6 decimals
        client = x402ClientSync()
        register_exact_evm_client(
            client,
            EthAccountSigner(account),
            policies=[max_amount(cap_atomic)],
        )
        self._session = wrapRequestsWithPayment(requests.Session(), client)
        return self._session

    def _read_one(self, url: str) -> Document:
        session = self._get_session()
        endpoint = self.base_url.rstrip("/") + "/api/v1/read"

        try:
            res = session.post(
                endpoint,
                json={"url": url, "mode": "basic"},
                timeout=self.timeout,
            )
        except Exception as exc:  # network / payment-signing failure
            raise RuntimeError(
                f"Skim request failed: {exc}. Common causes: the wallet has no USDC "
                f"on Base, or the price exceeded max_price_usd (${self.max_price_usd})."
            ) from exc

        if not getattr(res, "ok", res.status_code < 400):
            body = (res.text or "").strip()
            raise RuntimeError(
                f"Skim returned {res.status_code} {getattr(res, 'reason', '')}: "
                f"{body or '(no body)'}"
            )

        try:
            data = res.json()
        except ValueError as exc:
            raise RuntimeError(
                "Skim returned a non-JSON response. This usually means the request "
                f"did not reach the Skim API. Underlying error: {exc}"
            ) from exc

        markdown = data.get("markdown") or data.get("text") or ""

        metadata = {"source": url}
        if self.include_metadata:
            page_meta = data.get("metadata")
            if isinstance(page_meta, dict):
                for k, v in page_meta.items():
                    if v is None or v == "":
                        continue
                    metadata[k] = v if isinstance(v, _SIMPLE_TYPES) else str(v)

        return Document(text=markdown, metadata=metadata)

    def load_data(self, urls: Union[str, List[str]]) -> List[Document]:
        """Fetch one or more URLs and return them as LlamaIndex ``Document`` objects.

        Args:
            urls: A single URL string, or a list of URL strings to read.

        Returns:
            A list of ``Document`` objects, one per URL, each containing the cleaned
            Markdown as ``text`` and the page metadata in ``metadata``.
        """
        if isinstance(urls, str):
            urls = [urls]
        if not urls:
            raise ValueError("SkimReader.load_data requires at least one URL.")
        return [self._read_one(url) for url in urls]
