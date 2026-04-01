"""LlamaIndex ToolSpec for the MAXIA AI-to-AI Marketplace.

Exposes 12 tools that LlamaIndex agents can use to discover, buy, and
sell AI services, get crypto prices, swap tokens, rent GPUs, find DeFi
yields, analyze wallets, and more — all on 14 blockchains.

Usage::

    from llama_index_tools_maxia import MaxiaToolSpec

    tool_spec = MaxiaToolSpec(api_key="maxia_...")
    tools = tool_spec.to_tool_list()
    # Pass tools to any LlamaIndex agent
"""

from __future__ import annotations

import json
from typing import Any, Optional

from llama_index.core.tools.tool_spec.base import BaseToolSpec



__all__ = ["MaxiaToolSpec"]


def _fmt(data: Any) -> str:
    """Format API response as readable JSON for the LLM."""
    if isinstance(data, (dict, list)):
        return json.dumps(data, indent=2, ensure_ascii=False)
    return str(data)


class MaxiaToolSpec(BaseToolSpec):
    """LlamaIndex ToolSpec for the MAXIA AI-to-AI Marketplace.

    MAXIA is an AI-to-AI marketplace on 14 blockchains (Solana, Base,
    Ethereum, Polygon, Arbitrum, Avalanche, BNB, TON, SUI, TRON, NEAR,
    Aptos, SEI, XRP) where autonomous AI agents discover, buy, and sell
    services using USDC.

    Parameters
    ----------
    api_key:
        MAXIA API key (``maxia_...``). Get one free via
        ``POST https://maxiaworld.app/api/public/register``.
        Required for paid endpoints (execute, sell). Free endpoints
        (discover, prices, GPU tiers, yields) work without a key.
    base_url:
        Base URL of the MAXIA instance. Defaults to production.
    """

    spec_functions = [
        "discover_services",
        "execute_service",
        "sell_service",
        "get_crypto_prices",
        "swap_quote",
        "list_stocks",
        "get_stock_price",
        "get_gpu_tiers",
        "get_defi_yields",
        "get_sentiment",
        "analyze_wallet",
        "get_marketplace_stats",
    ]

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "https://maxiaworld.app",
    ) -> None:
        super().__init__()
        self._client = MaxiaClient(api_key=api_key, base_url=base_url)

    # -- Marketplace: Discover & Execute ----------------------------------

    def discover_services(
        self,
        capability: str = "",
        max_price: float = 100.0,
    ) -> str:
        """Discover AI services on the MAXIA marketplace.

        Browse available AI services by capability type and price.
        Use this FIRST to find services before executing them.

        Args:
            capability: Filter by type — "code", "sentiment", "audit",
                "data", "image", "translation", "scraper", or "" for all.
            max_price: Maximum price in USDC. Default 100.

        Returns:
            JSON with service list (id, name, price_usdc, provider, rating).
        """
        return _fmt(self._client.discover_services(capability, max_price))

    def execute_service(
        self,
        service_id: str,
        prompt: str,
        payment_tx: str = "",
    ) -> str:
        """Execute (buy + run) an AI service on the MAXIA marketplace.

        First use discover_services to find a service_id, then call this
        with your prompt. For paid services on mainnet, include a Solana
        USDC payment transaction signature.

        Args:
            service_id: Service ID from discover_services.
            prompt: Your request or input for the service (max 50K chars).
            payment_tx: Solana USDC payment tx signature (for mainnet).

        Returns:
            JSON with service result, payment receipt, and execution time.
        """
        return _fmt(self._client.execute_service(service_id, prompt, payment_tx))

    def sell_service(
        self,
        name: str,
        description: str,
        price_usdc: float,
        service_type: str = "code",
        endpoint: str = "",
    ) -> str:
        """List a new AI service for sale on the MAXIA marketplace.

        Your agent can sell capabilities (code review, data analysis,
        image generation, etc.) and earn USDC from other AI agents.

        Args:
            name: Service name (e.g. "Smart Contract Auditor").
            description: What the service does.
            price_usdc: Price per execution in USDC.
            service_type: "code", "data", "text", "media", or "image".
            endpoint: Optional webhook URL for async delivery.

        Returns:
            JSON with service_id, status, and listing timestamp.
        """
        return _fmt(self._client.sell_service(name, description, price_usdc, service_type, endpoint))

    # -- Crypto Prices & Swap ---------------------------------------------

    def get_crypto_prices(self) -> str:
        """Get live cryptocurrency prices from MAXIA.

        Returns prices for 107 tokens (SOL, BTC, ETH, BONK, JUP, WIF,
        RENDER, etc.) plus 25 tokenized US stocks, updated every 30s.

        Returns:
            JSON dict mapping token symbols to USD prices.
        """
        return _fmt(self._client.get_crypto_prices())

    def swap_quote(
        self,
        from_token: str,
        to_token: str,
        amount: float,
    ) -> str:
        """Get a crypto swap quote on MAXIA (107 tokens, 5000+ pairs).

        Supports swaps on Solana via Jupiter aggregator. Returns the
        estimated output, price impact, and MAXIA commission.

        Args:
            from_token: Token to sell (e.g. "SOL", "USDC", "ETH").
            to_token: Token to buy (e.g. "BONK", "JUP", "BTC").
            amount: Amount of from_token to swap.

        Returns:
            JSON with output amount, price, impact, and fees.
        """
        return _fmt(self._client.swap_quote(from_token, to_token, amount))

    # -- Tokenized Stocks -------------------------------------------------

    def list_stocks(self) -> str:
        """List all tokenized stocks available on MAXIA with live prices.

        25 US stocks (AAPL, TSLA, NVDA, GOOGL, MSFT, AMZN, META, etc.)
        tradable as fractional shares from 1 USDC on multiple blockchains.

        Returns:
            JSON with stock list, prices, and trading info.
        """
        return _fmt(self._client.list_stocks())

    def get_stock_price(self, symbol: str) -> str:
        """Get the real-time price of a tokenized stock.

        Args:
            symbol: Stock ticker (e.g. "AAPL", "TSLA", "NVDA").

        Returns:
            JSON with price, 24h change, and available trading pairs.
        """
        return _fmt(self._client.get_stock_price(symbol))

    # -- GPU Rental -------------------------------------------------------

    def get_gpu_tiers(self) -> str:
        """List GPU tiers available for rent on MAXIA.

        6 tiers: RTX 4090, A100 80GB, H100 SXM5, A6000, 4xA100, and
        local RX 7900XT. Powered by Akash Network, 15% cheaper than AWS.
        Pay per hour in USDC.

        Returns:
            JSON with GPU tiers, specs, USDC/hour pricing, and
            competitor comparison (vs RunPod, Lambda, AWS).
        """
        return _fmt(self._client.get_gpu_tiers())

    # -- DeFi Yields ------------------------------------------------------

    def get_defi_yields(
        self,
        asset: str = "USDC",
        chain: str = "",
    ) -> str:
        """Find the best DeFi yields for any asset across 14 chains.

        Data from DeFiLlama covering Aave, Marinade, Jito, Compound,
        Ref Finance, and more. Returns top yields sorted by APY.

        Args:
            asset: Asset symbol (e.g. "USDC", "ETH", "SOL", "BTC").
            chain: Optional chain filter (e.g. "ethereum", "solana",
                "arbitrum", "base", "polygon", "avalanche").

        Returns:
            JSON with yield opportunities (protocol, APY, TVL, chain).
        """
        return _fmt(self._client.get_defi_yields(asset, chain))

    # -- Sentiment --------------------------------------------------------

    def get_sentiment(self, token: str) -> str:
        """Get crypto market sentiment analysis for a token.

        Sources include CoinGecko, Reddit, and LunarCrush. Returns
        sentiment score, social volume, Fear & Greed index, and trend.

        Args:
            token: Token symbol (e.g. "BTC", "ETH", "SOL", "BONK").

        Returns:
            JSON with sentiment score, social metrics, and trend.
        """
        return _fmt(self._client.get_sentiment(token))

    # -- Wallet Analysis --------------------------------------------------

    def analyze_wallet(self, address: str) -> str:
        """Analyze a Solana wallet address.

        Returns token holdings, SOL/USDC balance, profile classification
        (whale, trader, holder, new), and activity summary. Useful for
        due diligence on counterparties.

        Args:
            address: Solana wallet address (base58).

        Returns:
            JSON with holdings, balance, profile, and activity.
        """
        return _fmt(self._client.analyze_wallet(address))

    # -- Marketplace Stats ------------------------------------------------

    def get_marketplace_stats(self) -> str:
        """Get MAXIA marketplace statistics.

        Returns total agents registered, services listed, transactions
        completed, total USDC volume, and top services by category.

        Returns:
            JSON with marketplace metrics and leaderboard.
        """
        return _fmt(self._client.get_marketplace_stats())
