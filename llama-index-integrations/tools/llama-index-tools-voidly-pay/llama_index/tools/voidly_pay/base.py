"""Voidly Pay tool spec — pay for HTTP 402 endpoints from a LlamaIndex agent."""

from typing import Any, Dict, List, Optional

from llama_index.core.tools.tool_spec.base import BaseToolSpec


class VoidlyPayToolSpec(BaseToolSpec):
    """
    Voidly Pay tool spec.

    Lets a LlamaIndex agent autonomously pay for HTTP 402 endpoints via the
    x402 protocol. Settlement happens off-chain in Voidly Pay credits
    (Stage 1) or on-chain USDC on Base mainnet (Stage 2). The vault is
    Sourcify-verified at 0xb592512932a7b354969bb48039c2dc7ad6ad1c12 with
    public reserves at https://voidly.ai/pay/proof.

    Wraps the official `voidly-pay` PyPI SDK. Identity is an Ed25519 keypair
    on disk — no API keys, no Stripe customer object.
    """

    spec_functions = [
        "pay_for_url",
        "discover_paid_endpoints",
        "marketplace_browse",
        "health_check",
        "list_listing",
    ]

    def __init__(
        self,
        keypair_path: Optional[str] = None,
        facilitator_url: str = "https://api.voidly.ai/v1/pay/x402",
    ) -> None:
        """
        Initialize the Voidly Pay tool spec.

        Args:
            keypair_path: Path to the Ed25519 keypair JSON. If None, uses the
                default `~/.voidly-pay-keypair.json` (provisioned via
                https://voidly.ai/pay/claim with a 10-credit faucet grant).
            facilitator_url: Voidly Pay x402 facilitator endpoint.

        """
        from voidly_pay import VoidlyPay

        self.client = VoidlyPay(
            keypair_path=keypair_path,
            facilitator_url=facilitator_url,
        )

    def pay_for_url(
        self,
        url: str,
        method: str = "GET",
        params: Optional[Dict[str, Any]] = None,
        body: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Call an x402-paywalled URL, auto-paying any 402 challenge.

        Args:
            url: The endpoint to call.
            method: HTTP method (default: GET).
            params: Query parameters.
            body: JSON body for POST/PUT.

        Returns:
            A dict with `status_code`, `body`, and (if a 402 was settled)
            `payment_receipt` containing the signed transfer envelope and
            settlement timestamp.

        """
        return self.client.fetch(url, method=method, params=params, json=body)

    def discover_paid_endpoints(self) -> List[Dict[str, Any]]:
        """
        List all paid endpoints currently advertised on the facilitator.

        Returns:
            A list of endpoint descriptors with `url`, `price_usdc`,
            `recipient_did`, and `description`.

        """
        return self.client.discover()

    def marketplace_browse(
        self, category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Browse the open Voidly Pay marketplace.

        Args:
            category: Optional category filter (e.g. "data", "scraping",
                "compute").

        Returns:
            A list of marketplace listings — Voidly's own 17 paid endpoints
            plus self-served third-party listings. Each item has `name`,
            `pricing`, `endpoint_url`, and `description`.

        """
        return self.client.marketplace.browse(category=category)

    def health_check(self) -> Dict[str, Any]:
        """
        Run the Voidly Pay 6-check trust report.

        Returns:
            A dict with checks for facilitator reachability, vault
            verification, wallet balance, keypair validity, and recent
            settlement health. Useful before a long-running paid task.

        """
        return self.client.health_check()

    def list_listing(
        self,
        name: str,
        endpoint_url: str,
        price_usdc: float,
        description: str,
        category: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        List your own paid endpoint on the Voidly Pay marketplace.

        Args:
            name: Human-readable name.
            endpoint_url: Your URL that returns 402 with a Voidly Pay quote.
            price_usdc: Price per call in USDC.
            description: What the endpoint does.
            category: Optional category.

        Returns:
            The created listing with its assigned id and discoverability URL.

        """
        return self.client.marketplace.list(
            name=name,
            endpoint_url=endpoint_url,
            price_usdc=price_usdc,
            description=description,
            category=category,
        )
