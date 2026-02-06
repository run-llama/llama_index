"""Nory x402 Payment Tool Spec for LlamaIndex.

Tools for AI agents to make payments using the x402 HTTP protocol.
Supports Solana and 7 EVM chains with sub-400ms settlement.
"""

import requests
from typing import Optional, List
from llama_index.core.schema import Document
from llama_index.core.tools.tool_spec.base import BaseToolSpec

NORY_API_BASE = "https://noryx402.com"


class NoryX402ToolSpec(BaseToolSpec):
    """
    Nory x402 Payment tool spec.

    Enables AI agents to make payments using the x402 HTTP payment protocol.
    Supports Solana and 7 EVM chains (Base, Polygon, Arbitrum, Optimism,
    Avalanche, Sei, IoTeX) with sub-400ms settlement times.

    Use Cases:
    - Pay for premium API access on-the-fly
    - Handle HTTP 402 Payment Required responses automatically
    - Make micropayments for AI-to-AI services
    - Access paid resources without pre-configured subscriptions

    Args:
        api_key (str, optional): Nory API key for authenticated endpoints.
    """

    spec_functions = [
        "get_payment_requirements",
        "verify_payment",
        "settle_payment",
        "lookup_transaction",
        "health_check",
    ]

    def __init__(self, api_key: Optional[str] = None) -> None:
        """Initialize with optional API key."""
        self.api_key = api_key

    def _get_headers(self, with_json: bool = False) -> dict:
        """Get request headers with optional auth."""
        headers = {}
        if with_json:
            headers["Content-Type"] = "application/json"
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def get_payment_requirements(
        self,
        resource: str,
        amount: str,
        network: Optional[str] = None,
    ) -> List[Document]:
        """
        Get x402 payment requirements for accessing a paid resource.

        Use this when you encounter an HTTP 402 Payment Required response
        and need to know how much to pay and where to send payment.

        Args:
            resource (str): The resource path requiring payment (e.g., /api/premium/data)
            amount (str): Amount in human-readable format (e.g., '0.10' for $0.10 USDC)
            network (str, optional): Preferred blockchain network. Options:
                - solana-mainnet, solana-devnet
                - base-mainnet, polygon-mainnet, arbitrum-mainnet
                - optimism-mainnet, avalanche-mainnet, sei-mainnet, iotex-mainnet

        Returns:
            List[Document]: Payment requirements including amount, supported networks, and wallet address.
        """
        params = {"resource": resource, "amount": amount}
        if network:
            params["network"] = network

        response = requests.get(
            f"{NORY_API_BASE}/api/x402/requirements",
            params=params,
            headers=self._get_headers(),
        )
        response.raise_for_status()
        return [Document(text=response.text)]

    def verify_payment(self, payload: str) -> List[Document]:
        """
        Verify a signed payment transaction before settlement.

        Use this to validate that a payment transaction is correct
        before submitting it to the blockchain.

        Args:
            payload (str): Base64-encoded payment payload containing signed transaction.

        Returns:
            List[Document]: Verification result including validity and payer info.
        """
        response = requests.post(
            f"{NORY_API_BASE}/api/x402/verify",
            json={"payload": payload},
            headers=self._get_headers(with_json=True),
        )
        response.raise_for_status()
        return [Document(text=response.text)]

    def settle_payment(self, payload: str) -> List[Document]:
        """
        Settle a payment on-chain.

        Use this to submit a verified payment transaction to the blockchain.
        Settlement typically completes in under 400ms.

        Args:
            payload (str): Base64-encoded payment payload.

        Returns:
            List[Document]: Settlement result including transaction ID.
        """
        response = requests.post(
            f"{NORY_API_BASE}/api/x402/settle",
            json={"payload": payload},
            headers=self._get_headers(with_json=True),
        )
        response.raise_for_status()
        return [Document(text=response.text)]

    def lookup_transaction(self, transaction_id: str, network: str) -> List[Document]:
        """
        Look up transaction status.

        Use this to check the status of a previously submitted payment.

        Args:
            transaction_id (str): Transaction ID or signature.
            network (str): Network where the transaction was submitted.

        Returns:
            List[Document]: Transaction details including status and confirmations.
        """
        response = requests.get(
            f"{NORY_API_BASE}/api/x402/transactions/{transaction_id}",
            params={"network": network},
            headers=self._get_headers(),
        )
        response.raise_for_status()
        return [Document(text=response.text)]

    def health_check(self) -> List[Document]:
        """
        Check Nory service health.

        Use this to verify the payment service is operational
        and see supported networks.

        Returns:
            List[Document]: Health status and supported networks.
        """
        response = requests.get(f"{NORY_API_BASE}/api/x402/health")
        response.raise_for_status()
        return [Document(text=response.text)]
