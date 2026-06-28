"""Verifly email verification tool spec."""

import os
from typing import Optional

import requests

from llama_index.core.schema import Document
from llama_index.core.tools.tool_spec.base import BaseToolSpec

DEFAULT_BASE_URL = "https://verifly.email"


class VeriflyToolSpec(BaseToolSpec):
    """
    Verifly email verification tool spec.

    Agent-native email verification. Wraps the Verifly API
    (https://verifly.email) so an agent can check whether an email address is
    real and safe to send to before adding it to a list or firing off a
    message. Verifly checks deliverability (syntax, domain, MX, SMTP) and risk
    flags (disposable, role, catch-all, free provider), and returns a verdict
    with a send / do-not-send recommendation.
    """

    spec_functions = ["verify_email"]

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 30.0,
    ) -> None:
        """
        Initialize with parameters.

        Args:
            api_key: Verifly API key (``vf_...``). Falls back to the
                ``VERIFLY_API_KEY`` environment variable. An agent can obtain a
                key with no human in the loop via Verifly's autonomous
                registration endpoint.
            base_url: Verifly API base URL. Defaults to
                ``https://verifly.email``.
            timeout: Per-request timeout in seconds.

        """
        api_key = api_key or os.environ.get("VERIFLY_API_KEY")
        if not api_key:
            raise ValueError(
                "A Verifly API key is required. Pass api_key=... or set the "
                "VERIFLY_API_KEY environment variable."
            )
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def verify_email(self, email: str) -> Document:
        """
        Verify a single email address with Verifly.

        Checks deliverability (syntax, domain, MX, SMTP) and risk flags
        (disposable, role, catch-all, free provider), and returns a verdict
        plus a send / do-not-send recommendation. Use this before sending to a
        new address or adding it to a mailing list.

        Args:
            email: The email address to verify, e.g. ``lead@example.com``.

        Returns:
            A Document whose ``text`` is a short human-readable verdict and
            whose ``metadata`` holds the full structured Verifly result
            (``is_valid``, ``result``, ``reason``, ``details``,
            ``recommendation``, ``credits``).

        """
        response = requests.get(
            f"{self.base_url}/api/v1/verify",
            params={"email": email},
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()

        result = data.get("result", "unknown")
        reason = data.get("reason")
        recommendation = data.get("recommendation")
        summary = f"{email}: {result}"
        if reason:
            summary += f" ({reason})"
        if recommendation:
            summary += f" - recommendation: {recommendation}"

        return Document(text=summary, metadata=data)
