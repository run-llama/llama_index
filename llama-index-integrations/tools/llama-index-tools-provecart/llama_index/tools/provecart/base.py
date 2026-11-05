"""ProveCart ToolSpec for LlamaIndex.

Wraps the ProveCart Verification API (https://api.provecart.com):

- ``verify_cart`` — the pre-payment conformity check. Compares a
  cart an agent built against a signed customer mandate and
  returns the signed verdict (MATCH / REVIEW / BLOCK) plus, on
  REVIEW or BLOCK, machine-readable corrections. Requires an
  integrator API key.
- ``certification_info`` — reads the public ProveCart Verified
  Agent record (current certifications, scores, result links).
  No key required.

Agent instructions: https://provecart.com/agents.md
"""
import json
import urllib.error
import urllib.request

from llama_index.core.tools.tool_spec.base import BaseToolSpec

BASE = "https://api.provecart.com"


def _request(method, path, api_key=None, body=None):
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    req = urllib.request.Request(BASE + path, data=data, headers=headers,
                                 method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        try:
            payload = json.loads(e.read() or b"{}")
        except Exception:
            payload = {}
        raise RuntimeError(f"ProveCart {method} {path} -> {e.code}: "
                           f"{payload.get('detail', payload)}")


class ProveCartToolSpec(BaseToolSpec):
    """Tools for checking agent-built carts before payment."""

    spec_functions = ["verify_cart", "certification_info"]

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize with an integrator API key (verify_cart only)."""
        self.api_key = api_key

    def verify_cart(self, mandate_id: str, merchant: dict,
                    items: list, total_price: float,
                    currency: str = "USD",
                    previous_record_id: str | None = None) -> dict:
        """Check a built cart against a signed customer mandate.

        Capture payment only when the returned verdict is MATCH.
        On BLOCK, the response's corrections name exactly what to
        fix; apply only fixes inside the customer's existing
        authorization and resubmit with previous_record_id. On
        REVIEW, stop and ask the buyer.

        Args:
            mandate_id: Mandate id from POST /mandates.
            merchant: {"name": ..., "domain": ...} of the cart.
            items: [{name, quantity, unit_price, ...}].
            total_price: Grand total of the cart.
            currency: ISO currency code (default USD).
            previous_record_id: Set on a corrected resubmission.
        """
        if not self.api_key:
            raise ValueError("verify_cart needs an integrator API key")
        body = {"mandate_id": mandate_id, "merchant": merchant,
                "items": items, "total_price": total_price,
                "currency": currency}
        if previous_record_id:
            body["previous_record_id"] = previous_record_id
        return _request("POST", "/verify", self.api_key, body)

    def certification_info(self) -> dict:
        """Read the public Verified Agent certification record.

        Returns the agents currently certified on ProveCart's exam
        (status, score, cohort, expiry, permanent result URL).
        Certification attests measured buying behavior; it is not
        a gate. No API key needed.
        """
        return _request("GET", "/agents/certifications")
