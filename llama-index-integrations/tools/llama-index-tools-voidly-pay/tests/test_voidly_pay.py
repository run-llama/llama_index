from unittest.mock import MagicMock, patch

import pytest

from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.tools.voidly_pay import VoidlyPayToolSpec


def test_class():
    names_of_base_classes = [b.__name__ for b in VoidlyPayToolSpec.__mro__]
    assert BaseToolSpec.__name__ in names_of_base_classes


def test_spec_functions():
    """Test that spec_functions exposes all public methods."""
    expected = {
        "pay_for_url",
        "discover_paid_endpoints",
        "marketplace_browse",
        "health_check",
        "list_listing",
    }
    assert expected.issubset(set(VoidlyPayToolSpec.spec_functions))


@patch("voidly_pay.VoidlyPay")
def test_init_default(mock_voidly_pay):
    """Test init uses default facilitator URL."""
    VoidlyPayToolSpec()
    mock_voidly_pay.assert_called_once_with(
        keypair_path=None,
        facilitator_url="https://api.voidly.ai/v1/pay/x402",
    )


@patch("voidly_pay.VoidlyPay")
def test_init_custom_keypair(mock_voidly_pay):
    """Test init accepts a custom keypair path."""
    VoidlyPayToolSpec(keypair_path="/tmp/my-key.json")
    mock_voidly_pay.assert_called_once_with(
        keypair_path="/tmp/my-key.json",
        facilitator_url="https://api.voidly.ai/v1/pay/x402",
    )


@patch("voidly_pay.VoidlyPay")
def test_pay_for_url_get(mock_voidly_pay):
    """Test pay_for_url calls fetch with GET defaults."""
    mock_client = MagicMock()
    mock_client.fetch.return_value = {
        "status_code": 200,
        "body": {"summary": "Alan Turing was a mathematician."},
        "payment_receipt": {"transfer_id": "abc123", "amount_usdc": 0.01},
    }
    mock_voidly_pay.return_value = mock_client

    tool = VoidlyPayToolSpec()
    result = tool.pay_for_url("https://api.voidly.ai/v1/pay/wikipedia/Alan_Turing")

    mock_client.fetch.assert_called_once_with(
        "https://api.voidly.ai/v1/pay/wikipedia/Alan_Turing",
        method="GET",
        params=None,
        json=None,
    )
    assert result["status_code"] == 200
    assert "payment_receipt" in result


@patch("voidly_pay.VoidlyPay")
def test_pay_for_url_post(mock_voidly_pay):
    """Test pay_for_url with POST + body."""
    mock_client = MagicMock()
    mock_client.fetch.return_value = {"status_code": 200, "body": {}}
    mock_voidly_pay.return_value = mock_client

    tool = VoidlyPayToolSpec()
    tool.pay_for_url(
        "https://api.example.com/scrape",
        method="POST",
        body={"url": "https://example.com"},
    )

    mock_client.fetch.assert_called_once_with(
        "https://api.example.com/scrape",
        method="POST",
        params=None,
        json={"url": "https://example.com"},
    )


@patch("voidly_pay.VoidlyPay")
def test_discover_paid_endpoints(mock_voidly_pay):
    """Test discover delegates to client."""
    mock_client = MagicMock()
    mock_client.discover.return_value = [
        {
            "url": "https://api.voidly.ai/v1/pay/wikipedia/{title}",
            "price_usdc": 0.01,
            "recipient_did": "did:voidly:abc",
            "description": "Wikipedia summary",
        }
    ]
    mock_voidly_pay.return_value = mock_client

    tool = VoidlyPayToolSpec()
    result = tool.discover_paid_endpoints()

    mock_client.discover.assert_called_once()
    assert len(result) == 1
    assert result[0]["price_usdc"] == 0.01


@patch("voidly_pay.VoidlyPay")
def test_marketplace_browse(mock_voidly_pay):
    """Test marketplace browse with category filter."""
    mock_client = MagicMock()
    mock_marketplace = MagicMock()
    mock_marketplace.browse.return_value = [
        {"name": "PDF→text", "pricing": {"amount_usdc": 0.02}}
    ]
    mock_client.marketplace = mock_marketplace
    mock_voidly_pay.return_value = mock_client

    tool = VoidlyPayToolSpec()
    result = tool.marketplace_browse(category="data")

    mock_marketplace.browse.assert_called_once_with(category="data")
    assert result[0]["name"] == "PDF→text"


@patch("voidly_pay.VoidlyPay")
def test_health_check(mock_voidly_pay):
    """Test health check returns 6-check report."""
    mock_client = MagicMock()
    mock_client.health_check.return_value = {
        "facilitator_reachable": True,
        "vault_verified": True,
        "wallet_balance_credits": 10,
        "keypair_valid": True,
        "recent_settlements_ok": True,
        "all_checks_passed": True,
    }
    mock_voidly_pay.return_value = mock_client

    tool = VoidlyPayToolSpec()
    report = tool.health_check()

    assert report["all_checks_passed"] is True
    assert report["wallet_balance_credits"] == 10


@patch("voidly_pay.VoidlyPay")
def test_list_listing(mock_voidly_pay):
    """Test creating a marketplace listing."""
    mock_client = MagicMock()
    mock_marketplace = MagicMock()
    mock_marketplace.list.return_value = {
        "id": "lst_xyz",
        "discoverability_url": "https://voidly.ai/pay/listings/lst_xyz",
    }
    mock_client.marketplace = mock_marketplace
    mock_voidly_pay.return_value = mock_client

    tool = VoidlyPayToolSpec()
    result = tool.list_listing(
        name="My API",
        endpoint_url="https://api.example.com/paid",
        price_usdc=0.05,
        description="Does X",
        category="data",
    )

    mock_marketplace.list.assert_called_once_with(
        name="My API",
        endpoint_url="https://api.example.com/paid",
        price_usdc=0.05,
        description="Does X",
        category="data",
    )
    assert result["id"] == "lst_xyz"
