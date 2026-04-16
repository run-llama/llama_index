"""Unit tests for the InsumerToolSpec — mocked HTTP, verifies request/response
shape plumbing without hitting the live API.

Integration tests that hit api.insumermodel.com live are in
``tests/test_integration.py`` and are skipped unless ``INSUMER_API_KEY`` is set
in the environment.
"""

from unittest.mock import MagicMock, patch

import pytest

from llama_index.tools.insumer import InsumerToolSpec


API_KEY = "insr_live_0000000000000000000000000000000000000000"


@pytest.fixture
def spec() -> InsumerToolSpec:
    return InsumerToolSpec(api_key=API_KEY)


def _mock_response(payload: dict, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = payload
    resp.raise_for_status.return_value = None
    return resp


def test_spec_functions_exported(spec: InsumerToolSpec) -> None:
    assert spec.spec_functions == [
        "attest_wallet",
        "get_trust_profile",
        "list_compliance_templates",
        "get_jwks",
    ]


def test_no_key_raises_for_authed_call() -> None:
    spec = InsumerToolSpec()
    with pytest.raises(ValueError, match="InsumerAPI key required"):
        spec.attest_wallet(conditions=[], wallet="0x" + "a" * 40)


@patch("llama_index.tools.insumer.base.requests.post")
def test_attest_wallet_evm_token_balance(mock_post: MagicMock, spec: InsumerToolSpec) -> None:
    mock_post.return_value = _mock_response({
        "ok": True,
        "data": {
            "attestation": {
                "id": "ATST-ABCDEF0123456789",
                "pass": True,
                "results": [{"met": True, "conditionHash": "0xabc"}],
                "passCount": 1,
                "failCount": 0,
                "attestedAt": "2026-04-16T00:00:00.000Z",
                "expiresAt": "2026-04-16T00:30:00.000Z",
            },
            "sig": "aGVsbG8=",
            "kid": "insumer-attest-v1",
        },
        "meta": {"creditsRemaining": 99, "creditsCharged": 1},
    })

    wallet = "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"
    result = spec.attest_wallet(
        wallet=wallet,
        conditions=[{
            "type": "token_balance",
            "contractAddress": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
            "chainId": 8453,
            "threshold": 100,
            "decimals": 6,
            "label": "USDC on Base >= 100",
        }],
    )

    mock_post.assert_called_once()
    call_args = mock_post.call_args
    assert call_args.args[0] == "https://api.insumermodel.com/v1/attest"
    body = call_args.kwargs["json"]
    assert body["wallet"] == wallet
    assert body["conditions"][0]["type"] == "token_balance"
    assert body["conditions"][0]["chainId"] == 8453
    assert body["conditions"][0]["threshold"] == 100
    assert "solanaWallet" not in body
    assert "xrplWallet" not in body
    headers = call_args.kwargs["headers"]
    assert headers["X-API-Key"] == API_KEY
    assert headers["Content-Type"] == "application/json"

    assert result["ok"] is True
    assert result["data"]["attestation"]["pass"] is True
    assert result["data"]["kid"] == "insumer-attest-v1"
    assert result["meta"]["creditsCharged"] == 1


@patch("llama_index.tools.insumer.base.requests.post")
def test_attest_wallet_jwt_format(mock_post: MagicMock, spec: InsumerToolSpec) -> None:
    mock_post.return_value = _mock_response({
        "ok": True,
        "data": {
            "attestation": {"id": "ATST-1", "pass": True, "results": [],
                            "passCount": 0, "failCount": 0,
                            "attestedAt": "2026-04-16T00:00:00.000Z",
                            "expiresAt": "2026-04-16T00:30:00.000Z"},
            "sig": "aGVsbG8=",
            "kid": "insumer-attest-v1",
            "jwt": "eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6Imluc3VtZXItYXR0ZXN0LXYxIn0.eyJwYXNzIjp0cnVlfQ.sig",
        },
        "meta": {},
    })
    result = spec.attest_wallet(
        wallet="0x" + "a" * 40,
        conditions=[{"type": "farcaster_id"}],
        format="jwt",
    )
    body = mock_post.call_args.kwargs["json"]
    assert body["format"] == "jwt"
    assert result["data"]["jwt"].startswith("eyJ")


@patch("llama_index.tools.insumer.base.requests.post")
def test_attest_wallet_xrpl(mock_post: MagicMock, spec: InsumerToolSpec) -> None:
    mock_post.return_value = _mock_response({"ok": True, "data": {}, "meta": {}})
    spec.attest_wallet(
        xrpl_wallet="rN7n3473SaZBCG4dFL83w7p1W9cgPJqKro",
        conditions=[{
            "type": "token_balance",
            "contractAddress": "native",
            "chainId": "xrpl",
            "threshold": 100,
            "label": "XRP >= 100",
        }],
    )
    body = mock_post.call_args.kwargs["json"]
    assert body["xrplWallet"] == "rN7n3473SaZBCG4dFL83w7p1W9cgPJqKro"
    assert body["conditions"][0]["chainId"] == "xrpl"
    assert "wallet" not in body


@patch("llama_index.tools.insumer.base.requests.post")
def test_attest_wallet_merkle_proof(mock_post: MagicMock, spec: InsumerToolSpec) -> None:
    mock_post.return_value = _mock_response({"ok": True, "data": {}, "meta": {}})
    spec.attest_wallet(
        wallet="0x" + "a" * 40,
        conditions=[{"type": "token_balance", "contractAddress": "0x" + "b" * 40,
                     "chainId": 1, "threshold": 1, "decimals": 18}],
        proof="merkle",
    )
    body = mock_post.call_args.kwargs["json"]
    assert body["proof"] == "merkle"


@patch("llama_index.tools.insumer.base.requests.post")
def test_get_trust_profile(mock_post: MagicMock, spec: InsumerToolSpec) -> None:
    mock_post.return_value = _mock_response({
        "ok": True,
        "data": {
            "trust": {
                "id": "TRST-A1B2C",
                "wallet": "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045",
                "conditionSetVersion": "v1",
                "dimensions": {
                    "stablecoins": {"checks": [], "passCount": 0, "failCount": 0, "total": 0},
                    "governance": {"checks": [], "passCount": 0, "failCount": 0, "total": 0},
                    "nfts": {"checks": [], "passCount": 0, "failCount": 0, "total": 0},
                    "staking": {"checks": [], "passCount": 0, "failCount": 0, "total": 0},
                },
                "summary": {
                    "totalChecks": 0, "totalPassed": 0, "totalFailed": 0,
                    "dimensionsWithActivity": 0, "dimensionsChecked": 4,
                },
                "profiledAt": "2026-04-16T00:00:00.000Z",
                "expiresAt": "2026-04-16T00:30:00.000Z",
            },
            "sig": "aGVsbG8=",
            "kid": "insumer-attest-v1",
        },
        "meta": {"creditsRemaining": 97, "creditsCharged": 3},
    })

    wallet = "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"
    result = spec.get_trust_profile(wallet=wallet)

    mock_post.assert_called_once()
    assert mock_post.call_args.args[0] == "https://api.insumermodel.com/v1/trust"
    body = mock_post.call_args.kwargs["json"]
    assert body["wallet"] == wallet
    assert "solanaWallet" not in body

    assert result["data"]["trust"]["conditionSetVersion"] == "v1"
    assert set(result["data"]["trust"]["dimensions"].keys()) == {
        "stablecoins", "governance", "nfts", "staking",
    }
    assert result["data"]["kid"] == "insumer-attest-v1"
    assert result["meta"]["creditsCharged"] == 3


@patch("llama_index.tools.insumer.base.requests.post")
def test_get_trust_profile_multichain(mock_post: MagicMock, spec: InsumerToolSpec) -> None:
    mock_post.return_value = _mock_response({"ok": True, "data": {}, "meta": {}})
    spec.get_trust_profile(
        wallet="0x" + "a" * 40,
        solana_wallet="5Hdh2n3473SaZBCG4dFL83w7p1W9cgPJqKroabc",
        xrpl_wallet="rN7n3473SaZBCG4dFL83w7p1W9cgPJqKro",
        bitcoin_wallet="bc1qar0srrr7xfkvy5l643lydnw9re59gtzzwf5mdq",
    )
    body = mock_post.call_args.kwargs["json"]
    assert body["solanaWallet"].startswith("5Hdh")
    assert body["xrplWallet"].startswith("r")
    assert body["bitcoinWallet"].startswith("bc1")


@patch("llama_index.tools.insumer.base.requests.get")
def test_list_compliance_templates_no_auth(mock_get: MagicMock) -> None:
    mock_get.return_value = _mock_response({
        "ok": True,
        "data": {
            "templates": {
                "coinbase_verified_account": {
                    "provider": "Coinbase",
                    "description": "Coinbase Verified Account",
                    "chainId": 8453,
                    "chainName": "Base",
                },
                "gitcoin_passport_score": {
                    "provider": "Gitcoin",
                    "description": "Gitcoin Passport Score (>=20)",
                    "chainId": 10,
                    "chainName": "Optimism",
                },
            },
        },
        "meta": {},
    })

    # Template discovery works without an API key.
    spec = InsumerToolSpec()
    result = spec.list_compliance_templates()

    mock_get.assert_called_once()
    call_args = mock_get.call_args
    assert call_args.args[0] == "https://api.insumermodel.com/v1/compliance/templates"
    headers = call_args.kwargs["headers"]
    assert "X-API-Key" not in headers

    assert "coinbase_verified_account" in result["data"]["templates"]
    assert result["data"]["templates"]["coinbase_verified_account"]["chainId"] == 8453


@patch("llama_index.tools.insumer.base.requests.get")
def test_get_jwks_no_auth(mock_get: MagicMock) -> None:
    mock_get.return_value = _mock_response({
        "keys": [{
            "kty": "EC",
            "crv": "P-256",
            "x": "JtHPhDPnv8AfP0JSlGutxbOlxreV2Chey27Z76q3V2c",
            "y": "kn34HaxVSJfn8NxwNEBjjLkcrM_GDw1lgnqyADGuc4c",
            "use": "sig",
            "alg": "ES256",
            "kid": "insumer-attest-v1",
        }],
    })

    spec = InsumerToolSpec()  # no key — JWKS is public
    result = spec.get_jwks()

    mock_get.assert_called_once()
    assert mock_get.call_args.args[0].endswith("/.well-known/jwks.json")

    assert result["keys"][0]["kty"] == "EC"
    assert result["keys"][0]["crv"] == "P-256"
    assert result["keys"][0]["alg"] == "ES256"
    assert result["keys"][0]["kid"] == "insumer-attest-v1"


def test_custom_base_url() -> None:
    spec = InsumerToolSpec(api_key=API_KEY, base_url="https://staging.insumermodel.com/")
    # Trailing slash stripped at construction.
    assert spec.base_url == "https://staging.insumermodel.com"


def test_to_tool_list_integration(spec: InsumerToolSpec) -> None:
    """Verifies BaseToolSpec --> FunctionTool conversion works."""
    tools = spec.to_tool_list()
    assert len(tools) == 4
    tool_names = {t.metadata.name for t in tools}
    assert tool_names == {
        "attest_wallet",
        "get_trust_profile",
        "list_compliance_templates",
        "get_jwks",
    }
