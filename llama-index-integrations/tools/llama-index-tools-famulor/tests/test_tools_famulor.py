"""Unit tests for the Famulor tool spec. HTTP is mocked; nothing is dialed."""

from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest
import requests

from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.tools.famulor import FamulorAPIError, FamulorToolSpec
from llama_index.tools.famulor.client import DEFAULT_BASE_URL

ASSISTANT = "a1b2c3d4-0000-4000-8000-000000000001"
CALL = "c1b2c3d4-0000-4000-8000-000000000010"
PHONE = "b1b2c3d4-0000-4000-8000-000000000020"
CAMPAIGN = "d1b2c3d4-0000-4000-8000-000000000030"
LEAD = "e1b2c3d4-0000-4000-8000-000000000040"
API_KEY = "fam_test_key"


def _response(
    status: int = 200,
    payload: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    text: str = "",
    json_error: bool = False,
) -> MagicMock:
    response = MagicMock()
    response.status_code = status
    response.headers = headers or {}
    response.text = text
    response.reason = "Error"
    if json_error:
        response.json.side_effect = ValueError("not json")
    else:
        response.json.return_value = {"data": []} if payload is None else payload
    return response


@pytest.fixture
def spec() -> FamulorToolSpec:
    return FamulorToolSpec(api_key=API_KEY)


def _request_of(mock_request: MagicMock) -> Dict[str, Any]:
    assert mock_request.call_count == 1
    args, kwargs = mock_request.call_args
    return {"method": args[0], "url": args[1], **kwargs}


def test_class_is_tool_spec() -> None:
    assert BaseToolSpec in FamulorToolSpec.__mro__
    assert FamulorToolSpec.spec_functions == [
        "list_assistants",
        "get_assistant",
        "list_phone_numbers",
        "list_calls",
        "get_call",
        "list_leads",
        "create_lead",
        "get_lead",
        "list_campaigns",
        "get_campaign",
        "create_campaign",
        "make_call",
    ]


def test_make_call_docstring_describes_a_real_call() -> None:
    tool = FamulorToolSpec(api_key=API_KEY)
    tools = {item.metadata.name: item for item in tool.to_tool_list()}
    description = tools["make_call"].metadata.description
    assert "POST /calls" in description
    assert "real outbound phone call" in description
    assert "not a draft" in description


def test_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAMULOR_API_KEY", "fam_from_env")
    tool = FamulorToolSpec()
    assert tool.api_key == "fam_from_env"


def test_explicit_api_key_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAMULOR_API_KEY", "fam_from_env")
    tool = FamulorToolSpec(api_key="fam_explicit")
    assert tool.api_key == "fam_explicit"


def test_missing_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FAMULOR_API_KEY", raising=False)
    with pytest.raises(ValueError, match="FAMULOR_API_KEY"):
        FamulorToolSpec()


def test_api_key_rejects_internal_whitespace() -> None:
    with pytest.raises(ValueError, match="whitespace"):
        FamulorToolSpec(api_key="fam_bad key")


@pytest.mark.parametrize(
    "base_url",
    [
        "http://app.famulor.io/api/v1",
        "https://app.famulor.de/api/v1",
        "https://legacy.famulor.de/api/v1",
        "https://app.famulor.io/api/user",
        "https://app.famulor.io/api/user/campaigns",
        "https://app.famulor.io/api/v1/assistants",
        "https://user:secret@app.famulor.io/api/v1",
        "https://app.famulor.io/api/v1?token=fam_secret",
    ],
)
def test_rejects_unsupported_base_urls(base_url: str) -> None:
    with pytest.raises(ValueError):
        FamulorToolSpec(api_key=API_KEY, base_url=base_url)


def test_normalizes_whitelabel_base_url() -> None:
    tool = FamulorToolSpec(
        api_key=API_KEY,
        base_url="https://Voice.Example.com/api/v1/",
    )
    assert tool.base_url == "https://voice.example.com/api/v1"


def test_rejects_bad_timeout() -> None:
    with pytest.raises(ValueError, match="timeout"):
        FamulorToolSpec(api_key=API_KEY, timeout=0)
    with pytest.raises(ValueError, match="timeout"):
        FamulorToolSpec(api_key=API_KEY, timeout=True)  # type: ignore[arg-type]


def test_rejects_blank_base_url() -> None:
    with pytest.raises(ValueError, match="base_url"):
        FamulorToolSpec(api_key=API_KEY, base_url="  ")


@patch("llama_index.tools.famulor.client.requests.request")
def test_list_assistants(mock_request: MagicMock, spec: FamulorToolSpec) -> None:
    mock_request.return_value = _response(
        payload={"data": [{"id": ASSISTANT, "name": "Support"}], "meta": {}}
    )
    result = spec.list_assistants(limit=10, offset=5)
    sent = _request_of(mock_request)
    assert sent["method"] == "GET"
    assert sent["url"] == f"{DEFAULT_BASE_URL}/assistants"
    assert sent["headers"] == {
        "Authorization": f"Bearer {API_KEY}",
        "Accept": "application/json",
    }
    assert "json" not in sent
    assert sent["params"] == {"limit": 10, "offset": 5}
    assert sent["timeout"] == 30.0
    assert sent["allow_redirects"] is False
    assert result["data"][0]["id"] == ASSISTANT


@patch("llama_index.tools.famulor.client.requests.request")
def test_list_assistants_omits_empty_query(
    mock_request: MagicMock, spec: FamulorToolSpec
) -> None:
    mock_request.return_value = _response()
    spec.list_assistants()
    assert "params" not in _request_of(mock_request)


@patch("llama_index.tools.famulor.client.requests.request")
def test_get_assistant(mock_request: MagicMock, spec: FamulorToolSpec) -> None:
    mock_request.return_value = _response(payload={"data": {"id": ASSISTANT}})
    result = spec.get_assistant(ASSISTANT)
    sent = _request_of(mock_request)
    assert sent["method"] == "GET"
    assert sent["url"] == f"{DEFAULT_BASE_URL}/assistants/{ASSISTANT}"
    assert result["data"]["id"] == ASSISTANT


@patch("llama_index.tools.famulor.client.requests.request")
def test_list_phone_numbers(mock_request: MagicMock, spec: FamulorToolSpec) -> None:
    mock_request.return_value = _response(payload={"data": [{"id": PHONE}]})
    spec.list_phone_numbers(
        assistant_id=ASSISTANT,
        status="active",
        country="de",
        limit=20,
        offset=0,
    )
    sent = _request_of(mock_request)
    assert sent["url"] == f"{DEFAULT_BASE_URL}/phone-numbers"
    assert sent["params"] == {
        "assistant_id": ASSISTANT,
        "status": "active",
        "country": "DE",
        "limit": 20,
        "offset": 0,
    }


@patch("llama_index.tools.famulor.client.requests.request")
def test_list_calls_maps_documented_filters(
    mock_request: MagicMock, spec: FamulorToolSpec
) -> None:
    mock_request.return_value = _response()
    spec.list_calls(
        assistant_id=ASSISTANT,
        campaign_id=CAMPAIGN,
        status="completed",
        direction="outbound",
        sort="updated_at",
        created_from="2026-07-01T00:00:00Z",
        created_to="2026-07-02T00:00:00Z",
        sentiment="positive",
        success=False,
        q="invoice",
        limit=5,
        offset=1,
    )
    assert _request_of(mock_request)["params"] == {
        "assistant_id": ASSISTANT,
        "campaign_id": CAMPAIGN,
        "status": "completed",
        "direction": "outbound",
        "sort": "updated_at",
        "from": "2026-07-01T00:00:00Z",
        "to": "2026-07-02T00:00:00Z",
        "sentiment": "positive",
        "success": "false",
        "q": "invoice",
        "limit": 5,
        "offset": 1,
    }


@patch("llama_index.tools.famulor.client.requests.request")
def test_get_call_returns_transcript_and_summary(
    mock_request: MagicMock, spec: FamulorToolSpec
) -> None:
    payload = {
        "data": {
            "id": CALL,
            "summary": "Caller asked about billing.",
            "transcript": {"items": [{"role": "user", "content": ["hello"]}]},
            "structured_transcript": "human: hello",
            "recording_url": "https://example.invalid/recording",
        }
    }
    mock_request.return_value = _response(payload=payload)
    result = spec.get_call(CALL)
    sent = _request_of(mock_request)
    assert sent["method"] == "GET"
    assert sent["url"] == f"{DEFAULT_BASE_URL}/calls/{CALL}"
    assert result["data"]["summary"] == "Caller asked about billing."
    assert result["data"]["structured_transcript"] == "human: hello"
    assert result["data"]["recording_url"].endswith("/recording")


@patch("llama_index.tools.famulor.client.requests.request")
def test_list_leads(mock_request: MagicMock, spec: FamulorToolSpec) -> None:
    mock_request.return_value = _response(payload={"data": [{"id": LEAD}]})
    result = spec.list_leads(
        search="Jane",
        campaign_id=CAMPAIGN,
        display_status="created",
        dnc=True,
        limit=15,
        offset=2,
    )
    sent = _request_of(mock_request)
    assert sent["url"] == f"{DEFAULT_BASE_URL}/leads"
    assert sent["params"]["campaign_id"] == CAMPAIGN
    assert sent["params"]["dnc"] == "true"
    assert sent["params"]["search"] == "Jane"
    assert result["data"][0]["id"] == LEAD

    mock_request.reset_mock()
    mock_request.return_value = _response(payload={"data": []})
    spec.list_leads(campaign_id="none")
    assert _request_of(mock_request)["params"] == {"campaign_id": "none"}

    mock_request.reset_mock()
    mock_request.return_value = _response(payload={"data": []})
    spec.list_leads()
    assert "params" not in _request_of(mock_request)


@patch("llama_index.tools.famulor.client.requests.request")
def test_create_lead_posts_contact(
    mock_request: MagicMock, spec: FamulorToolSpec
) -> None:
    mock_request.return_value = _response(
        status=201, payload={"data": {"id": LEAD, "created": True}}
    )
    result = spec.create_lead(
        phone="+4915123456789",
        email="jane@example.com",
        name="Jane Doe",
        tags=["vip"],
        custom_fields={"company": "Acme"},
    )
    sent = _request_of(mock_request)
    assert sent["method"] == "POST"
    assert sent["url"] == f"{DEFAULT_BASE_URL}/leads"
    assert sent["json"] == {
        "phone": "+4915123456789",
        "email": "jane@example.com",
        "name": "Jane Doe",
        "tags": ["vip"],
        "custom_fields": {"company": "Acme"},
    }
    assert result["data"]["created"] is True


@patch("llama_index.tools.famulor.client.requests.request")
def test_create_lead_existing_contact_is_success(
    mock_request: MagicMock, spec: FamulorToolSpec
) -> None:
    mock_request.return_value = _response(
        status=200, payload={"data": {"id": LEAD, "created": False}}
    )
    result = spec.create_lead(email="jane@example.com")
    sent = _request_of(mock_request)
    assert "phone" not in sent["json"]
    assert sent["json"] == {"email": "jane@example.com"}
    assert result["data"]["created"] is False


@patch("llama_index.tools.famulor.client.requests.request")
def test_get_lead(mock_request: MagicMock, spec: FamulorToolSpec) -> None:
    mock_request.return_value = _response(
        payload={"data": {"id": LEAD, "custom_fields": {}, "system_variables": {}}}
    )
    result = spec.get_lead(CAMPAIGN, LEAD)
    sent = _request_of(mock_request)
    assert sent["method"] == "GET"
    assert sent["url"] == f"{DEFAULT_BASE_URL}/campaigns/{CAMPAIGN}/leads/{LEAD}"
    assert result["data"]["id"] == LEAD


@patch("llama_index.tools.famulor.client.requests.request")
def test_list_and_get_campaigns(mock_request: MagicMock, spec: FamulorToolSpec) -> None:
    mock_request.return_value = _response(payload={"data": [{"id": CAMPAIGN}]})
    spec.list_campaigns(status="draft", limit=10, offset=0)
    listed = _request_of(mock_request)
    assert listed["url"] == f"{DEFAULT_BASE_URL}/campaigns"
    assert listed["params"] == {"status": "draft", "limit": 10, "offset": 0}

    mock_request.reset_mock()
    mock_request.return_value = _response(
        payload={"data": {"id": CAMPAIGN, "total_leads": 3}}
    )
    result = spec.get_campaign(CAMPAIGN)
    fetched = _request_of(mock_request)
    assert fetched["url"] == f"{DEFAULT_BASE_URL}/campaigns/{CAMPAIGN}"
    assert result["data"]["total_leads"] == 3


@patch("llama_index.tools.famulor.client.requests.request")
def test_create_campaign_sends_only_selected_fields(
    mock_request: MagicMock, spec: FamulorToolSpec
) -> None:
    mock_request.return_value = _response(
        status=201, payload={"data": {"id": CAMPAIGN, "status": "draft"}}
    )
    spec.create_campaign(name="Spring outreach")
    assert _request_of(mock_request)["json"] == {"name": "Spring outreach"}

    mock_request.reset_mock()
    mock_request.return_value = _response(
        status=201, payload={"data": {"id": CAMPAIGN, "status": "draft"}}
    )
    spec.create_campaign(
        name="Spring outreach",
        assistant_id=ASSISTANT,
        channel="phone_call",
        timezone="Europe/Berlin",
    )
    assert _request_of(mock_request)["json"] == {
        "name": "Spring outreach",
        "assistant_id": ASSISTANT,
        "channel": "phone_call",
        "timezone": "Europe/Berlin",
    }


@patch("llama_index.tools.famulor.client.requests.request")
def test_make_call_posts_required_body(
    mock_request: MagicMock, spec: FamulorToolSpec
) -> None:
    mock_request.return_value = _response(
        status=201,
        payload={"data": {"id": CALL, "status": "queued", "to_number": "+4930123456"}},
    )
    result = spec.make_call(assistant_id=ASSISTANT, to_number="+4930123456")
    sent = _request_of(mock_request)
    assert sent["method"] == "POST"
    assert sent["url"] == f"{DEFAULT_BASE_URL}/calls"
    assert sent["headers"]["Authorization"] == f"Bearer {API_KEY}"
    assert set(sent["headers"]) == {"Authorization", "Accept"}
    assert sent["json"] == {"assistant_id": ASSISTANT, "to_number": "+4930123456"}
    assert result["data"]["status"] == "queued"


@patch("llama_index.tools.famulor.client.requests.request")
def test_make_call_includes_optional_fields(
    mock_request: MagicMock, spec: FamulorToolSpec
) -> None:
    mock_request.return_value = _response(
        status=201, payload={"data": {"id": CALL, "status": "queued"}}
    )
    spec.make_call(
        assistant_id=ASSISTANT,
        to_number="+4915123456789",
        phone_number_id=PHONE,
        lead={"name": "Jane Doe", "company": "Acme Corp"},
        variables={"account_id": "42", "priority": 1, "vip": True, "ratio": 0.5},
    )
    assert _request_of(mock_request)["json"] == {
        "assistant_id": ASSISTANT,
        "to_number": "+4915123456789",
        "phone_number_id": PHONE,
        "lead": {"name": "Jane Doe", "company": "Acme Corp"},
        "variables": {"account_id": "42", "priority": 1, "vip": True, "ratio": 0.5},
    }


@patch("llama_index.tools.famulor.client.requests.request")
def test_make_call_uses_whitelabel_host(
    mock_request: MagicMock,
) -> None:
    mock_request.return_value = _response(
        status=201, payload={"data": {"id": CALL, "status": "queued"}}
    )
    tool = FamulorToolSpec(
        api_key=API_KEY,
        base_url="https://voice.example.com/api/v1",
        timeout=5,
    )
    tool.make_call(assistant_id=ASSISTANT, to_number="+4930123456")
    sent = _request_of(mock_request)
    assert sent["url"] == "https://voice.example.com/api/v1/calls"
    assert sent["timeout"] == 5.0


@patch("llama_index.tools.famulor.client.requests.request")
@pytest.mark.parametrize(
    ("method_name", "kwargs"),
    [
        ("get_assistant", {"assistant_id": "not-a-uuid"}),
        ("get_call", {"call_id": "not-a-uuid"}),
        ("get_lead", {"campaign_id": CAMPAIGN, "lead_id": "nope"}),
        ("get_campaign", {"campaign_id": "nope"}),
        ("list_assistants", {"limit": 0}),
        ("list_assistants", {"limit": 201}),
        ("list_assistants", {"offset": -1}),
        ("list_phone_numbers", {"status": "deleted"}),
        ("list_phone_numbers", {"country": "DEU"}),
        ("list_calls", {"direction": "sideways"}),
        ("list_calls", {"success": "yes"}),
        ("list_leads", {"campaign_id": "not-uuid-or-none"}),
        ("list_leads", {"dnc": "yes"}),
        ("create_lead", {}),
        ("create_lead", {"phone": "+4915123456789", "tags": "vip"}),
        ("create_lead", {"phone": "+4915123456789", "tags": ["a"] * 33}),
        ("create_lead", {"phone": "+4915123456789", "tags": [""]}),
        (
            "create_lead",
            {"tags": ["this tag is definitely longer than forty characters"]},
        ),
        ("create_campaign", {"name": None}),
        ("create_campaign", {"name": "  "}),
        ("create_campaign", {"name": "Draft", "channel": "email"}),
        ("make_call", {"assistant_id": ASSISTANT, "to_number": "015123456789"}),
        (
            "make_call",
            {
                "assistant_id": ASSISTANT,
                "to_number": "+4930123456",
                "variables": {"Bad": "x"},
            },
        ),
        (
            "make_call",
            {
                "assistant_id": ASSISTANT,
                "to_number": "+4930123456",
                "variables": {"note": "x" * 2001},
            },
        ),
        (
            "make_call",
            {
                "assistant_id": ASSISTANT,
                "to_number": "+4930123456",
                "variables": {"__proto__": "no"},
            },
        ),
        (
            "make_call",
            {
                "assistant_id": ASSISTANT,
                "to_number": "+4930123456",
                "variables": {"score": float("nan")},
            },
        ),
        (
            "make_call",
            {"assistant_id": ASSISTANT, "to_number": "+4930123456", "lead": ["Jane"]},
        ),
        (
            "make_call",
            {
                "assistant_id": ASSISTANT,
                "to_number": "+4930123456",
                "variables": ["nope"],
            },
        ),
        (
            "make_call",
            {
                "assistant_id": ASSISTANT,
                "to_number": "+4930123456",
                "variables": {"score": {"nested": True}},
            },
        ),
    ],
)
def test_validation_does_not_send_http(
    mock_request: MagicMock,
    spec: FamulorToolSpec,
    method_name: str,
    kwargs: Dict[str, Any],
) -> None:
    with pytest.raises(ValueError):
        getattr(spec, method_name)(**kwargs)
    mock_request.assert_not_called()


@patch("llama_index.tools.famulor.client.requests.request")
def test_api_error_envelope(mock_request: MagicMock, spec: FamulorToolSpec) -> None:
    mock_request.return_value = _response(
        status=403,
        payload={
            "error": {
                "code": "destination_forbidden",
                "message": "That destination is not allowed.",
            }
        },
    )
    with pytest.raises(FamulorAPIError) as caught:
        spec.make_call(assistant_id=ASSISTANT, to_number="+4930123456")
    error = caught.value
    assert error.status_code == 403
    assert error.code == "destination_forbidden"
    assert error.message == "That destination is not allowed."


@patch("llama_index.tools.famulor.client.requests.request")
def test_rate_limit_includes_retry_after(
    mock_request: MagicMock, spec: FamulorToolSpec
) -> None:
    mock_request.return_value = _response(
        status=429,
        payload={"error": {"code": "rate_limited", "message": "Too many requests."}},
        headers={"Retry-After": "12"},
    )
    with pytest.raises(FamulorAPIError) as caught:
        spec.list_assistants()
    assert caught.value.retry_after == "12"
    assert "12" in str(caught.value)
    assert caught.value.code == "rate_limited"


@patch("llama_index.tools.famulor.client.requests.request")
def test_non_json_error(mock_request: MagicMock, spec: FamulorToolSpec) -> None:
    mock_request.return_value = _response(
        status=502, text="bad gateway", json_error=True
    )
    with pytest.raises(FamulorAPIError) as caught:
        spec.get_assistant(ASSISTANT)
    assert caught.value.status_code == 502
    assert caught.value.code == "http_error"
    assert caught.value.message == "bad gateway"


@patch("llama_index.tools.famulor.client.requests.request")
def test_json_error_without_envelope(
    mock_request: MagicMock, spec: FamulorToolSpec
) -> None:
    mock_request.return_value = _response(
        status=500, payload={"oops": True}, text="nope"
    )
    with pytest.raises(FamulorAPIError) as caught:
        spec.list_campaigns()
    assert caught.value.code == "http_error"
    assert caught.value.message == "nope"


@patch("llama_index.tools.famulor.client.requests.request")
def test_success_without_json_object(
    mock_request: MagicMock, spec: FamulorToolSpec
) -> None:
    response = _response(status=200)
    response.json.return_value = ["not", "an", "object"]
    mock_request.return_value = response
    with pytest.raises(FamulorAPIError) as caught:
        spec.list_assistants()
    assert caught.value.code == "invalid_response"


@patch("llama_index.tools.famulor.client.requests.request")
def test_redirect_is_not_followed(
    mock_request: MagicMock, spec: FamulorToolSpec
) -> None:
    mock_request.return_value = _response(status=302, text="moved", json_error=True)
    with pytest.raises(FamulorAPIError) as caught:
        spec.list_phone_numbers()
    assert caught.value.status_code == 302
    assert _request_of(mock_request)["allow_redirects"] is False


@patch("llama_index.tools.famulor.client.requests.request")
def test_transport_error(mock_request: MagicMock, spec: FamulorToolSpec) -> None:
    mock_request.side_effect = requests.ConnectionError("connection refused")
    with pytest.raises(FamulorAPIError) as caught:
        spec.list_calls()
    assert caught.value.status_code == 0
    assert caught.value.code == "request_failed"
    assert isinstance(caught.value.__cause__, requests.ConnectionError)
