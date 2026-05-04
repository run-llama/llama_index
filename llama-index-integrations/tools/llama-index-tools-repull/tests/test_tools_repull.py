"""
Tests for ``llama-index-tools-repull``.

The tool wraps the typed ``repull-sdk`` HTTP client. We don't hit the network in
unit tests — instead we patch the per-endpoint ``sync`` callables in the
``repull.api.*`` modules and assert the tool spec marshals arguments and
unwraps responses correctly.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.tools.repull import RepullToolSpec


# ---------------------------------------------------------------------- #
# helpers
# ---------------------------------------------------------------------- #


class _Stub:
    """
    Minimal stand-in for a repull-sdk attrs response object.

    Mirrors the contract ``RepullToolSpec._to_dict`` relies on: anything with a
    ``.to_dict()`` method gets unwrapped.
    """

    def __init__(self, payload):
        self._payload = payload

    def to_dict(self):
        return self._payload


@pytest.fixture()
def tool() -> RepullToolSpec:
    return RepullToolSpec(api_key="sk_test_unit", base_url="https://api.repull.dev")


# ---------------------------------------------------------------------- #
# class-level
# ---------------------------------------------------------------------- #


def test_class():
    """Smoke test: the spec inherits from BaseToolSpec (LlamaHub contract)."""
    names_of_base_classes = [b.__name__ for b in RepullToolSpec.__mro__]
    assert BaseToolSpec.__name__ in names_of_base_classes


def test_spec_functions_exposed_as_tools(tool: RepullToolSpec):
    """
    Every entry in ``spec_functions`` must resolve to a real method and be
    convertible into a LlamaIndex ``FunctionTool``.
    """
    for name in tool.spec_functions:
        assert callable(getattr(tool, name)), f"missing method: {name}"

    fn_tools = tool.to_tool_list()
    exposed = {t.metadata.name for t in fn_tools}
    assert exposed == set(tool.spec_functions)


def test_to_tool_list_preserves_docstrings(tool: RepullToolSpec):
    """
    Docstrings are load-bearing — they're what the LLM reads to decide
    whether to call the tool. Make sure they survive the FunctionTool wrap.
    """
    fn_tools = {t.metadata.name: t for t in tool.to_tool_list()}
    assert "properties" in fn_tools["list_properties"].metadata.description.lower()
    assert "oauth" in fn_tools["create_connect_session"].metadata.description.lower()


# ---------------------------------------------------------------------- #
# helpers
# ---------------------------------------------------------------------- #


def test_to_dict_passes_through_primitives():
    """Plain values, lists, dicts, and ``None`` all round-trip unchanged."""
    assert RepullToolSpec._to_dict(None) is None
    assert RepullToolSpec._to_dict(42) == 42
    assert RepullToolSpec._to_dict("hello") == "hello"
    assert RepullToolSpec._to_dict([1, 2]) == [1, 2]
    assert RepullToolSpec._to_dict({"a": 1}) == {"a": 1}


def test_to_dict_unwraps_attrs_objects():
    """
    Anything exposing ``to_dict()`` gets unwrapped — that's the SDK
    response contract.
    """
    payload = {"id": 1, "name": "Cabin"}
    assert RepullToolSpec._to_dict(_Stub(payload)) == payload

    nested = [_Stub({"id": 1}), _Stub({"id": 2})]
    assert RepullToolSpec._to_dict(nested) == [{"id": 1}, {"id": 2}]


# ---------------------------------------------------------------------- #
# endpoint wiring
# ---------------------------------------------------------------------- #


def test_list_properties_dispatches_to_sdk(tool: RepullToolSpec):
    """
    ``list_properties`` calls ``repull.api.properties.list_properties.sync``
    with the client and a status enum, and unwraps the response.
    """
    fake_response = _Stub({"data": [{"id": 1}], "pagination": {"has_more": False}})

    with patch(
        "repull.api.properties.list_properties.sync", return_value=fake_response
    ) as mock:
        out = tool.list_properties(limit=10, status="active")

    assert out == {"data": [{"id": 1}], "pagination": {"has_more": False}}
    kwargs = mock.call_args.kwargs
    assert kwargs["client"] is tool._client
    assert kwargs["limit"] == 10
    # status string lifted into the SDK enum
    assert kwargs["status"].value == "active"


def test_get_property_dispatches_to_sdk(tool: RepullToolSpec):
    fake_response = _Stub({"id": 99, "name": "Lakeside Cabin"})

    with patch(
        "repull.api.properties.get_property.sync", return_value=fake_response
    ) as mock:
        out = tool.get_property(99)

    assert out == {"id": 99, "name": "Lakeside Cabin"}
    assert mock.call_args.kwargs["id"] == 99
    assert mock.call_args.kwargs["client"] is tool._client


def test_list_reservations_parses_iso_dates(tool: RepullToolSpec):
    """
    Date params come in as ISO strings from the LLM and must be parsed
    into ``datetime.date`` for the SDK.
    """
    import datetime as _dt

    fake_response = _Stub({"data": [], "pagination": {"has_more": False}})

    with patch(
        "repull.api.reservations.list_reservations.sync", return_value=fake_response
    ) as mock:
        tool.list_reservations(
            platform="airbnb",
            check_in_after="2026-05-10",
            check_in_before="2026-05-20",
            status="confirmed",
        )

    kwargs = mock.call_args.kwargs
    assert kwargs["platform"] == "airbnb"
    assert kwargs["check_in_after"] == _dt.date(2026, 5, 10)
    assert kwargs["check_in_before"] == _dt.date(2026, 5, 20)
    assert kwargs["status"].value == "confirmed"


def test_search_markets_lifts_sort_into_enum(tool: RepullToolSpec):
    fake_response = _Stub({"data": [{"city": "Aspen"}]})

    with patch(
        "repull.api.markets.list_market_browse.sync", return_value=fake_response
    ) as mock:
        out = tool.search_markets(q="Aspen", country="US", sort="listings_desc")

    assert out == {"data": [{"city": "Aspen"}]}
    kwargs = mock.call_args.kwargs
    assert kwargs["q"] == "Aspen"
    assert kwargs["country"] == "US"
    assert kwargs["sort"].value == "listings_desc"


def test_get_market_dispatches_with_city_and_page(tool: RepullToolSpec):
    fake_response = _Stub({"city": "Lisbon", "comps": []})

    with patch(
        "repull.api.markets.get_market.sync", return_value=fake_response
    ) as mock:
        out = tool.get_market("Lisbon", comps_page=2)

    assert out == {"city": "Lisbon", "comps": []}
    assert mock.call_args.kwargs["city"] == "Lisbon"
    assert mock.call_args.kwargs["comps_page"] == 2


def test_list_conversations_validates_platform_enum(tool: RepullToolSpec):
    """
    Bad platform values must raise — the SDK enum is the source of truth.
    This protects the agent from silently sending an invalid filter.
    """
    with pytest.raises(ValueError):
        tool.list_conversations(platform="not-a-real-channel")


def test_create_connect_session_builds_body(tool: RepullToolSpec):
    """
    The session call wraps its inputs in ``CreateConnectSessionBody`` —
    verify both fields land in the constructed body.
    """
    from repull.models.create_connect_session_body import CreateConnectSessionBody

    fake_response = _Stub(
        {"url": "https://connect.repull.dev/sess_123", "session_id": "sess_123"}
    )

    with patch(
        "repull.api.connect.create_connect_session.sync", return_value=fake_response
    ) as mock:
        out = tool.create_connect_session(
            redirect_url="https://example.com/return",
            allowed_providers=["airbnb"],
        )

    assert out == {
        "url": "https://connect.repull.dev/sess_123",
        "session_id": "sess_123",
    }
    body = mock.call_args.kwargs["body"]
    assert isinstance(body, CreateConnectSessionBody)
    assert body.redirect_url == "https://example.com/return"
    assert body.allowed_providers == ["airbnb"]


def test_list_markets_passes_through(tool: RepullToolSpec):
    fake_response = _Stub({"markets": [{"city": "Miami"}], "browse": {}})

    with patch(
        "repull.api.markets.list_markets.sync", return_value=fake_response
    ) as mock:
        out = tool.list_markets()

    assert out == {"markets": [{"city": "Miami"}], "browse": {}}
    assert mock.call_args.kwargs["client"] is tool._client


def test_handles_none_response_from_sdk(tool: RepullToolSpec):
    """
    ``repull-sdk`` returns ``None`` for unexpected statuses (e.g. 404)
    when ``raise_on_unexpected_status`` is off. The tool must surface that
    cleanly rather than crashing.
    """
    with patch("repull.api.properties.get_property.sync", return_value=None):
        assert tool.get_property(99999) is None
