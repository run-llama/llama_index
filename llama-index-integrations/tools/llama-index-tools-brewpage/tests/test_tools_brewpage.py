"""Tests for BrewPage Tool Spec."""

import json

import responses
from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.tools.brewpage import BrewPageToolSpec


def test_class() -> None:
    """Test that BrewPageToolSpec is a valid BaseToolSpec."""
    names_of_base_classes = [b.__name__ for b in BrewPageToolSpec.__mro__]
    assert BaseToolSpec.__name__ in names_of_base_classes


@responses.activate
def test_publish_content() -> None:
    """Test publishing content to BrewPage."""
    tool = BrewPageToolSpec()

    # Mock the POST request
    mock_response = {
        "namespace": "public",
        "id": "abc123def4",
        "link": "https://brewpage.app/public/abc123def4",
    }
    responses.add(
        responses.POST,
        "https://brewpage.app/api/html",
        json=mock_response,
        status=200,
    )

    # Test publish
    result = tool.publish_content(
        "<h1>Test Content</h1>",
        namespace="public",
    )

    assert result == "https://brewpage.app/public/abc123def4"
    assert len(responses.calls) == 1
    request_body = json.loads(responses.calls[0].request.body)
    assert request_body["content"] == "<h1>Test Content</h1>"
    assert request_body["namespace"] == "public"


@responses.activate
def test_publish_content_with_ttl() -> None:
    """Test publishing content with custom TTL."""
    tool = BrewPageToolSpec()

    mock_response = {
        "namespace": "public",
        "id": "xyz789abc1",
        "link": "https://brewpage.app/public/xyz789abc1",
    }
    responses.add(
        responses.POST,
        "https://brewpage.app/api/html",
        json=mock_response,
        status=200,
    )

    result = tool.publish_content(
        "# Markdown Content",
        namespace="public",
        ttl_days=7,
    )

    assert result == "https://brewpage.app/public/xyz789abc1"
    request_body = json.loads(responses.calls[0].request.body)
    assert request_body["ttl_days"] == 7


@responses.activate
def test_get_content() -> None:
    """Test retrieving content from BrewPage."""
    tool = BrewPageToolSpec()

    mock_response = {
        "namespace": "public",
        "id": "abc123def4",
        "body": "<h1>Test Content</h1>",
    }
    responses.add(
        responses.GET,
        "https://brewpage.app/api/html/public/abc123def4",
        json=mock_response,
        status=200,
    )

    result = tool.get_content("public", "abc123def4")

    assert result == "<h1>Test Content</h1>"
    assert len(responses.calls) == 1


@responses.activate
def test_get_content_not_found() -> None:
    """Test retrieving non-existent content raises error."""
    tool = BrewPageToolSpec()

    responses.add(
        responses.GET,
        "https://brewpage.app/api/html/public/notfound",
        status=404,
    )

    try:
        tool.get_content("public", "notfound")
        raise AssertionError("Expected RequestException")
    except Exception as e:
        assert "404" in str(e)


def test_custom_base_url() -> None:
    """Test initializing with custom base URL."""
    custom_url = "http://localhost:8000"
    tool = BrewPageToolSpec(base_url=custom_url)
    assert tool.base_url == custom_url
