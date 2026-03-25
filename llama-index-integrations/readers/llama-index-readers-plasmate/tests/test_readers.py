"""Tests for PlasmateWebReader."""

import json
from unittest.mock import MagicMock, patch

import pytest
from llama_index.readers.plasmate import PlasmateWebReader


def test_class_exists():
    """Verify PlasmateWebReader can be imported."""
    assert PlasmateWebReader is not None


def test_default_init():
    """Test default initialization parameters."""
    reader = PlasmateWebReader()
    assert reader.timeout == 30
    assert reader.javascript is True


def test_custom_init():
    """Test custom initialization parameters."""
    reader = PlasmateWebReader(timeout=60, javascript=False)
    assert reader.timeout == 60
    assert reader.javascript is False


@patch("llama_index.readers.plasmate.base.subprocess.run")
def test_load_data_success(mock_run):
    """Test successful document loading."""
    som_output = {
        "title": "Example Page",
        "lang": "en",
        "som_version": "1.0",
        "regions": [
            {
                "elements": [
                    {
                        "role": "heading",
                        "text": "Hello World",
                        "attrs": {"level": 1},
                    },
                    {"role": "paragraph", "text": "This is a test page."},
                ]
            }
        ],
    }
    mock_run.return_value = MagicMock(
        returncode=0, stdout=json.dumps(som_output), stderr=""
    )

    reader = PlasmateWebReader()
    docs = reader.load_data(urls=["https://example.com"])

    assert len(docs) == 1
    assert "# Hello World" in docs[0].text
    assert "This is a test page." in docs[0].text
    assert docs[0].metadata["url"] == "https://example.com"
    assert docs[0].metadata["title"] == "Example Page"
    assert docs[0].metadata["source"] == "plasmate"


@patch("llama_index.readers.plasmate.base.subprocess.run")
def test_load_data_failure(mock_run):
    """Test that failed fetches are skipped gracefully."""
    mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="error")

    reader = PlasmateWebReader()
    docs = reader.load_data(urls=["https://bad-url.example"])

    assert len(docs) == 0


@patch("llama_index.readers.plasmate.base.subprocess.run")
def test_load_data_multiple_urls(mock_run):
    """Test loading from multiple URLs."""
    som_output = {
        "title": "Page",
        "lang": "en",
        "som_version": "1.0",
        "regions": [{"elements": [{"role": "paragraph", "text": "Content"}]}],
    }
    mock_run.return_value = MagicMock(
        returncode=0, stdout=json.dumps(som_output), stderr=""
    )

    reader = PlasmateWebReader()
    docs = reader.load_data(
        urls=["https://example.com/1", "https://example.com/2"]
    )

    assert len(docs) == 2
    assert mock_run.call_count == 2
