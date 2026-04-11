"""Tests for async document resolution (aresolve_binary and aresolve_document).

These tests verify that:
1. aresolve_binary correctly handles raw bytes, file paths, data URLs, and HTTP URLs
2. aresolve_document on DocumentBlock uses async HTTP for URL-based documents
3. The async path produces identical results to the sync path for non-network sources
"""

import asyncio
import base64
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# aresolve_binary tests
# ---------------------------------------------------------------------------

from llama_index.core.utils import aresolve_binary, resolve_binary


@pytest.mark.asyncio
async def test_aresolve_binary_raw_bytes():
    """aresolve_binary should handle raw bytes identically to resolve_binary."""
    data = b"hello world"
    result = await aresolve_binary(raw_bytes=data)
    expected = resolve_binary(raw_bytes=data)
    assert result.read() == expected.read()


@pytest.mark.asyncio
async def test_aresolve_binary_raw_bytes_base64():
    """aresolve_binary should base64-encode raw bytes when as_base64=True."""
    data = b"hello world"
    result = await aresolve_binary(raw_bytes=data, as_base64=True)
    expected = resolve_binary(raw_bytes=data, as_base64=True)
    assert result.read() == expected.read()


@pytest.mark.asyncio
async def test_aresolve_binary_path(tmp_path):
    """aresolve_binary should read from file paths."""
    test_file = tmp_path / "test.bin"
    test_file.write_bytes(b"file content")

    result = await aresolve_binary(path=test_file)
    expected = resolve_binary(path=test_file)
    assert result.read() == expected.read()


@pytest.mark.asyncio
async def test_aresolve_binary_data_url():
    """aresolve_binary should handle data: URLs without network I/O."""
    encoded = base64.b64encode(b"test data").decode()
    data_url = f"data:application/octet-stream;base64,{encoded}"

    result = await aresolve_binary(url=data_url)
    expected = resolve_binary(url=data_url)
    assert result.read() == expected.read()


@pytest.mark.asyncio
async def test_aresolve_binary_http_url_uses_httpx():
    """aresolve_binary should use httpx.AsyncClient for HTTP URLs."""
    mock_response = MagicMock()
    mock_response.content = b"remote content"
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("llama_index.core.utils.httpx") as mock_httpx:
        mock_httpx.AsyncClient.return_value = mock_client
        mock_httpx.Timeout = MagicMock()

        result = await aresolve_binary(url="https://example.com/file.pdf")
        assert result.read() == b"remote content"
        mock_client.get.assert_awaited_once()


@pytest.mark.asyncio
async def test_aresolve_binary_http_url_base64():
    """aresolve_binary should base64-encode HTTP response when as_base64=True."""
    mock_response = MagicMock()
    mock_response.content = b"remote content"
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("llama_index.core.utils.httpx") as mock_httpx:
        mock_httpx.AsyncClient.return_value = mock_client
        mock_httpx.Timeout = MagicMock()

        result = await aresolve_binary(
            url="https://example.com/file.pdf", as_base64=True
        )
        assert result.read() == base64.b64encode(b"remote content")


@pytest.mark.asyncio
async def test_aresolve_binary_no_source():
    """aresolve_binary should raise ValueError when no source is provided."""
    with pytest.raises(ValueError, match="No valid source provided"):
        await aresolve_binary()


# ---------------------------------------------------------------------------
# DocumentBlock.aresolve_document tests
# ---------------------------------------------------------------------------

from llama_index.core.base.llms.types import DocumentBlock


@pytest.mark.asyncio
async def test_document_block_aresolve_document_with_data():
    """aresolve_document should work with pre-loaded data."""
    raw = b"document content"
    block = DocumentBlock(data=raw, document_mimetype="application/pdf")

    result = await block.aresolve_document()
    result.seek(0)
    content = result.read()
    assert len(content) > 0


@pytest.mark.asyncio
async def test_document_block_aresolve_document_with_url():
    """aresolve_document should use async HTTP for URL-based documents."""
    mock_response = MagicMock()
    mock_response.content = b"fetched document"
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("llama_index.core.utils.httpx") as mock_httpx:
        mock_httpx.AsyncClient.return_value = mock_client
        mock_httpx.Timeout = MagicMock()

        block = DocumentBlock(
            url="https://example.com/doc.pdf",
            document_mimetype="application/pdf",
        )
        result = await block.aresolve_document()
        result.seek(0)
        assert result.read() == b"fetched document"
        mock_client.get.assert_awaited_once()


@pytest.mark.asyncio
async def test_document_block_aresolve_document_zero_bytes():
    """aresolve_document should raise ValueError for empty documents."""
    mock_response = MagicMock()
    mock_response.content = b""
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("llama_index.core.utils.httpx") as mock_httpx:
        mock_httpx.AsyncClient.return_value = mock_client
        mock_httpx.Timeout = MagicMock()

        block = DocumentBlock(
            url="https://example.com/empty.pdf",
            document_mimetype="application/pdf",
        )
        with pytest.raises(ValueError, match="resolve_document returned zero bytes"):
            await block.aresolve_document()


@pytest.mark.asyncio
async def test_document_block_aresolve_document_with_path(tmp_path):
    """aresolve_document should work with local file paths."""
    test_file = tmp_path / "test.pdf"
    test_file.write_bytes(b"%PDF-1.4 test content")

    block = DocumentBlock(
        path=str(test_file), document_mimetype="application/pdf"
    )
    result = await block.aresolve_document()
    result.seek(0)
    assert result.read() == b"%PDF-1.4 test content"


@pytest.mark.asyncio
async def test_document_block_aestimate_tokens_uses_async():
    """aestimate_tokens should use aresolve_document (async path)."""
    mock_response = MagicMock()
    mock_response.content = b"some content"
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("llama_index.core.utils.httpx") as mock_httpx:
        mock_httpx.AsyncClient.return_value = mock_client
        mock_httpx.Timeout = MagicMock()

        block = DocumentBlock(
            url="https://example.com/doc.pdf",
            document_mimetype="application/pdf",
        )
        tokens = await block.aestimate_tokens()
        assert tokens == 512
        # Verify httpx was used (async path), not requests (sync path)
        mock_client.get.assert_awaited_once()
