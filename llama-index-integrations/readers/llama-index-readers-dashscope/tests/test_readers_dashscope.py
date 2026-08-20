import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from llama_index.core.readers.base import BasePydanticReader
from llama_index.readers.dashscope import DashScopeParse


def test_class():
    names_of_base_classes = [b.__name__ for b in DashScopeParse.__mro__]
    assert BasePydanticReader.__name__ in names_of_base_classes


def test_upload_file_with_lease_runs_blocking_upload(tmp_path):
    parse = DashScopeParse(api_key="k")
    src = tmp_path / "doc.pdf"
    src.write_bytes(b"data")
    fake_lease = MagicMock()

    with patch.object(
        DashScopeParse, "_DashScopeParse__upload_lease", return_value=fake_lease
    ):
        result = parse._upload_file_with_lease(str(src), {})

    assert result is fake_lease
    fake_lease.upload.assert_called_once()


def test_create_job_offloads_blocking_upload():
    parse = DashScopeParse(api_key="k")
    fake_lease = MagicMock(lease_id="lease-1")

    async def run():
        client = MagicMock()
        client.post = AsyncMock(return_value=MagicMock())
        with (
            patch(
                "llama_index.readers.dashscope.base.UploadFileLeaseResult.is_file_valid"
            ),
            patch.object(DashScopeParse, "_get_dashscope_header", return_value={}),
            patch.object(
                DashScopeParse, "_upload_file_with_lease", return_value=fake_lease
            ) as mock_upload,
            patch(
                "llama_index.readers.dashscope.base.httpx.AsyncClient"
            ) as mock_client_cls,
            patch(
                "llama_index.readers.dashscope.base.dashscope_response_handler",
                return_value=MagicMock(file_id="file-1"),
            ),
            patch("asyncio.to_thread", wraps=asyncio.to_thread) as spy,
        ):
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await parse._create_job("doc.pdf")

        # The blocking upload must run off the event loop via asyncio.to_thread.
        spy.assert_called_once()
        mock_upload.assert_called_once()
        assert result == "file-1"

    asyncio.run(run())
