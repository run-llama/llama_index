from unittest.mock import MagicMock, patch

import pytest
from llama_index.core.readers.base import BaseReader
from llama_index.readers.docugami import DocugamiReader


def test_class():
    names_of_base_classes = [b.__name__ for b in DocugamiReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes


def test_download_error_includes_status_code():
    artifacts = MagicMock(ok=True)
    artifacts.json.return_value = {
        "artifacts": [
            {
                "name": "report-values.xml",
                "url": "https://example.com/artifact",
                "document": {"id": "doc1"},
            }
        ]
    }
    failed = MagicMock(ok=False, status_code=403)

    reader = DocugamiReader(access_token="test")
    with patch(
        "llama_index.readers.docugami.base.requests.request",
        side_effect=[artifacts, failed],
    ):
        with pytest.raises(Exception, match="status: 403"):
            reader._metadata_for_project({"id": "p1"})
