"""
Tests for download module security validations.

Validates that path traversal and SSRF protections prevent malicious
values from remote manifests (library.json) from being used to write
files outside intended directories or make requests to non-HTTPS URLs.
"""

import json
from unittest.mock import patch

import pytest
from llama_index.core.download.utils import (
    _validate_path_component,
    _validate_url,
)


class TestValidatePathComponent:
    """Test _validate_path_component against directory traversal attacks."""

    def test_simple_filename(self):
        """Normal filenames should pass validation."""
        _validate_path_component("base.py", "file")
        _validate_path_component("utils.py", "file")
        _validate_path_component("__init__.py", "file")

    def test_nested_path(self):
        """Nested paths like 'web/simple_web' should pass validation."""
        _validate_path_component("web/simple_web", "module_id")
        _validate_path_component("readers/github", "module_id")

    def test_traversal_simple(self):
        """Paths with '..' should be rejected."""
        with pytest.raises(ValueError, match="directory traversal"):
            _validate_path_component("../../etc/passwd", "module_id")

    def test_traversal_embedded(self):
        """Paths with embedded '..' should be rejected."""
        with pytest.raises(ValueError, match="directory traversal"):
            _validate_path_component("web/../../etc/passwd", "module_id")

    def test_traversal_backslash(self):
        """Windows-style backslash traversal should be rejected."""
        with pytest.raises(ValueError, match="directory traversal"):
            _validate_path_component("..\\..\\windows\\system32", "module_id")

    def test_absolute_path_unix(self):
        """Unix absolute paths should be rejected."""
        with pytest.raises(ValueError, match="absolute path"):
            _validate_path_component("/etc/passwd", "module_id")

    def test_absolute_path_windows(self):
        """Windows absolute paths should be rejected."""
        with pytest.raises(ValueError, match="absolute path"):
            _validate_path_component("C:\\Windows\\System32", "module_id")

    def test_null_byte(self):
        """Null bytes should be rejected."""
        with pytest.raises(ValueError, match="null byte"):
            _validate_path_component("file\x00.py", "file")

    def test_empty_string(self):
        """Empty strings should be rejected."""
        with pytest.raises(ValueError, match="Empty"):
            _validate_path_component("", "module_id")


class TestValidateUrl:
    """Test _validate_url against SSRF attacks."""

    def test_https_url(self):
        """HTTPS URLs should pass validation."""
        _validate_url("https://raw.githubusercontent.com/org/repo/main/file.json")

    def test_http_url_rejected(self):
        """HTTP URLs should be rejected (downgrade attack)."""
        with pytest.raises(ValueError, match="not allowed"):
            _validate_url("http://internal-server.local/library.json")

    def test_file_url_rejected(self):
        """file:// URLs should be rejected."""
        with pytest.raises(ValueError, match="not allowed"):
            _validate_url("file:///etc/passwd")

    def test_ftp_url_rejected(self):
        """FTP URLs should be rejected."""
        with pytest.raises(ValueError, match="not allowed"):
            _validate_url("ftp://server/file")


class TestGetModuleInfoPathTraversal:
    """Test that get_module_info rejects malicious library.json entries."""

    @patch("llama_index.core.download.module.get_file_content")
    def test_malicious_module_id_from_remote(self, mock_get):
        """A malicious library.json with traversal in module_id should be rejected."""
        from llama_index.core.download.module import get_module_info

        malicious_library = {
            "MyLoader": {
                "id": "../../.ssh/authorized_keys",
                "extra_files": [],
            }
        }
        mock_get.return_value = (json.dumps(malicious_library), 200)

        with pytest.raises(ValueError, match="directory traversal"):
            get_module_info(
                local_dir_path="/tmp/test",
                remote_dir_path="https://raw.githubusercontent.com/test",
                module_class="MyLoader",
                refresh_cache=True,
            )

    @patch("llama_index.core.download.module.get_file_content")
    def test_malicious_extra_file_from_remote(self, mock_get):
        """A malicious library.json with traversal in extra_files should be rejected."""
        from llama_index.core.download.module import get_module_info

        malicious_library = {
            "MyLoader": {
                "id": "web/loader",
                "extra_files": ["../../../.bashrc"],
            }
        }
        mock_get.return_value = (json.dumps(malicious_library), 200)

        with pytest.raises(ValueError, match="directory traversal"):
            get_module_info(
                local_dir_path="/tmp/test",
                remote_dir_path="https://raw.githubusercontent.com/test",
                module_class="MyLoader",
                refresh_cache=True,
            )

    def test_malicious_module_id_from_cache(self, tmp_path):
        """A tampered cached library.json should also be validated."""
        from llama_index.core.download.module import get_module_info

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        malicious_library = {
            "MyLoader": {
                "id": "../../etc",
                "extra_files": [],
            }
        }
        (cache_dir / "library.json").write_text(json.dumps(malicious_library))

        with pytest.raises(ValueError, match="directory traversal"):
            get_module_info(
                local_dir_path=str(cache_dir),
                remote_dir_path="https://raw.githubusercontent.com/test",
                module_class="MyLoader",
            )
