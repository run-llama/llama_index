"""Regression tests for #24358: NLTK cache dir permissions.

nltk >=3.10.4 refuses to read from or download into a directory that is
group- or world-writable. wait_for_nltk_check() must create (or fix)
the cache directory so that it is not group/world-writable.
"""

import os
import stat
from pathlib import Path
from unittest.mock import patch

import pytest

from llama_index.core.utils import GlobalsHelper


@pytest.fixture()
def tmp_cache(tmp_path: Path):
    """Provide a temporary cache directory and patch env."""
    cache_dir = tmp_path / "nltk_cache"
    with patch.dict(os.environ, {"NLTK_DATA": str(cache_dir)}):
        yield cache_dir


class TestNLTKCachePermissions:
    """Cache directory must not be group/world-writable after setup."""

    def test_new_dir_created_without_group_world_write(self, tmp_cache: Path):
        """A freshly created cache dir must not be g/o writable."""
        assert not tmp_cache.exists()

        helper = GlobalsHelper()
        helper.wait_for_nltk_check()

        assert tmp_cache.exists()
        mode = tmp_cache.stat().st_mode
        assert not (mode & stat.S_IWGRP), "Cache dir is group-writable"
        assert not (mode & stat.S_IWOTH), "Cache dir is world-writable"

    def test_existing_writable_dir_gets_fixed(self, tmp_cache: Path):
        """An existing group-writable dir must be tightened."""
        tmp_cache.mkdir(parents=True)
        os.chmod(tmp_cache, 0o775)
        assert tmp_cache.stat().st_mode & stat.S_IWGRP

        helper = GlobalsHelper()
        helper.wait_for_nltk_check()

        mode = tmp_cache.stat().st_mode
        assert not (mode & stat.S_IWGRP), "Cache dir is still group-writable"
        assert not (mode & stat.S_IWOTH), "Cache dir is still world-writable"

    def test_already_secure_dir_unchanged(self, tmp_cache: Path):
        """A dir that is already 700 should not be changed."""
        tmp_cache.mkdir(parents=True, mode=0o700)
        original_mode = tmp_cache.stat().st_mode

        helper = GlobalsHelper()
        helper.wait_for_nltk_check()

        assert tmp_cache.stat().st_mode == original_mode
