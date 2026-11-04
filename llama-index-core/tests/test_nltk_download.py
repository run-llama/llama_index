"""Tests for the bounded NLTK data download in ``llama_index.core.utils``."""

import os
import time

from llama_index.core import utils as utils_module
from llama_index.core.utils import GlobalsHelper, globals_helper


def test_download_nltk_data_times_out(monkeypatch, tmp_path, capsys) -> None:
    """A stuck ``nltk.download`` must not block the caller forever."""
    monkeypatch.setattr(globals_helper, "_nltk_data_dir", str(tmp_path))
    monkeypatch.setattr(utils_module, "_NLTK_DOWNLOAD_TIMEOUT_SECONDS", 0.2)

    attempted = []

    def _stuck_download(*args, **kwargs):
        attempted.append(args)
        time.sleep(10)
        return True

    monkeypatch.setattr("nltk.download", _stuck_download)

    start = time.time()
    globals_helper._download_nltk_data()
    elapsed = time.time() - start

    assert attempted, "nltk.download should have been attempted"
    assert elapsed < 5, f"download blocked for {elapsed:.1f}s instead of timing out"
    assert "timed out" in capsys.readouterr().out


def test_clear_stale_nltk_lock(monkeypatch, tmp_path) -> None:
    """Old locks are removed; fresh locks (an active download) are kept."""
    helper = GlobalsHelper()
    monkeypatch.setattr(helper, "_nltk_data_dir", str(tmp_path))

    lock = tmp_path / "tokenizers" / "punkt_tab.zip.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_bytes(b"")

    old = time.time() - (utils_module._NLTK_LOCK_STALE_SECONDS + 10)
    os.utime(lock, (old, old))

    helper._clear_stale_nltk_lock("punkt_tab")
    assert not lock.exists()

    lock.write_bytes(b"")
    helper._clear_stale_nltk_lock("punkt_tab")
    assert lock.exists(), "a fresh lock must be preserved"


def test_download_nltk_data_skips_present_packages(monkeypatch, tmp_path) -> None:
    """No download is attempted when the cache already has the data."""
    helper = GlobalsHelper()
    monkeypatch.setattr(helper, "_nltk_data_dir", str(tmp_path))
    monkeypatch.setattr("nltk.data.find", lambda *args, **kwargs: str(tmp_path))

    def _fail_download(*args, **kwargs):
        raise AssertionError("download should not be called when data is present")

    monkeypatch.setattr("nltk.download", _fail_download)
    helper._download_nltk_data()
