import os
import stat

import pytest
from llama_index.tools.google._token_file import write_token_file

pytestmark = pytest.mark.skipif(
    os.name == "nt", reason="POSIX permission bits are not enforced on Windows"
)


@pytest.fixture
def umask_022():
    previous = os.umask(0o022)
    yield
    os.umask(previous)


def test_new_token_file_is_owner_only(tmp_path, umask_022):
    path = tmp_path / "token.json"

    write_token_file(str(path), "new")

    assert path.read_text(encoding="utf-8") == "new"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_existing_world_readable_token_file_is_tightened(tmp_path, umask_022):
    path = tmp_path / "token.json"
    path.write_text("old token that is longer than the new one", encoding="utf-8")
    path.chmod(0o644)

    write_token_file(str(path), "new")

    assert path.read_text(encoding="utf-8") == "new"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_token_file_is_written_when_it_cannot_be_chmodded(
    tmp_path, umask_022, monkeypatch
):
    path = tmp_path / "token.json"
    path.write_text("old", encoding="utf-8")

    def deny(*_args):
        raise PermissionError("not the owner")

    monkeypatch.setattr(os, "fchmod", deny)

    write_token_file(str(path), "new")

    assert path.read_text(encoding="utf-8") == "new"
