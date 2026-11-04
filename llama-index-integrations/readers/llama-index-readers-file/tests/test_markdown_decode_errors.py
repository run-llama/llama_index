from pathlib import Path

from fsspec.implementations.memory import MemoryFileSystem
import pytest

from llama_index.readers.file.markdown import MarkdownReader


@pytest.mark.parametrize("remote", [True, False])
@pytest.mark.parametrize(
    ("errors", "expected"),
    [("ignore", "beforeafter"), ("replace", "before\ufffdafter")],
)
def test_markdown_decode_error_policy(
    tmp_path: Path, remote: bool, errors: str, expected: str
) -> None:
    content = b"before\xffafter"
    reader = MarkdownReader()
    if remote:
        fs = MemoryFileSystem()
        fs.pipe("/invalid.md", content)
        result = reader.parse_tups("/invalid.md", errors=errors, fs=fs)
    else:
        file = tmp_path / "invalid.md"
        file.write_bytes(content)
        result = reader.parse_tups(str(file), errors=errors)

    assert result == [(None, expected)]


def test_markdown_default_decode_policy(tmp_path: Path) -> None:
    file = tmp_path / "invalid.md"
    file.write_bytes(b"before\xffafter")

    documents = MarkdownReader().load_data(str(file))

    assert [document.text for document in documents] == ["beforeafter"]


def test_markdown_strict_decode_policy(tmp_path: Path) -> None:
    file = tmp_path / "invalid.md"
    file.write_bytes(b"before\xffafter")

    with pytest.raises(UnicodeDecodeError):
        MarkdownReader().parse_tups(str(file), errors="strict")


def test_markdown_valid_utf8(tmp_path: Path) -> None:
    file = tmp_path / "valid.md"
    file.write_text("# Heading\n中文正文", encoding="utf-8")

    assert MarkdownReader().parse_tups(str(file)) == [("Heading", "中文正文")]
