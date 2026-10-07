"""EpubReader returns a book's text in its reading order."""

from pathlib import Path

from ebooklib import epub

from llama_index.readers.file import EpubReader


def _write_book(path: Path, manifest: list[str], spine: list[str]) -> None:
    """
    Write an EPUB whose manifest lists `manifest` and whose spine lists `spine`.

    A name ending in `.svg` becomes an SVG image instead of an XHTML document.
    """
    book = epub.EpubBook()
    book.set_identifier("reading-order")
    book.set_title("Reading order")
    book.set_language("en")
    items = {}
    for name in manifest:
        if name.endswith(".svg"):
            item = epub.EpubItem(
                uid=name.replace(".", "-"),
                file_name=name,
                media_type="image/svg+xml",
                content=(
                    '<svg xmlns="http://www.w3.org/2000/svg">'
                    f"<text>{name} text.</text></svg>"
                ).encode(),
            )
        else:
            item = epub.EpubHtml(title=name, file_name=f"{name}.xhtml", lang="en")
            item.content = f"<html><body><p>{name} text.</p></body></html>"
        book.add_item(item)
        items[name] = item
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = [items[name] for name in spine]
    epub.write_epub(str(path), book)


def _document_lines(path: Path) -> list[str]:
    text = EpubReader().load_data(path)[0].text
    return [line for line in text.splitlines() if line.endswith(" text.")]


def test_epub_reader_follows_the_spine_not_the_manifest(tmp_path: Path) -> None:
    path = tmp_path / "book.epub"
    _write_book(
        path,
        manifest=["chapter-2", "titlepage", "chapter-1", "cover.svg"],
        # -- an image in the spine is not a document and is not read, as before --
        spine=["cover.svg", "titlepage", "chapter-1", "chapter-2"],
    )

    assert _document_lines(path) == [
        "titlepage text.",
        "chapter-1 text.",
        "chapter-2 text.",
    ]


def test_epub_reader_keeps_documents_the_spine_leaves_out(tmp_path: Path) -> None:
    path = tmp_path / "book.epub"
    _write_book(
        path,
        manifest=["notes", "chapter-2", "chapter-1"],
        # -- a spine that lists a document twice still yields its text once --
        spine=["chapter-1", "chapter-2", "chapter-1"],
    )

    assert _document_lines(path) == [
        "chapter-1 text.",
        "chapter-2 text.",
        "notes text.",
    ]
