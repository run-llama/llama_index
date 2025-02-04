from pathlib import Path
from typing import List

from llama_index.core.readers.base import BaseReader
from llama_index.readers.obsidian import ObsidianReader


def test_class():
    names_of_base_classes = [b.__name__ for b in ObsidianReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes


# Helper function to create a markdown file in the given directory.
def create_markdown_file(directory: Path, file_name: str, content: str) -> Path:
    file_path = directory / file_name
    file_path.write_text(content, encoding="utf-8")
    return file_path


###########################################
# File Metadata (file_name & folder_path)
###########################################


def test_file_metadata(tmp_path: Path):
    """
    Test that a simple markdown file returns a document with correct file metadata.
    """
    content = "This is a simple document."
    create_markdown_file(tmp_path, "test.md", content)

    reader = ObsidianReader(input_dir=str(tmp_path))
    docs = reader.load_data()
    assert len(docs) == 1
    doc = docs[0]
    # Check that file metadata was added
    assert doc.metadata.get("file_name") == "test.md"
    # folder_path should match the temporary directory
    assert Path(doc.metadata.get("folder_path")).resolve() == tmp_path.resolve()


def test_file_metadata_nested(tmp_path: Path):
    """
    Test that a markdown file in a subdirectory gets the correct folder path metadata.
    """
    subdir = tmp_path / "subfolder"
    subdir.mkdir()
    content = "Nested file content."
    create_markdown_file(subdir, "nested.md", content)

    reader = ObsidianReader(input_dir=str(tmp_path))
    docs = reader.load_data()
    # Expect one document loaded from the nested directory
    assert len(docs) == 1
    doc = docs[0]
    assert doc.metadata.get("file_name") == "nested.md"
    assert Path(doc.metadata.get("folder_path")).resolve() == subdir.resolve()


###########################################
# Wikilink Extraction
###########################################


def test_wikilink_extraction(tmp_path: Path):
    """
    Test that wikilinks (including alias links) are extracted and stored in metadata.
    """
    content = "Refer to [[NoteOne]] and [[NoteTwo|Alias]] for more details."
    create_markdown_file(tmp_path, "wikilinks.md", content)

    reader = ObsidianReader(input_dir=str(tmp_path))
    docs = reader.load_data()
    assert len(docs) == 1
    doc = docs[0]
    wikilinks: List[str] = doc.metadata.get("wikilinks", [])
    # Order does not matter; both targets should be present.
    assert set(wikilinks) == {"NoteOne", "NoteTwo"}


def test_wikilink_extraction_duplicates(tmp_path: Path):
    """
    Test that duplicate wikilinks (with or without aliases) are only stored once.
    """
    content = "See [[Note]] and also [[Note|Alias]]."
    create_markdown_file(tmp_path, "dup_wikilinks.md", content)

    reader = ObsidianReader(input_dir=str(tmp_path))
    docs = reader.load_data()
    assert len(docs) == 1
    doc = docs[0]
    wikilinks: List[str] = doc.metadata.get("wikilinks", [])
    # Only one unique wikilink should be present.
    assert set(wikilinks) == {"Note"}


###########################################
# Tasks Extraction (without removal)
###########################################


def test_tasks_extraction(tmp_path: Path):
    """
    Test that markdown tasks are correctly extracted into metadata when removal is disabled.
    """
    content = (
        "Task list:\n"
        "- [ ] Task A\n"
        "Some intervening text\n"
        "- [x] Task B\n"
        "More text follows."
    )
    create_markdown_file(tmp_path, "tasks.md", content)

    reader = ObsidianReader(
        input_dir=str(tmp_path), extract_tasks=True, remove_tasks_from_text=False
    )
    docs = reader.load_data()
    assert len(docs) == 1
    doc = docs[0]
    tasks: List[str] = doc.metadata.get("tasks", [])
    # Check that tasks are extracted.
    assert "Task A" in tasks
    assert "Task B" in tasks
    # Since removal is disabled, the original text should still contain the task lines.
    assert "- [ ] Task A" in doc.text
    assert "- [x] Task B" in doc.text


###########################################
# Tasks Removal from Text
###########################################


def test_remove_tasks_from_text(tmp_path: Path):
    """
    Test that when removal is enabled, task lines are removed from the document text.
    """
    content = "Intro text\n" "- [ ] Task 1\n" "- [x] Task 2\n" "Conclusion text"
    create_markdown_file(tmp_path, "tasks_removed.md", content)

    reader = ObsidianReader(
        input_dir=str(tmp_path), extract_tasks=True, remove_tasks_from_text=True
    )
    docs = reader.load_data()
    assert len(docs) == 1
    doc = docs[0]
    tasks: List[str] = doc.metadata.get("tasks", [])
    assert "Task 1" in tasks
    assert "Task 2" in tasks
    # Ensure the task lines have been removed from the main text.
    assert "- [ ] Task 1" not in doc.text
    assert "- [x] Task 2" not in doc.text
    # Ensure that non-task text is still present.
    assert "Intro text" in doc.text
    assert "Conclusion text" in doc.text


###########################################
# Backlink Extraction
###########################################


def get_doc_by_note_name(docs, note_name: str):
    """
    Utility function to return the first document with the specified note name.
    """
    for doc in docs:
        if doc.metadata.get("note_name") == note_name:
            return doc
    return None


def test_single_backlink(tmp_path: Path):
    """
    Test a simple case where one note (A.md) links to another (B.md).

    Expected behavior:
      - Note A should have no backlinks.
      - Note B should have a backlink from A.
    """
    # Create two markdown files:
    # A.md links to B, while B.md contains no wikilinks.
    create_markdown_file(tmp_path, "A.md", "This is note A linking to [[B]].")
    create_markdown_file(tmp_path, "B.md", "This is note B with no links.")

    reader = ObsidianReader(input_dir=str(tmp_path))
    docs = reader.load_data()

    doc_a = get_doc_by_note_name(docs, "A")
    doc_b = get_doc_by_note_name(docs, "B")

    # Verify that doc_a exists and has no backlinks.
    assert doc_a is not None
    assert doc_a.metadata.get("backlinks") == []

    # Verify that doc_b exists and has a backlink from A.
    assert doc_b is not None
    assert doc_b.metadata.get("backlinks") == ["A"]


def test_multiple_backlinks(tmp_path: Path):
    """
    Test a scenario with multiple notes linking to a single note.

    Create three files:
      - A.md: links to B and C.
      - B.md: links to C.
      - C.md: contains no wikilinks.

    Expected behavior:
      - Note A should have no backlinks.
      - Note B should have a backlink from A.
      - Note C should have backlinks from both A and B.
    """
    create_markdown_file(tmp_path, "A.md", "Linking to [[B]] and [[C]].")
    create_markdown_file(tmp_path, "B.md", "Linking to [[C]].")
    create_markdown_file(tmp_path, "C.md", "No links here.")

    reader = ObsidianReader(input_dir=str(tmp_path))
    docs = reader.load_data()

    doc_a = get_doc_by_note_name(docs, "A")
    doc_b = get_doc_by_note_name(docs, "B")
    doc_c = get_doc_by_note_name(docs, "C")

    # Verify note A has no backlinks.
    assert doc_a is not None
    assert doc_a.metadata.get("backlinks") == []

    # Verify note B has a backlink from A.
    assert doc_b is not None
    assert doc_b.metadata.get("backlinks") == ["A"]

    # Note C should have backlinks from A and B.
    # Since file processing order might vary, we compare as sets.
    assert doc_c is not None
    backlinks_c = doc_c.metadata.get("backlinks")
    assert set(backlinks_c) == {"A", "B"}


def test_no_links(tmp_path: Path):
    """
    Test that a note with no outgoing links gets an empty backlinks list.
    """
    create_markdown_file(tmp_path, "A.md", "This is a note with no links.")
    reader = ObsidianReader(input_dir=str(tmp_path))
    docs = reader.load_data()

    doc_a = get_doc_by_note_name(docs, "A")
    assert doc_a is not None
    # Since no note links to A, its backlinks should be an empty list.
    assert doc_a.metadata.get("backlinks") == []
