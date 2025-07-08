"""
Obsidian reader class.

Pass in the path to an Obsidian vault and it will parse all markdown
files into a List of Documents. Documents are split by header in
the Markdown Reader we use.

Each document will contain the following metadata:
- file_name: the name of the markdown file
- folder_path: the full path to the folder containing the file
- folder_name: the relative path to the folder containing the file
- note_name: the name of the note (without the .md extension)
- wikilinks: a list of all wikilinks found in the document
- backlinks: a list of all notes that link to this note

Optionally, tasks can be extracted from the text and stored in metadata.

"""

import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Tuple

if TYPE_CHECKING:
    from langchain.docstore.document import Document as LCDocument

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from llama_index.readers.file import MarkdownReader


def is_hardlink(filepath: Path) -> bool:
    """
    Check if a file is a hardlink by checking the number of links to/from it.

    Args:
        filepath (Path): path to the file.

    """
    stat_info = os.stat(filepath)
    return stat_info.st_nlink > 1


class ObsidianReader(BaseReader):
    """
    input_dir (str): Path to the Obsidian vault.
    extract_tasks (bool): If True, extract tasks from the text and store them in metadata.
                            Default is False.
    remove_tasks_from_text (bool): If True and extract_tasks is True, remove the task
                                    lines from the main document text.
                                    Default is False.
    """

    def __init__(
        self,
        input_dir: str,
        extract_tasks: bool = False,
        remove_tasks_from_text: bool = False,
    ):
        self.input_dir = Path(input_dir)
        self.extract_tasks = extract_tasks
        self.remove_tasks_from_text = remove_tasks_from_text

    def load_data(self, *args: Any, **load_kwargs: Any) -> List[Document]:
        """
        Walks through the vault, loads each markdown file, and adds extra metadata
        (file name, folder path, wikilinks, backlinks, and optionally tasks).
        """
        docs: List[Document] = []
        # This map will hold: {target_note: [linking_note1, linking_note2, ...]}
        backlinks_map = {}
        input_dir_abs = self.input_dir.resolve()

        for dirpath, dirnames, filenames in os.walk(self.input_dir, followlinks=False):
            # Skip hidden directories.
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            for filename in filenames:
                if filename.endswith(".md"):
                    filepath = os.path.join(dirpath, filename)
                    file_path_obj = Path(filepath).resolve()
                    try:
                        if is_hardlink(filepath=file_path_obj):
                            print(
                                f"Warning: Skipping file because it is a hardlink (potential malicious exploit): {filepath}"
                            )
                            continue
                        if not str(file_path_obj).startswith(str(input_dir_abs)):
                            print(
                                f"Warning: Skipping file outside input directory: {filepath}"
                            )
                            continue
                        md_docs = MarkdownReader().load_data(Path(filepath))
                        for i, doc in enumerate(md_docs):
                            file_path_obj = Path(filepath)
                            note_name = file_path_obj.stem
                            doc.metadata["file_name"] = file_path_obj.name
                            doc.metadata["folder_path"] = str(file_path_obj.parent)
                            try:
                                folder_name = str(
                                    file_path_obj.parent.relative_to(input_dir_abs)
                                )
                            except ValueError:
                                # Fallback if relative_to fails (should not happen)
                                folder_name = str(file_path_obj.parent)
                            doc.metadata["folder_name"] = folder_name
                            doc.metadata["note_name"] = note_name
                            wikilinks = self._extract_wikilinks(doc.text)
                            doc.metadata["wikilinks"] = wikilinks
                            # For each wikilink found in this document, record a backlink from this note.
                            for link in wikilinks:
                                # Each link is expected to match a note name (without .md)
                                backlinks_map.setdefault(link, []).append(note_name)

                            # Optionally, extract tasks from the text.
                            if self.extract_tasks:
                                tasks, cleaned_text = self._extract_tasks(doc.text)
                                doc.metadata["tasks"] = tasks
                                if self.remove_tasks_from_text:
                                    md_docs[i] = Document(
                                        text=cleaned_text, metadata=doc.metadata
                                    )
                        docs.extend(md_docs)
                    except Exception as e:
                        print(f"Error processing file {filepath}: {e!s}")
                        continue

        # Now that we have processed all files, assign backlinks metadata.
        for doc in docs:
            note_name = doc.metadata.get("note_name")
            # If no backlinks exist for this note, default to an empty list.
            doc.metadata["backlinks"] = backlinks_map.get(note_name, [])
        return docs

    def load_langchain_documents(self, **load_kwargs: Any) -> List["LCDocument"]:
        """
        Loads data in the LangChain document format.
        """
        docs = self.load_data(**load_kwargs)
        return [d.to_langchain_format() for d in docs]

    def _extract_wikilinks(self, text: str) -> List[str]:
        """
        Extracts Obsidian wikilinks from the given text.

        Matches patterns like:
          - [[Note Name]]
          - [[Note Name|Alias]]

        Returns a list of unique wikilink targets (aliases are ignored).
        """
        pattern = r"\[\[([^\]]+)\]\]"
        matches = re.findall(pattern, text)
        links = []
        for match in matches:
            # If a pipe is present (e.g. [[Note|Alias]]), take only the part before it.
            target = match.split("|")[0].strip()
            links.append(target)
        return list(set(links))

    def _extract_tasks(self, text: str) -> Tuple[List[str], str]:
        """
        Extracts markdown tasks from the text.

        A task is expected to be a checklist item in markdown, for example:
            - [ ] Do something
            - [x] Completed task

        Returns a tuple:
            (list of task strings, text with task lines removed if removal is enabled).
        """
        # This regex matches lines starting with '-' or '*' followed by a checkbox.
        task_pattern = re.compile(
            r"^\s*[-*]\s*\[\s*(?:x|X| )\s*\]\s*(.*)$", re.MULTILINE
        )
        tasks = task_pattern.findall(text)
        cleaned_text = text
        if self.remove_tasks_from_text:
            cleaned_text = task_pattern.sub("", text)
        return tasks, cleaned_text
