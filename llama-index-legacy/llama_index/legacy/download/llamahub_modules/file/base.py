"""Simple reader that reads files of different formats from a directory."""

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

from llama_index.readers.base import BaseReader
from llama_index.readers.download import download_loader
from llama_index.readers.schema.base import Document

DEFAULT_FILE_EXTRACTOR: Dict[str, str] = {
    ".hwp": "HWPReader",
    ".pdf": "PDFReader",
    ".docx": "DocxReader",
    ".pptx": "PptxReader",
    ".jpg": "ImageReader",
    ".png": "ImageReader",
    ".jpeg": "ImageReader",
    ".mp3": "AudioTranscriber",
    ".mp4": "AudioTranscriber",
    ".csv": "PagedCSVReader",
    ".epub": "EpubReader",
    ".md": "MarkdownReader",
    ".mbox": "MboxReader",
    ".eml": "UnstructuredReader",
    ".html": "UnstructuredReader",
    ".json": "JSONReader",
}


class SimpleDirectoryReader(BaseReader):
    """Simple directory reader.

    Can read files into separate documents, or concatenates
    files into one document text.

    Args:
        input_dir (str): Path to the directory.
        exclude_hidden (bool): Whether to exclude hidden files (dotfiles).
        errors (str): how encoding and decoding errors are to be handled,
              see https://docs.python.org/3/library/functions.html#open
        recursive (bool): Whether to recursively search in subdirectories.
            False by default.
        required_exts (Optional[List[str]]): List of required extensions.
            Default is None.
        file_extractor (Optional[Dict[str, BaseReader]]): A mapping of file
            extension to a BaseReader class that specifies how to convert that file
            to text. See DEFAULT_FILE_EXTRACTOR.
        num_files_limit (Optional[int]): Maximum number of files to read.
            Default is None.
        file_metadata (Optional[Callable[str, Dict]]): A function that takes
            in a filename and returns a Dict of metadata for the Document.
            Default is None.
    """

    def __init__(
        self,
        input_dir: str,
        exclude_hidden: bool = True,
        errors: str = "ignore",
        recursive: bool = False,
        required_exts: Optional[List[str]] = None,
        file_extractor: Optional[Dict[str, Union[str, BaseReader]]] = None,
        num_files_limit: Optional[int] = None,
        file_metadata: Optional[Callable[[str], Dict]] = None,
    ) -> None:
        """Initialize with parameters."""
        super().__init__()
        self.input_dir = Path(input_dir)
        self.errors = errors

        self.recursive = recursive
        self.exclude_hidden = exclude_hidden
        self.required_exts = required_exts
        self.num_files_limit = num_files_limit

        self.input_files = self._add_files(self.input_dir)
        self.file_extractor = file_extractor or DEFAULT_FILE_EXTRACTOR
        self.file_metadata = file_metadata

    def _add_files(self, input_dir: Path) -> List[Path]:
        """Add files."""
        input_files = sorted(input_dir.iterdir())
        new_input_files = []
        dirs_to_explore = []
        for input_file in input_files:
            if self.exclude_hidden and input_file.stem.startswith("."):
                continue
            elif input_file.is_dir():
                if self.recursive:
                    dirs_to_explore.append(input_file)
            elif (
                self.required_exts is not None
                and input_file.suffix not in self.required_exts
            ):
                continue
            else:
                new_input_files.append(input_file)

        for dir_to_explore in dirs_to_explore:
            sub_input_files = self._add_files(dir_to_explore)
            new_input_files.extend(sub_input_files)

        if self.num_files_limit is not None and self.num_files_limit > 0:
            new_input_files = new_input_files[0 : self.num_files_limit]

        # print total number of files added
        logging.debug(
            f"> [SimpleDirectoryReader] Total files added: {len(new_input_files)}"
        )

        return new_input_files

    def load_data(self) -> List[Document]:
        """Load data from the input directory.

        Args:
            concatenate (bool): whether to concatenate all files into one document.
                If set to True, file metadata is ignored.
                False by default.

        Returns:
            List[Document]: A list of documents.

        """

        documents = []
        for input_file in self.input_files:
            metadata = None
            if self.file_metadata is not None:
                metadata = self.file_metadata(str(input_file))

            if input_file.suffix in self.file_extractor:
                reader = self.file_extractor[input_file.suffix]

                if isinstance(reader, str):
                    try:
                        from llama_hub.utils import import_loader

                        reader = import_loader(reader)()
                    except ImportError:
                        reader = download_loader(reader)()

                extracted_documents = reader.load_data(
                    file=input_file, extra_info=metadata
                )
                documents.extend(extracted_documents)
            else:
                data = ""
                # do standard read
                with open(input_file, "r", errors=self.errors) as f:
                    data = f.read()
                document = Document(text=data, extra_info=metadata or {})
                documents.append(document)

        return documents
