"""Simple reader that reads files of different formats from a directory."""
import logging
from pathlib import Path
from typing import Callable, Dict, Generator, List, Optional, Type

from llama_index.readers.base import BaseReader
from llama_index.readers.file.docs_reader import DocxReader, PDFReader
from llama_index.readers.file.epub_reader import EpubReader
from llama_index.readers.file.image_reader import ImageReader
from llama_index.readers.file.ipynb_reader import IPYNBReader
from llama_index.readers.file.markdown_reader import MarkdownReader
from llama_index.readers.file.mbox_reader import MboxReader
from llama_index.readers.file.slides_reader import PptxReader
from llama_index.readers.file.tabular_reader import PandasCSVReader
from llama_index.readers.file.video_audio_reader import VideoAudioReader
from llama_index.readers.schema.base import Document

DEFAULT_FILE_READER_CLS: Dict[str, Type[BaseReader]] = {
    ".pdf": PDFReader,
    ".docx": DocxReader,
    ".pptx": PptxReader,
    ".jpg": ImageReader,
    ".png": ImageReader,
    ".jpeg": ImageReader,
    ".mp3": VideoAudioReader,
    ".mp4": VideoAudioReader,
    ".csv": PandasCSVReader,
    ".epub": EpubReader,
    ".md": MarkdownReader,
    ".mbox": MboxReader,
    ".ipynb": IPYNBReader,
}

logger = logging.getLogger(__name__)


class SimpleDirectoryReader(BaseReader):
    """Simple directory reader.

    Can read files into separate documents, or concatenates
    files into one document text.

    Args:
        input_dir (str): Path to the directory.
        input_files (List): List of file paths to read
            (Optional; overrides input_dir, exclude)
        exclude (List): glob of python file paths to exclude (Optional)
        exclude_hidden (bool): Whether to exclude hidden files (dotfiles).
        errors (str): how encoding and decoding errors are to be handled,
              see https://docs.python.org/3/library/functions.html#open
        recursive (bool): Whether to recursively search in subdirectories.
            False by default.
        required_exts (Optional[List[str]]): List of required extensions.
            Default is None.
        file_extractor (Optional[Dict[str, BaseReader]]): A mapping of file
            extension to a BaseReader class that specifies how to convert that file
            to text. If not specified, use default from DEFAULT_FILE_READER_CLS.
        num_files_limit (Optional[int]): Maximum number of files to read.
            Default is None.
        file_metadata (Optional[Callable[str, Dict]]): A function that takes
            in a filename and returns a Dict of metadata for the Document.
            Default is None.
    """

    def __init__(
        self,
        input_dir: Optional[str] = None,
        input_files: Optional[List] = None,
        exclude: Optional[List] = None,
        exclude_hidden: bool = True,
        errors: str = "ignore",
        recursive: bool = False,
        required_exts: Optional[List[str]] = None,
        file_extractor: Optional[Dict[str, BaseReader]] = None,
        num_files_limit: Optional[int] = None,
        file_metadata: Optional[Callable[[str], Dict]] = None,
    ) -> None:
        """Initialize with parameters."""
        super().__init__()

        if not input_dir and not input_files:
            raise ValueError("Must provide either `input_dir` or `input_files`.")

        self.errors = errors

        self.exclude = exclude
        self.recursive = recursive
        self.exclude_hidden = exclude_hidden
        self.required_exts = required_exts
        self.num_files_limit = num_files_limit

        if input_files:
            self.input_files = []
            for path in input_files:
                input_file = Path(path)
                self.input_files.append(input_file)
        elif input_dir:
            self.input_dir = Path(input_dir)
            self.exclude = exclude
            self.input_files = self._add_files(self.input_dir)

        if file_extractor is not None:
            self.file_extractor = file_extractor
        else:
            self.file_extractor = {}

        self.supported_suffix = list(DEFAULT_FILE_READER_CLS.keys())
        self.file_metadata = file_metadata

    def _add_files(self, input_dir: Path) -> List[Path]:
        """Add files."""
        all_files = set()
        rejected_files = set()

        if self.exclude is not None:
            for excluded_pattern in self.exclude:
                if self.recursive:
                    # Recursive glob
                    for file in input_dir.rglob(excluded_pattern):
                        rejected_files.add(Path(file))
                else:
                    # Non-recursive glob
                    for file in input_dir.glob(excluded_pattern):
                        rejected_files.add(Path(file))

        file_refs: Generator[Path, None, None]
        if self.recursive:
            file_refs = Path(input_dir).rglob("*")
        else:
            file_refs = Path(input_dir).glob("*")

        for ref in file_refs:
            # Manually check if file is hidden or directory instead of
            # in glob for backwards compatibility.
            is_dir = ref.is_dir()
            skip_because_hidden = self.exclude_hidden and ref.name.startswith(".")
            skip_because_bad_ext = (
                self.required_exts is not None and ref.suffix not in self.required_exts
            )
            skip_because_excluded = ref in rejected_files

            if (
                is_dir
                or skip_because_hidden
                or skip_because_bad_ext
                or skip_because_excluded
            ):
                continue
            else:
                all_files.add(ref)

        new_input_files = sorted(list(all_files))

        if self.num_files_limit is not None and self.num_files_limit > 0:
            new_input_files = new_input_files[0 : self.num_files_limit]

        # print total number of files added
        logger.debug(
            f"> [SimpleDirectoryReader] Total files added: {len(new_input_files)}"
        )

        return new_input_files

    def load_data(self) -> List[Document]:
        """Load data from the input directory.

        Args:
            concatenate (bool): whether to concatenate all text docs into a single doc.
                If set to True, file metadata is ignored. False by default.
                This setting does not apply to image docs (always one doc per image).

        Returns:
            List[Document]: A list of documents.

        """
        documents = []
        for input_file in self.input_files:
            metadata: Optional[dict] = None
            if self.file_metadata is not None:
                metadata = self.file_metadata(str(input_file))

            if input_file.suffix.lower() in self.supported_suffix:
                # use file readers
                if input_file.suffix not in self.file_extractor:
                    # instantiate file reader if not already
                    reader_cls = DEFAULT_FILE_READER_CLS[input_file.suffix]
                    self.file_extractor[input_file.suffix] = reader_cls()
                reader = self.file_extractor[input_file.suffix]
                docs = reader.load_data(input_file, extra_info=metadata)
                documents.extend(docs)
            else:
                # do standard read
                with open(input_file, "r", errors=self.errors, encoding="utf8") as f:
                    data = f.read()

                doc = Document(data, extra_info=metadata)
                documents.append(doc)

        return documents
