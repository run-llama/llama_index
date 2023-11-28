"""Simple reader that reads files of different formats from a directory."""
import logging
import mimetypes
import os
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Generator, List, Optional, Type

from tqdm import tqdm

from llama_index.readers.base import BaseReader
from llama_index.readers.file.docs_reader import DocxReader, HWPReader, PDFReader
from llama_index.readers.file.epub_reader import EpubReader
from llama_index.readers.file.image_reader import ImageReader
from llama_index.readers.file.ipynb_reader import IPYNBReader
from llama_index.readers.file.markdown_reader import MarkdownReader
from llama_index.readers.file.mbox_reader import MboxReader
from llama_index.readers.file.slides_reader import PptxReader
from llama_index.readers.file.tabular_reader import PandasCSVReader
from llama_index.readers.file.video_audio_reader import VideoAudioReader
from llama_index.schema import Document

DEFAULT_FILE_READER_CLS: Dict[str, Type[BaseReader]] = {
    ".hwp": HWPReader,
    ".pdf": PDFReader,
    ".docx": DocxReader,
    ".pptx": PptxReader,
    ".ppt": PptxReader,
    ".pptm": PptxReader,
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


def default_file_metadata_func(file_path: str) -> Dict:
    """Get some handy metadate from filesystem.

    Args:
        file_path: str: file path in str
    """
    return {
        "file_path": file_path,
        "file_name": os.path.basename(file_path),
        "file_type": mimetypes.guess_type(file_path)[0],
        "file_size": os.path.getsize(file_path),
        "creation_date": datetime.fromtimestamp(
            Path(file_path).stat().st_ctime
        ).strftime("%Y-%m-%d"),
        "last_modified_date": datetime.fromtimestamp(
            Path(file_path).stat().st_mtime
        ).strftime("%Y-%m-%d"),
        "last_accessed_date": datetime.fromtimestamp(
            Path(file_path).stat().st_atime
        ).strftime("%Y-%m-%d"),
    }


logger = logging.getLogger(__name__)


class SimpleDirectoryReader(BaseReader):
    """Simple directory reader.

    Load files from file directory.
    Automatically select the best file reader given file extensions.

    Args:
        input_dir (str): Path to the directory.
        input_files (List): List of file paths to read
            (Optional; overrides input_dir, exclude)
        exclude (List): glob of python file paths to exclude (Optional)
        exclude_hidden (bool): Whether to exclude hidden files (dotfiles).
        encoding (str): Encoding of the files.
            Default is utf-8.
        errors (str): how encoding and decoding errors are to be handled,
              see https://docs.python.org/3/library/functions.html#open
        recursive (bool): Whether to recursively search in subdirectories.
            False by default.
        filename_as_id (bool): Whether to use the filename as the document id.
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
        encoding: str = "utf-8",
        filename_as_id: bool = False,
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
        self.encoding = encoding

        self.exclude = exclude
        self.recursive = recursive
        self.exclude_hidden = exclude_hidden
        self.required_exts = required_exts
        self.num_files_limit = num_files_limit

        if input_files:
            self.input_files = []
            for path in input_files:
                if not os.path.isfile(path):
                    raise ValueError(f"File {path} does not exist.")
                input_file = Path(path)
                self.input_files.append(input_file)
        elif input_dir:
            if not os.path.isdir(input_dir):
                raise ValueError(f"Directory {input_dir} does not exist.")
            self.input_dir = Path(input_dir)
            self.exclude = exclude
            self.input_files = self._add_files(self.input_dir)

        if file_extractor is not None:
            self.file_extractor = file_extractor
        else:
            self.file_extractor = {}

        self.supported_suffix = list(DEFAULT_FILE_READER_CLS.keys())
        self.file_metadata = file_metadata or default_file_metadata_func
        self.filename_as_id = filename_as_id

    def is_hidden(self, path: Path) -> bool:
        return any(
            part.startswith(".") and part not in [".", ".."] for part in path.parts
        )

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
            skip_because_hidden = self.exclude_hidden and self.is_hidden(ref)
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

        new_input_files = sorted(all_files)

        if len(new_input_files) == 0:
            raise ValueError(f"No files found in {input_dir}.")

        if self.num_files_limit is not None and self.num_files_limit > 0:
            new_input_files = new_input_files[0 : self.num_files_limit]

        # print total number of files added
        logger.debug(
            f"> [SimpleDirectoryReader] Total files added: {len(new_input_files)}"
        )

        return new_input_files

    def load_data(self, show_progress: bool = False) -> List[Document]:
        """Load data from the input directory.

        Args:
            show_progress (bool): Whether to show tqdm progress bars. Defaults to False.

        Returns:
            List[Document]: A list of documents.
        """
        documents = []

        files_to_process = self.input_files

        if show_progress:
            files_to_process = tqdm(self.input_files, desc="Loading files", unit="file")

        for input_file in files_to_process:
            metadata: Optional[dict] = None
            if self.file_metadata is not None:
                metadata = self.file_metadata(str(input_file))

            file_suffix = input_file.suffix.lower()
            if (
                file_suffix in self.supported_suffix
                or file_suffix in self.file_extractor
            ):
                # use file readers
                if file_suffix not in self.file_extractor:
                    # instantiate file reader if not already
                    reader_cls = DEFAULT_FILE_READER_CLS[file_suffix]
                    self.file_extractor[file_suffix] = reader_cls()
                reader = self.file_extractor[file_suffix]

                # load data -- catch all errors except for ImportError
                try:
                    docs = reader.load_data(input_file, extra_info=metadata)
                except ImportError as e:
                    # ensure that ImportError is raised so user knows
                    # about missing dependencies
                    raise ImportError(str(e))
                except Exception as e:
                    # otherwise, just skip the file and report the error
                    print(
                        f"Failed to load file {input_file} with error: {e}. Skipping...",
                        flush=True,
                    )
                    continue

                # iterate over docs if needed
                if self.filename_as_id:
                    for i, doc in enumerate(docs):
                        doc.id_ = f"{input_file!s}_part_{i}"

                documents.extend(docs)
            else:
                # do standard read
                with open(input_file, errors=self.errors, encoding=self.encoding) as f:
                    data = f.read()

                doc = Document(text=data, metadata=metadata or {})
                if self.filename_as_id:
                    doc.id_ = str(input_file)

                documents.append(doc)

        for doc in documents:
            # Keep only metadata['file_path'] in both embedding and llm content
            # str, which contain extreme important context that about the chunks.
            # Dates is provided for convenience of postprocessor such as
            # TimeWeightedPostprocessor, but excluded for embedding and LLMprompts
            doc.excluded_embed_metadata_keys.extend(
                [
                    "file_name",
                    "file_type",
                    "file_size",
                    "creation_date",
                    "last_modified_date",
                    "last_accessed_date",
                ]
            )
            doc.excluded_llm_metadata_keys.extend(
                [
                    "file_name",
                    "file_type",
                    "file_size",
                    "creation_date",
                    "last_modified_date",
                    "last_accessed_date",
                ]
            )

        return documents
