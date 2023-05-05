"""Simple reader that reads files of different formats from a directory."""
import logging
from copy import deepcopy
from pathlib import Path
from typing import Callable, Dict, Generator, List, Optional, Union, cast

from llama_index.readers.base import BaseReader
from llama_index.readers.file.base_parser import BaseParser, ImageParserOutput
from llama_index.readers.file.docs_parser import DocxParser, PDFParser
from llama_index.readers.file.epub_parser import EpubParser
from llama_index.readers.file.image_parser import ImageParser
from llama_index.readers.file.markdown_parser import MarkdownParser
from llama_index.readers.file.mbox_parser import MboxParser
from llama_index.readers.file.slides_parser import PptxParser
from llama_index.readers.file.tabular_parser import PandasCSVParser
from llama_index.readers.file.video_audio import VideoAudioParser
from llama_index.readers.file.ipynb_parser import IPYNBParser
from llama_index.readers.schema.base import Document, ImageDocument

DEFAULT_FILE_EXTRACTOR: Dict[str, BaseParser] = {
    ".pdf": PDFParser(),
    ".docx": DocxParser(),
    ".pptx": PptxParser(),
    ".jpg": ImageParser(),
    ".png": ImageParser(),
    ".jpeg": ImageParser(),
    ".mp3": VideoAudioParser(),
    ".mp4": VideoAudioParser(),
    ".csv": PandasCSVParser(),
    ".epub": EpubParser(),
    ".md": MarkdownParser(),
    ".mbox": MboxParser(),
    ".ipynb": IPYNBParser(),
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
        file_extractor (Optional[Dict[str, BaseParser]]): A mapping of file
            extension to a BaseParser class that specifies how to convert that file
            to text. See DEFAULT_FILE_EXTRACTOR.
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
        file_extractor: Optional[Dict[str, BaseParser]] = None,
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

        self.file_extractor = file_extractor or DEFAULT_FILE_EXTRACTOR
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

    def load_data(self, concatenate: bool = False) -> List[Document]:
        """Load data from the input directory.

        Args:
            concatenate (bool): whether to concatenate all text docs into a single doc.
                If set to True, file metadata is ignored. False by default.
                This setting does not apply to image docs (always one doc per image).

        Returns:
            List[Document]: A list of documents.

        """
        # TODO: refactor parser output interface
        data: Union[str, List[str], ImageParserOutput] = ""
        data_list: List[str] = []
        metadata_list: List[Optional[dict]] = []
        image_docs: List[ImageDocument] = []
        for input_file in self.input_files:
            if input_file.suffix.lower() in self.file_extractor:
                parser = self.file_extractor[input_file.suffix]
                if not parser.parser_config_set:
                    parser.init_parser()
                data = parser.parse_file(input_file, errors=self.errors)
            else:
                # do standard read
                with open(input_file, "r", errors=self.errors, encoding="utf8") as f:
                    data = f.read()

            metadata: Optional[dict] = None
            if self.file_metadata is not None:
                metadata = self.file_metadata(str(input_file))

            if isinstance(data, ImageParserOutput):
                # process image
                image_docs.append(
                    ImageDocument(text=data.text, extra_info=metadata, image=data.image)
                )
            elif isinstance(data, List):
                # process list of str
                data_list.extend(data)
                repeated_metadata: List[Optional[dict]] = [
                    deepcopy(metadata) for _ in range(len(data))
                ]
                metadata_list.extend(repeated_metadata)
            else:
                # process single str
                data_list.append(str(data))
                metadata_list.append(metadata)

        if concatenate:
            text_docs = [Document("\n".join(data_list))]
        elif self.file_metadata is not None:
            text_docs = [
                Document(d, extra_info=m) for d, m in zip(data_list, metadata_list)
            ]
        else:
            text_docs = [Document(d) for d in data_list]

        return text_docs + cast(List[Document], image_docs)
