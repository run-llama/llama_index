"""Simple reader that reads files of different formats from a directory."""
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

from gpt_index.readers.base import BaseReader
from gpt_index.readers.file.base_parser import BaseParser
from gpt_index.readers.file.docs_parser import DocxParser, PDFParser
from gpt_index.readers.file.epub_parser import EpubParser
from gpt_index.readers.file.image_parser import ImageParser
from gpt_index.readers.file.markdown_parser import MarkdownParser
from gpt_index.readers.file.mbox_parser import MboxParser
from gpt_index.readers.file.slides_parser import PptxParser
from gpt_index.readers.file.tabular_parser import PandasCSVParser
from gpt_index.readers.file.video_audio import VideoAudioParser
from gpt_index.readers.schema.base import Document

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
}


class SimpleDirectoryReader(BaseReader):
    """Simple directory reader.

    Can read files into separate documents, or concatenates
    files into one document text.

    Args:
        input_dir (str): Path to the directory.
        input_files (List): List of file paths to read (Optional; overrides input_dir)
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
            self.input_files = self._add_files(self.input_dir)

        self.file_extractor = file_extractor or DEFAULT_FILE_EXTRACTOR
        self.file_metadata = file_metadata

    def _add_files(self, input_dir: Path) -> List[Path]:
        """Add files."""
        input_files = sorted(input_dir.iterdir())
        new_input_files = []
        dirs_to_explore = []
        for input_file in input_files:
            if input_file.is_dir():
                if self.recursive:
                    dirs_to_explore.append(input_file)
            elif self.exclude_hidden and input_file.name.startswith("."):
                continue
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

    def load_data(self, concatenate: bool = False) -> List[Document]:
        """Load data from the input directory.

        Args:
            concatenate (bool): whether to concatenate all files into one document.
                If set to True, file metadata is ignored.
                False by default.

        Returns:
            List[Document]: A list of documents.

        """
        data: Union[str, List[str]] = ""
        data_list: List[str] = []
        metadata_list = []
        for input_file in self.input_files:
            if input_file.suffix in self.file_extractor:
                parser = self.file_extractor[input_file.suffix]
                if not parser.parser_config_set:
                    parser.init_parser()
                data = parser.parse_file(input_file, errors=self.errors)
            else:
                # do standard read
                with open(input_file, "r", errors=self.errors) as f:
                    data = f.read()
            if isinstance(data, List):
                data_list.extend(data)
            else:
                data_list.append(str(data))
            if self.file_metadata is not None:
                metadata_list.append(self.file_metadata(str(input_file)))

        if concatenate:
            return [Document("\n".join(data_list))]
        elif self.file_metadata is not None:
            return [Document(d, extra_info=m) for d, m in zip(data_list, metadata_list)]
        else:
            return [Document(d) for d in data_list]
