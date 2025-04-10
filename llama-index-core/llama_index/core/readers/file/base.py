"""Simple reader that reads files of different formats from a directory."""

from __future__ import annotations

import asyncio
import logging
import mimetypes
import multiprocessing
import os
import warnings
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from functools import reduce
from itertools import repeat
from pathlib import Path, PurePosixPath
from typing import (
    Any,
    Callable,
    Generator,
    Type,
    cast,
)

import fsspec
from fsspec.implementations.local import LocalFileSystem
from tqdm import tqdm

from llama_index.core.async_utils import get_asyncio_module, run_jobs
from llama_index.core.readers.base import BaseReader, ResourcesReaderMixin
from llama_index.core.schema import Document


logger = logging.getLogger(__name__)


class FileSystemReaderMixin(ABC):
    @abstractmethod
    def read_file_content(self, input_file: Path, **kwargs: Any) -> bytes:
        """
        Read the bytes content of a file.

        Args:
            input_file (Path): Path to the file.

        Returns:
            bytes: File content.
        """

    async def aread_file_content(
        self, input_file: Path, **kwargs: Any
    ) -> bytes:  # pragma: no cover
        """
        A thin wrapper around read_file_content.

        Args:
            input_file (Path): Path to the file.

        Returns:
            bytes: File content.
        """
        return self.read_file_content(input_file, **kwargs)


def _try_loading_included_file_formats() -> (
    dict[str, Type[BaseReader]]
):  # pragma: no cover
    try:
        from llama_index.readers.file import (
            DocxReader,
            EpubReader,
            HWPReader,
            ImageReader,
            IPYNBReader,
            MboxReader,
            PandasCSVReader,
            PandasExcelReader,
            PDFReader,
            PptxReader,
            VideoAudioReader,
        )  # pants: no-infer-dep
    except ImportError:
        logger.warning(
            "`llama-index-readers-file` package not found, some file readers will not be available "
            "if not provided by the `file_extractor` parameter."
        )
        return {}

    default_file_reader_cls: dict[str, Type[BaseReader]] = {
        ".hwp": HWPReader,
        ".pdf": PDFReader,
        ".docx": DocxReader,
        ".pptx": PptxReader,
        ".ppt": PptxReader,
        ".pptm": PptxReader,
        ".gif": ImageReader,
        ".jpg": ImageReader,
        ".png": ImageReader,
        ".jpeg": ImageReader,
        ".webp": ImageReader,
        ".mp3": VideoAudioReader,
        ".mp4": VideoAudioReader,
        ".csv": PandasCSVReader,
        ".epub": EpubReader,
        ".mbox": MboxReader,
        ".ipynb": IPYNBReader,
        ".xls": PandasExcelReader,
        ".xlsx": PandasExcelReader,
    }
    return default_file_reader_cls


def _format_file_timestamp(
    timestamp: float | None, include_time: bool = False
) -> str | None:
    """
    Format file timestamp to a string.
    The format will be %Y-%m-%d if include_time is False or missing,
    %Y-%m-%dT%H:%M:%SZ if include_time is True.

    Args:
        timestamp (float): timestamp in float
        include_time (bool): whether to include time in the formatted string

    Returns:
        str: formatted timestamp
        None: if the timestamp passed was None
    """
    if timestamp is None:
        return None

    # Convert timestamp to UTC
    # Check if timestamp is already a datetime object
    if isinstance(timestamp, datetime):
        timestamp_dt = timestamp.astimezone(timezone.utc)
    else:
        timestamp_dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)

    if include_time:
        return timestamp_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return timestamp_dt.strftime("%Y-%m-%d")


def default_file_metadata_func(
    file_path: str, fs: fsspec.AbstractFileSystem | None = None
) -> dict:
    """
    Get some handy metadata from filesystem.

    Args:
        file_path: str: file path in str
    """
    fs = fs or get_default_fs()
    stat_result = fs.stat(file_path)

    try:
        file_name = os.path.basename(str(stat_result["name"]))
    except Exception as e:
        file_name = os.path.basename(file_path)

    creation_date = _format_file_timestamp(stat_result.get("created"))
    last_modified_date = _format_file_timestamp(stat_result.get("mtime"))
    last_accessed_date = _format_file_timestamp(stat_result.get("atime"))
    default_meta = {
        "file_path": file_path,
        "file_name": file_name,
        "file_type": mimetypes.guess_type(file_path)[0],
        "file_size": stat_result.get("size"),
        "creation_date": creation_date,
        "last_modified_date": last_modified_date,
        "last_accessed_date": last_accessed_date,
    }

    # Return not null value
    return {
        meta_key: meta_value
        for meta_key, meta_value in default_meta.items()
        if meta_value is not None
    }


class _DefaultFileMetadataFunc:
    """
    Default file metadata function wrapper which stores the fs.
    Allows for pickling of the function.
    """

    def __init__(self, fs: fsspec.AbstractFileSystem | None = None):
        self.fs = fs or get_default_fs()

    def __call__(self, file_path: str) -> dict:
        return default_file_metadata_func(file_path, self.fs)


def get_default_fs() -> fsspec.AbstractFileSystem:
    return LocalFileSystem()


def is_default_fs(fs: fsspec.AbstractFileSystem) -> bool:
    return isinstance(fs, LocalFileSystem) and not fs.auto_mkdir


class SimpleDirectoryReader(BaseReader, ResourcesReaderMixin, FileSystemReaderMixin):
    """
    Simple directory reader.

    Load files from file directory.
    Automatically select the best file reader given file extensions.

    Args:
        input_dir (Union[Path, str]): Path to the directory.
        input_files (List): List of file paths to read
            (Optional; overrides input_dir, exclude)
        exclude (List): glob of python file paths to exclude (Optional)
        exclude_hidden (bool): Whether to exclude hidden files (dotfiles).
        exclude_empty (bool): Whether to exclude empty files (Optional).
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
        raise_on_error (bool): Whether to raise an error if a file cannot be read.
        fs (Optional[fsspec.AbstractFileSystem]): File system to use. Defaults
        to using the local file system. Can be changed to use any remote file system
        exposed via the fsspec interface.
    """

    supported_suffix_fn: Callable = _try_loading_included_file_formats

    def __init__(
        self,
        input_dir: Path | str | None = None,
        input_files: list | None = None,
        exclude: list | None = None,
        exclude_hidden: bool = True,
        exclude_empty: bool = False,
        errors: str = "ignore",
        recursive: bool = False,
        encoding: str = "utf-8",
        filename_as_id: bool = False,
        required_exts: list[str] | None = None,
        file_extractor: dict[str, BaseReader] | None = None,
        num_files_limit: int | None = None,
        file_metadata: Callable[[str], dict] | None = None,
        raise_on_error: bool = False,
        fs: fsspec.AbstractFileSystem | None = None,
    ) -> None:
        """Initialize with parameters."""
        super().__init__()

        if not input_dir and not input_files:
            raise ValueError("Must provide either `input_dir` or `input_files`.")

        self.fs = fs or get_default_fs()
        self.errors = errors
        self.encoding = encoding

        self.exclude = exclude
        self.recursive = recursive
        self.exclude_hidden = exclude_hidden
        self.exclude_empty = exclude_empty
        self.required_exts = required_exts
        self.num_files_limit = num_files_limit
        self.raise_on_error = raise_on_error
        _Path = Path if is_default_fs(self.fs) else PurePosixPath

        if input_files:
            self.input_files = []
            for path in input_files:
                if not self.fs.isfile(path):
                    raise ValueError(f"File {path} does not exist.")
                input_file = _Path(path)
                self.input_files.append(input_file)
        elif input_dir:
            if not self.fs.isdir(input_dir):
                raise ValueError(f"Directory {input_dir} does not exist.")
            self.input_dir = _Path(input_dir)
            self.exclude = exclude
            self.input_files = self._add_files(self.input_dir)

        self.file_extractor = file_extractor or {}
        self.file_metadata = file_metadata or _DefaultFileMetadataFunc(self.fs)
        self.filename_as_id = filename_as_id

    def is_hidden(self, path: Path | PurePosixPath) -> bool:
        return any(
            part.startswith(".") and part not in [".", ".."] for part in path.parts
        )

    def is_empty_file(self, path: Path | PurePosixPath) -> bool:
        if isinstance(path, PurePosixPath):
            path = Path(path)
        return path.is_file() and len(path.read_bytes()) == 0

    def _add_files(self, input_dir: Path | PurePosixPath) -> list[Path | PurePosixPath]:
        """Add files."""
        all_files: set[Path | PurePosixPath] = set()
        rejected_files: set[Path | PurePosixPath] = set()
        rejected_dirs: set[Path | PurePosixPath] = set()
        # Default to POSIX paths for non-default file systems (e.g. S3)
        _Path = Path if is_default_fs(self.fs) else PurePosixPath

        if self.exclude is not None:
            for excluded_pattern in self.exclude:
                if self.recursive:
                    # Recursive glob
                    excluded_glob = _Path(input_dir) / _Path("**") / excluded_pattern
                else:
                    # Non-recursive glob
                    excluded_glob = _Path(input_dir) / excluded_pattern
                for file in self.fs.glob(str(excluded_glob)):
                    if self.fs.isdir(file):
                        rejected_dirs.add(_Path(str(file)))
                    else:
                        rejected_files.add(_Path(str(file)))

        file_refs: list[str] = []
        if self.recursive:
            file_refs = cast(list[str], self.fs.glob(str(input_dir) + "/**/*"))
        else:
            file_refs = cast(list[str], self.fs.glob(str(input_dir) + "/*"))

        for _ref in file_refs:
            # Manually check if file is hidden or directory instead of
            # in glob for backwards compatibility.
            ref = _Path(_ref)
            is_dir = self.fs.isdir(ref)
            skip_because_hidden = self.exclude_hidden and self.is_hidden(ref)
            skip_because_empty = self.exclude_empty and self.is_empty_file(ref)
            skip_because_bad_ext = (
                self.required_exts is not None and ref.suffix not in self.required_exts
            )
            skip_because_excluded = ref in rejected_files
            if not skip_because_excluded:
                if is_dir:
                    ref_parent_dir = ref
                else:
                    ref_parent_dir = self.fs._parent(ref)
                for rejected_dir in rejected_dirs:
                    if str(ref_parent_dir).startswith(str(rejected_dir)):
                        skip_because_excluded = True
                        logger.debug(
                            "Skipping %s because it in parent dir %s which is in %s",
                            ref,
                            ref_parent_dir,
                            rejected_dir,
                        )
                        break

            if (
                is_dir
                or skip_because_hidden
                or skip_because_bad_ext
                or skip_because_excluded
                or skip_because_empty
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

    def _exclude_metadata(self, documents: list[Document]) -> list[Document]:
        """
        Exclude metadata from documents.

        Args:
            documents (List[Document]): List of documents.
        """
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

    def list_resources(self, *args: Any, **kwargs: Any) -> list[str]:
        """List files in the given filesystem."""
        return [str(x) for x in self.input_files]

    def get_resource_info(self, resource_id: str, *args: Any, **kwargs: Any) -> dict:
        info_result = self.fs.info(resource_id)

        creation_date = _format_file_timestamp(
            info_result.get("created"), include_time=True
        )
        last_modified_date = _format_file_timestamp(
            info_result.get("mtime"), include_time=True
        )

        info_dict = {
            "file_path": resource_id,
            "file_size": info_result.get("size"),
            "creation_date": creation_date,
            "last_modified_date": last_modified_date,
        }

        # Ignore None values
        return {
            meta_key: meta_value
            for meta_key, meta_value in info_dict.items()
            if meta_value is not None
        }

    def load_resource(
        self, resource_id: str, *args: Any, **kwargs: Any
    ) -> list[Document]:
        file_metadata = kwargs.get("file_metadata", self.file_metadata)
        file_extractor = kwargs.get("file_extractor", self.file_extractor)
        filename_as_id = kwargs.get("filename_as_id", self.filename_as_id)
        encoding = kwargs.get("encoding", self.encoding)
        errors = kwargs.get("errors", self.errors)
        raise_on_error = kwargs.get("raise_on_error", self.raise_on_error)
        fs = kwargs.get("fs", self.fs)

        path_func = Path if is_default_fs(fs) else PurePosixPath

        return SimpleDirectoryReader.load_file(
            input_file=path_func(resource_id),
            file_metadata=file_metadata,
            file_extractor=file_extractor,
            filename_as_id=filename_as_id,
            encoding=encoding,
            errors=errors,
            raise_on_error=raise_on_error,
            fs=fs,
            **kwargs,
        )

    async def aload_resource(
        self, resource_id: str, *args: Any, **kwargs: Any
    ) -> list[Document]:
        file_metadata = kwargs.get("file_metadata", self.file_metadata)
        file_extractor = kwargs.get("file_extractor", self.file_extractor)
        filename_as_id = kwargs.get("filename_as_id", self.filename_as_id)
        encoding = kwargs.get("encoding", self.encoding)
        errors = kwargs.get("errors", self.errors)
        raise_on_error = kwargs.get("raise_on_error", self.raise_on_error)
        fs = kwargs.get("fs", self.fs)

        return await SimpleDirectoryReader.aload_file(
            input_file=Path(resource_id),
            file_metadata=file_metadata,
            file_extractor=file_extractor,
            filename_as_id=filename_as_id,
            encoding=encoding,
            errors=errors,
            raise_on_error=raise_on_error,
            fs=fs,
            **kwargs,
        )

    def read_file_content(self, input_file: Path, **kwargs: Any) -> bytes:
        """Read file content."""
        fs: fsspec.AbstractFileSystem = kwargs.get("fs", self.fs)
        with fs.open(input_file, errors=self.errors, encoding=self.encoding) as f:
            # default mode is 'rb', we can cast the return value of f.read()
            return cast(bytes, f.read())

    @staticmethod
    def load_file(
        input_file: Path | PurePosixPath,
        file_metadata: Callable[[str], dict],
        file_extractor: dict[str, BaseReader],
        filename_as_id: bool = False,
        encoding: str = "utf-8",
        errors: str = "ignore",
        raise_on_error: bool = False,
        fs: fsspec.AbstractFileSystem | None = None,
    ) -> list[Document]:
        """
        Static method for loading file.

        NOTE: necessarily as a static method for parallel processing.

        Args:
            input_file (Path): _description_
            file_metadata (Callable[[str], Dict]): _description_
            file_extractor (Dict[str, BaseReader]): _description_
            filename_as_id (bool, optional): _description_. Defaults to False.
            encoding (str, optional): _description_. Defaults to "utf-8".
            errors (str, optional): _description_. Defaults to "ignore".
            fs (Optional[fsspec.AbstractFileSystem], optional): _description_. Defaults to None.

        input_file (Path): File path to read
        file_metadata ([Callable[str, Dict]]): A function that takes
            in a filename and returns a Dict of metadata for the Document.
        file_extractor (Dict[str, BaseReader]): A mapping of file
            extension to a BaseReader class that specifies how to convert that file
            to text.
        filename_as_id (bool): Whether to use the filename as the document id.
        encoding (str): Encoding of the files.
            Default is utf-8.
        errors (str): how encoding and decoding errors are to be handled,
              see https://docs.python.org/3/library/functions.html#open
        raise_on_error (bool): Whether to raise an error if a file cannot be read.
        fs (Optional[fsspec.AbstractFileSystem]): File system to use. Defaults
            to using the local file system. Can be changed to use any remote file system

        Returns:
            List[Document]: loaded documents
        """
        # TODO: make this less redundant
        default_file_reader_cls = SimpleDirectoryReader.supported_suffix_fn()
        default_file_reader_suffix = list(default_file_reader_cls.keys())
        metadata: dict | None = None
        documents: list[Document] = []

        if file_metadata is not None:
            metadata = file_metadata(str(input_file))

        file_suffix = input_file.suffix.lower()
        if file_suffix in default_file_reader_suffix or file_suffix in file_extractor:
            # use file readers
            if file_suffix not in file_extractor:
                # instantiate file reader if not already
                reader_cls = default_file_reader_cls[file_suffix]
                file_extractor[file_suffix] = reader_cls()
            reader = file_extractor[file_suffix]

            # load data -- catch all errors except for ImportError
            try:
                kwargs: dict[str, Any] = {"extra_info": metadata}
                if fs and not is_default_fs(fs):
                    kwargs["fs"] = fs
                docs = reader.load_data(input_file, **kwargs)
            except ImportError as e:
                # ensure that ImportError is raised so user knows
                # about missing dependencies
                raise ImportError(str(e))
            except Exception as e:
                if raise_on_error:
                    raise Exception("Error loading file") from e
                # otherwise, just skip the file and report the error
                print(
                    f"Failed to load file {input_file} with error: {e}. Skipping...",
                    flush=True,
                )
                return []

            # iterate over docs if needed
            if filename_as_id:
                for i, doc in enumerate(docs):
                    doc.id_ = f"{input_file!s}_part_{i}"

            documents.extend(docs)
        else:
            # do standard read
            fs = fs or get_default_fs()
            with fs.open(input_file, errors=errors, encoding=encoding) as f:
                data = cast(bytes, f.read()).decode(encoding, errors=errors)

            doc = Document(text=data, metadata=metadata or {})  # type: ignore
            if filename_as_id:
                doc.id_ = str(input_file)

            documents.append(doc)

        return documents

    @staticmethod
    async def aload_file(
        input_file: Path | PurePosixPath,
        file_metadata: Callable[[str], dict],
        file_extractor: dict[str, BaseReader],
        filename_as_id: bool = False,
        encoding: str = "utf-8",
        errors: str = "ignore",
        raise_on_error: bool = False,
        fs: fsspec.AbstractFileSystem | None = None,
    ) -> list[Document]:
        """Load file asynchronously."""
        # TODO: make this less redundant
        default_file_reader_cls = SimpleDirectoryReader.supported_suffix_fn()
        default_file_reader_suffix = list(default_file_reader_cls.keys())
        metadata: dict | None = None
        documents: list[Document] = []

        if file_metadata is not None:
            metadata = file_metadata(str(input_file))

        file_suffix = input_file.suffix.lower()
        if file_suffix in default_file_reader_suffix or file_suffix in file_extractor:
            # use file readers
            if file_suffix not in file_extractor:
                # instantiate file reader if not already
                reader_cls = default_file_reader_cls[file_suffix]
                file_extractor[file_suffix] = reader_cls()
            reader = file_extractor[file_suffix]

            # load data -- catch all errors except for ImportError
            try:
                kwargs: dict[str, Any] = {"extra_info": metadata}
                if fs and not is_default_fs(fs):
                    kwargs["fs"] = fs
                docs = await reader.aload_data(input_file, **kwargs)
            except ImportError as e:
                # ensure that ImportError is raised so user knows
                # about missing dependencies
                raise ImportError(str(e))
            except Exception as e:
                if raise_on_error:
                    raise
                # otherwise, just skip the file and report the error
                print(
                    f"Failed to load file {input_file} with error: {e}. Skipping...",
                    flush=True,
                )
                return []

            # iterate over docs if needed
            if filename_as_id:
                for i, doc in enumerate(docs):
                    doc.id_ = f"{input_file!s}_part_{i}"

            documents.extend(docs)
        else:
            # do standard read
            fs = fs or get_default_fs()
            with fs.open(input_file, errors=errors, encoding=encoding) as f:
                data = cast(bytes, f.read()).decode(encoding, errors=errors)

            doc = Document(text=data, metadata=metadata or {})  # type: ignore
            if filename_as_id:
                doc.id_ = str(input_file)

            documents.append(doc)

        return documents

    def load_data(
        self,
        show_progress: bool = False,
        num_workers: int | None = None,
        fs: fsspec.AbstractFileSystem | None = None,
    ) -> list[Document]:
        """
        Load data from the input directory.

        Args:
            show_progress (bool): Whether to show tqdm progress bars. Defaults to False.
            num_workers  (Optional[int]): Number of workers to parallelize data-loading over.
            fs (Optional[fsspec.AbstractFileSystem]): File system to use. If fs was specified
                in the constructor, it will override the fs parameter here.

        Returns:
            List[Document]: A list of documents.
        """
        documents = []

        files_to_process = self.input_files
        fs = fs or self.fs

        if num_workers and num_workers > 1:
            num_cpus = multiprocessing.cpu_count()
            if num_workers > num_cpus:
                warnings.warn(
                    "Specified num_workers exceed number of CPUs in the system. "
                    "Setting `num_workers` down to the maximum CPU count."
                )
                num_workers = num_cpus

            with multiprocessing.get_context("spawn").Pool(num_workers) as p:
                results = p.starmap(
                    SimpleDirectoryReader.load_file,
                    zip(
                        files_to_process,
                        repeat(self.file_metadata),
                        repeat(self.file_extractor),
                        repeat(self.filename_as_id),
                        repeat(self.encoding),
                        repeat(self.errors),
                        repeat(self.raise_on_error),
                        repeat(fs),
                    ),
                )
                documents = reduce(lambda x, y: x + y, results)

        else:
            if show_progress:
                files_to_process = tqdm(
                    self.input_files, desc="Loading files", unit="file"
                )
            for input_file in files_to_process:
                documents.extend(
                    SimpleDirectoryReader.load_file(
                        input_file=input_file,
                        file_metadata=self.file_metadata,
                        file_extractor=self.file_extractor,
                        filename_as_id=self.filename_as_id,
                        encoding=self.encoding,
                        errors=self.errors,
                        raise_on_error=self.raise_on_error,
                        fs=fs,
                    )
                )

        return self._exclude_metadata(documents)

    async def aload_data(
        self,
        show_progress: bool = False,
        num_workers: int | None = None,
        fs: fsspec.AbstractFileSystem | None = None,
    ) -> list[Document]:
        """
        Load data from the input directory.

        Args:
            show_progress (bool): Whether to show tqdm progress bars. Defaults to False.
            num_workers  (Optional[int]): Number of workers to parallelize data-loading over.
            fs (Optional[fsspec.AbstractFileSystem]): File system to use. If fs was specified
                in the constructor, it will override the fs parameter here.

        Returns:
            List[Document]: A list of documents.
        """
        files_to_process = self.input_files
        fs = fs or self.fs

        coroutines = [
            SimpleDirectoryReader.aload_file(
                input_file,
                self.file_metadata,
                self.file_extractor,
                self.filename_as_id,
                self.encoding,
                self.errors,
                self.raise_on_error,
                fs,
            )
            for input_file in files_to_process
        ]

        if num_workers:
            document_lists = await run_jobs(
                coroutines, show_progress=show_progress, workers=num_workers
            )
        elif show_progress:
            _asyncio = get_asyncio_module(show_progress=show_progress)
            document_lists = await _asyncio.gather(*coroutines)
        else:
            document_lists = await asyncio.gather(*coroutines)
        documents = [doc for doc_list in document_lists for doc in doc_list]

        return self._exclude_metadata(documents)

    def iter_data(
        self, show_progress: bool = False
    ) -> Generator[list[Document], Any, Any]:
        """
        Load data iteratively from the input directory.

        Args:
            show_progress (bool): Whether to show tqdm progress bars. Defaults to False.

        Returns:
            Generator[List[Document]]: A list of documents.
        """
        files_to_process = self.input_files

        if show_progress:
            files_to_process = tqdm(self.input_files, desc="Loading files", unit="file")

        for input_file in files_to_process:
            documents = SimpleDirectoryReader.load_file(
                input_file=input_file,
                file_metadata=self.file_metadata,
                file_extractor=self.file_extractor,
                filename_as_id=self.filename_as_id,
                encoding=self.encoding,
                errors=self.errors,
                raise_on_error=self.raise_on_error,
                fs=self.fs,
            )

            documents = self._exclude_metadata(documents)

            if len(documents) > 0:
                yield documents
