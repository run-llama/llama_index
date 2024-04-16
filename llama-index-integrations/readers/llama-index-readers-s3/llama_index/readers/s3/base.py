"""
S3 file and directory reader.

A loader that fetches a file or iterates through a directory on AWS S3.

"""

import warnings
from typing import Callable, Dict, List, Optional, Union

from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.readers.base import BaseReader, BasePydanticReader
from llama_index.core.schema import Document
from llama_index.core.bridge.pydantic import Field


class S3Reader(BasePydanticReader):
    """
    General reader for any S3 file or directory.

    If key is not set, the entire bucket (filtered by prefix) is parsed.

    Args:
    bucket (str): the name of your S3 bucket
    key (Optional[str]): the name of the specific file. If none is provided,
        this loader will iterate through the entire bucket.
    prefix (Optional[str]): the prefix to filter by in the case that the loader
        iterates through the entire bucket. Defaults to empty string.
    recursive (bool): Whether to recursively search in subdirectories.
        True by default.
    file_extractor (Optional[Dict[str, BaseReader]]): A mapping of file
        extension to a BaseReader class that specifies how to convert that file
        to text. See `SimpleDirectoryReader` for more details.
    required_exts (Optional[List[str]]): List of required extensions.
        Default is None.
    num_files_limit (Optional[int]): Maximum number of files to read.
        Default is None.
    file_metadata (Optional[Callable[str, Dict]]): A function that takes
        in a filename and returns a Dict of metadata for the Document.
        Default is None.
    aws_access_id (Optional[str]): provide AWS access key directly.
    aws_access_secret (Optional[str]): provide AWS access key directly.
    s3_endpoint_url (Optional[str]): provide S3 endpoint URL directly.
    """

    is_remote: bool = True

    bucket: str
    key: Optional[str] = None
    prefix: Optional[str] = ""
    recursive: bool = True
    file_extractor: Optional[Dict[str, Union[str, BaseReader]]] = Field(
        default=None, exclude=True
    )
    required_exts: Optional[List[str]] = None
    filename_as_id: bool = True
    num_files_limit: Optional[int] = None
    file_metadata: Optional[Callable[[str], Dict]] = Field(default=None, exclude=True)
    aws_access_id: Optional[str] = None
    aws_access_secret: Optional[str] = None
    aws_session_token: Optional[str] = None
    s3_endpoint_url: Optional[str] = None
    custom_reader_path: Optional[str] = None

    @classmethod
    def class_name(cls) -> str:
        return "S3Reader"

    def load_s3_files_as_docs(self, temp_dir=None) -> List[Document]:
        """Load file(s) from S3."""
        from s3fs import S3FileSystem

        s3fs = S3FileSystem(
            key=self.aws_access_id,
            endpoint_url=self.s3_endpoint_url,
            secret=self.aws_access_secret,
            token=self.aws_session_token,
        )

        input_dir = self.bucket
        input_files = None

        if self.key:
            input_files = [f"{self.bucket}/{self.key}"]
        elif self.prefix:
            input_dir = f"{input_dir}/{self.prefix}"

        loader = SimpleDirectoryReader(
            input_dir=input_dir,
            input_files=input_files,
            file_extractor=self.file_extractor,
            required_exts=self.required_exts,
            filename_as_id=self.filename_as_id,
            num_files_limit=self.num_files_limit,
            file_metadata=self.file_metadata,
            recursive=self.recursive,
            fs=s3fs,
        )

        return loader.load_data()

    def load_data(self, custom_temp_subdir: str = None) -> List[Document]:
        """
        Load the file(s) from S3.

        Args:
            custom_temp_subdir (str, optional): This parameter is deprecated and unused. Defaults to None.

        Returns:
            List[Document]: A list of documents loaded from S3.
        """
        if custom_temp_subdir is not None:
            warnings.warn(
                "The `custom_temp_subdir` parameter is deprecated and unused. Please remove it from your code.",
                DeprecationWarning,
            )

        documents = self.load_s3_files_as_docs()
        for doc in documents:
            if self.s3_endpoint_url:
                doc.id_ = self.s3_endpoint_url + "_" + doc.id_
            else:
                doc.id_ = "s3_" + doc.id_

        return documents
