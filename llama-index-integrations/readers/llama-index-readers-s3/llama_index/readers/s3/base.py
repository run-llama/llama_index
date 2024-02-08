"""S3 file and directory reader.

A loader that fetches a file or iterates through a directory on AWS S3.

"""
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class S3Reader(BaseReader):
    """General reader for any S3 file or directory."""

    def __init__(
        self,
        *args: Any,
        bucket: str,
        key: Optional[str] = None,
        prefix: Optional[str] = "",
        file_extractor: Optional[Dict[str, Union[str, BaseReader]]] = None,
        required_exts: Optional[List[str]] = None,
        filename_as_id: bool = True,
        num_files_limit: Optional[int] = None,
        file_metadata: Optional[Callable[[str], Dict]] = None,
        aws_access_id: Optional[str] = None,
        aws_access_secret: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        s3_endpoint_url: Optional[str] = "https://s3.amazonaws.com",
        custom_reader_path: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize S3 bucket and key, along with credentials if needed.

        If key is not set, the entire bucket (filtered by prefix) is parsed.

        Args:
        bucket (str): the name of your S3 bucket
        key (Optional[str]): the name of the specific file. If none is provided,
            this loader will iterate through the entire bucket.
        prefix (Optional[str]): the prefix to filter by in the case that the loader
            iterates through the entire bucket. Defaults to empty string.
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
        super().__init__(*args, **kwargs)

        self.bucket = bucket
        self.key = key
        self.prefix = prefix

        self.file_extractor = file_extractor
        self.required_exts = required_exts
        self.filename_as_id = filename_as_id
        self.num_files_limit = num_files_limit
        self.file_metadata = file_metadata
        self.custom_reader_path = custom_reader_path

        self.aws_access_id = aws_access_id
        self.aws_access_secret = aws_access_secret
        self.aws_session_token = aws_session_token
        self.s3_endpoint_url = s3_endpoint_url

    def load_s3_files_as_docs(self, temp_dir) -> List[Document]:
        """Load file(s) from S3."""
        import boto3

        s3 = boto3.resource("s3")
        s3_client = boto3.client("s3")
        if self.aws_access_id:
            session = boto3.Session(
                aws_access_key_id=self.aws_access_id,
                aws_secret_access_key=self.aws_access_secret,
                aws_session_token=self.aws_session_token,
            )
            s3 = session.resource("s3", endpoint_url=self.s3_endpoint_url)
            s3_client = session.client("s3", endpoint_url=self.s3_endpoint_url)

        if self.key:
            filename = Path(self.key).name
            suffix = Path(self.key).suffix
            filepath = f"{temp_dir}/{filename}"
            s3_client.download_file(self.bucket, self.key, filepath)
        else:
            bucket = s3.Bucket(self.bucket)
            for i, obj in enumerate(bucket.objects.filter(Prefix=self.prefix)):
                if self.num_files_limit is not None and i > self.num_files_limit:
                    break
                filename = Path(obj.key).name
                suffix = Path(obj.key).suffix

                is_dir = obj.key.endswith("/")  # skip folders
                is_bad_ext = (
                    self.required_exts is not None
                    and suffix not in self.required_exts  # skip other extentions
                )

                if is_dir or is_bad_ext:
                    continue

                filepath = f"{temp_dir}/{filename}"
                s3_client.download_file(self.bucket, obj.key, filepath)

        loader = SimpleDirectoryReader(
            temp_dir,
            file_extractor=self.file_extractor,
            required_exts=self.required_exts,
            filename_as_id=self.filename_as_id,
            num_files_limit=self.num_files_limit,
            file_metadata=self.file_metadata,
        )

        return loader.load_data()

    def load_data(self, custom_temp_subdir: str = None) -> List[Document]:
        """Decide which directory to load files in - randomly generated directories under /tmp or a custom subdirectory under /tmp."""
        if custom_temp_subdir is None:
            with tempfile.TemporaryDirectory() as temp_dir:
                documents = self.load_s3_files_as_docs(temp_dir)
        else:
            temp_dir = os.path.join("/tmp", custom_temp_subdir)
            os.makedirs(temp_dir, exist_ok=True)
            documents = self.load_s3_files_as_docs(temp_dir)
            shutil.rmtree(temp_dir)

        for doc in documents:
            doc.id_ = self.s3_endpoint_url + "_" + doc.id_

        return documents
