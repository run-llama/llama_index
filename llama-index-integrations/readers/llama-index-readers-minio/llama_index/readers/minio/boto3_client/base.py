"""
Minio file and directory reader.

A loader that fetches a file or iterates through a directory on Minio.

"""

import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class BotoMinioReader(BaseReader):
    """
    General reader for any S3 file or directory.
    A loader that fetches a file or iterates through a directory on minio using boto3.

    """

    def __init__(
        self,
        *args: Any,
        bucket: str,
        key: Optional[str] = None,
        prefix: Optional[str] = "",
        file_extractor: Optional[Dict[str, Union[str, BaseReader]]] = None,
        required_exts: Optional[List[str]] = None,
        filename_as_id: bool = False,
        num_files_limit: Optional[int] = None,
        file_metadata: Optional[Callable[[str], Dict]] = None,
        aws_access_id: Optional[str] = None,
        aws_access_secret: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        s3_endpoint_url: Optional[str] = "https://s3.amazonaws.com",
        **kwargs: Any,
    ) -> None:
        """
        Initialize S3 bucket and key, along with credentials if needed.

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

        self.aws_access_id = aws_access_id
        self.aws_access_secret = aws_access_secret
        self.aws_session_token = aws_session_token
        self.s3_endpoint_url = s3_endpoint_url

    def load_data(self) -> List[Document]:
        """Load file(s) from S3."""
        import boto3

        s3_client = boto3.client(
            "s3",
            endpoint_url=self.s3_endpoint_url,
            aws_access_key_id=self.aws_access_id,
            aws_secret_access_key=self.aws_access_secret,
            aws_session_token=self.aws_session_token,
            config=boto3.session.Config(signature_version="s3v4"),
            verify=False,
        )
        s3 = boto3.resource(
            "s3",
            endpoint_url=self.s3_endpoint_url,
            aws_access_key_id=self.aws_access_id,
            aws_secret_access_key=self.aws_access_secret,
            aws_session_token=self.aws_session_token,
            config=boto3.session.Config(signature_version="s3v4"),
            verify=False,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            if self.key:
                suffix = Path(self.key).suffix
                filepath = f"{temp_dir}/{next(tempfile._get_candidate_names())}{suffix}"
                s3_client.download_file(self.bucket, self.key, filepath)
            else:
                bucket = s3.Bucket(self.bucket)
                for i, obj in enumerate(bucket.objects.filter(Prefix=self.prefix)):
                    if self.num_files_limit is not None and i > self.num_files_limit:
                        break

                    suffix = Path(obj.key).suffix

                    is_dir = obj.key.endswith("/")  # skip folders
                    is_bad_ext = (
                        self.required_exts is not None
                        and suffix not in self.required_exts  # skip other extensions
                    )

                    if is_dir or is_bad_ext:
                        continue

                    filepath = (
                        f"{temp_dir}/{next(tempfile._get_candidate_names())}{suffix}"
                    )
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
