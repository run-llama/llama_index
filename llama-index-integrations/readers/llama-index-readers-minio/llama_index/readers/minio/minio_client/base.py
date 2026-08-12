"""
Minio file and directory reader.

A loader that fetches a file or iterates through a directory on Minio.

"""

import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class MinioReader(BaseReader):
    """General reader for any Minio file or directory."""

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
        minio_endpoint: Optional[str] = None,
        minio_secure: bool = False,
        minio_cert_check: bool = True,
        minio_access_key: Optional[str] = None,
        minio_secret_key: Optional[str] = None,
        minio_session_token: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize Minio bucket and key, along with credentials if needed.

        If key is not set, the entire bucket (filtered by prefix) is parsed.

        Args:
        bucket (str): the name of your Minio bucket
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
        minio_endpoint (Optional[str]): The Minio endpoint. Default is None.
        minio_port (Optional[int]): The Minio port. Default is None.
        minio_access_key (Optional[str]): The Minio access key. Default is None.
        minio_secret_key (Optional[str]): The Minio secret key. Default is None.
        minio_session_token (Optional[str]): The Minio session token.
        minio_secure: MinIO server runs in TLS mode
        minio_cert_check: allows the usage of a self-signed cert for MinIO server

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

        self.minio_endpoint = minio_endpoint
        self.minio_secure = minio_secure
        self.minio_cert_check = minio_cert_check
        self.minio_access_key = minio_access_key
        self.minio_secret_key = minio_secret_key
        self.minio_session_token = minio_session_token

    def load_data(self) -> List[Document]:
        """Load file(s) from Minio."""
        from minio import Minio

        minio_client = Minio(
            self.minio_endpoint,
            secure=self.minio_secure,
            cert_check=self.minio_cert_check,
            access_key=self.minio_access_key,
            secret_key=self.minio_secret_key,
            session_token=self.minio_session_token,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            if self.key:
                suffix = Path(self.key).suffix
                fd, filepath = tempfile.mkstemp(dir=temp_dir, suffix=suffix)
                # close the mkstemp handle so the download can write to the
                # file and the temp dir can be cleaned up on Windows
                os.close(fd)
                minio_client.fget_object(
                    bucket_name=self.bucket, object_name=self.key, file_path=filepath
                )
            else:
                objects = minio_client.list_objects(
                    bucket_name=self.bucket, prefix=self.prefix, recursive=True
                )
                temp_root = Path(temp_dir).resolve()
                for i, obj in enumerate(objects):
                    if self.num_files_limit is not None and i > self.num_files_limit:
                        break

                    suffix = Path(obj.object_name).suffix

                    is_dir = obj.object_name.endswith("/")  # skip folders
                    is_bad_ext = (
                        self.required_exts is not None
                        and suffix not in self.required_exts  # skip other extensions
                    )

                    if is_dir or is_bad_ext:
                        continue

                    # one directory per object so same-named files don't overwrite
                    file_name = Path(obj.object_name.replace("\\", "/")).name
                    download_dir = temp_root / f"{i:08d}"
                    filepath = (download_dir / file_name).resolve()
                    if not file_name or not filepath.is_relative_to(download_dir):
                        raise ValueError(f"Unsafe object name: {obj.object_name}")
                    download_dir.mkdir()
                    minio_client.fget_object(
                        self.bucket, obj.object_name, str(filepath)
                    )

            loader = SimpleDirectoryReader(
                temp_dir,
                recursive=True,
                file_extractor=self.file_extractor,
                required_exts=self.required_exts,
                filename_as_id=self.filename_as_id,
                num_files_limit=self.num_files_limit,
                file_metadata=self.file_metadata,
            )

            return loader.load_data()
