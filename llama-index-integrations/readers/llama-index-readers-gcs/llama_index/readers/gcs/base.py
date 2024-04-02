"""
GCS file and directory reader.

A loader that fetches a file or iterates through a directory on Google Cloud Storage (GCS).

"""
import json
from typing import Callable, Dict, List, Optional, Union

from google.oauth2 import service_account
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.readers.base import BaseReader, BasePydanticReader
from llama_index.core.schema import Document
from llama_index.core.bridge.pydantic import Field

# Scope for reading and downloading GCS files
SCOPES = ["https://www.googleapis.com/auth/devstorage.read_only"]


class GCSReader(BasePydanticReader):
    """
    General reader for any GCS file or directory.

    If key is not set, the entire bucket (filtered by prefix) is parsed.

    Args:
    bucket (str): the name of your GCS bucket
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
    service_account_key (Optional[Dict[str, str]]): provide GCP service account key directly.
    service_account_key_json (Optional[str]): provide GCP service account key as a JSON string.
    service_account_key_path (Optional[str]): provide path to file containing GCP service account key.
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
    service_account_key: Optional[Dict[str, str]] = None
    service_account_key_json: Optional[str] = None
    service_account_key_path: Optional[str] = None

    @classmethod
    def class_name(cls) -> str:
        return "GCSReader"

    def load_gcs_files_as_docs(self) -> List[Document]:
        """Load file(s) from GCS."""
        from gcsfs import GCSFileSystem

        if self.service_account_key is not None:
            creds = service_account.Credentials.from_service_account_info(
                self.service_account_key, scopes=SCOPES
            )
        elif self.service_account_key_json is not None:
            creds = service_account.Credentials.from_service_account_info(
                json.loads(self.service_account_key_json), scopes=SCOPES
            )
        elif self.service_account_key_path is not None:
            creds = service_account.Credentials.from_service_account_file(
                self.service_account_key_path, scopes=SCOPES
            )
        else:
            # Use anonymous access if none are specified
            creds = "anon"

        gcsfs = GCSFileSystem(
            token=creds,
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
            recursive=self.recursive,
            file_extractor=self.file_extractor,
            required_exts=self.required_exts,
            filename_as_id=self.filename_as_id,
            num_files_limit=self.num_files_limit,
            file_metadata=self.file_metadata,
            fs=gcsfs,
        )

        return loader.load_data()

    def load_data(self) -> List[Document]:
        """
        Load the file(s) from GCS.

        Returns:
            List[Document]: A list of documents loaded from GCS.
        """
        return self.load_gcs_files_as_docs()
