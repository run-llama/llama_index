"""
S3 file and directory reader.

A loader that fetches a file or iterates through a directory on AWS S3 or other compatible service.

"""

from typing import Dict, List, Optional, Union

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from llama_index.readers.opendal.base import OpendalReader


class OpendalS3Reader(BaseReader):
    """General reader for any S3 file or directory."""

    def __init__(
        self,
        bucket: str,
        path: str = "/",
        endpoint: str = "",
        region: str = "",
        access_key_id: str = "",
        secret_access_key: str = "",
        file_extractor: Optional[Dict[str, Union[str, BaseReader]]] = None,
    ) -> None:
        """
        Initialize S3 bucket and key, along with credentials if needed.

        If key is not set, the entire bucket (filtered by prefix) is parsed.

        Args:
        bucket (str): the name of your S3 bucket
        path (str): the path of the data. If none is provided,
            this loader will iterate through the entire bucket. If path is endswith `/`, this loader will iterate through the entire dir. Otherwise, this loeader will load the file.
        endpoint Optional[str]: the endpoint of the S3 service.
        region: Optional[str]: the region of the S3 service.
        access_key_id (Optional[str]): provide AWS access key directly.
        secret_access_key (Optional[str]): provide AWS access key directly.
        file_extractor (Optional[Dict[str, BaseReader]]): A mapping of file
            extension to a BaseReader class that specifies how to convert that file
            to text. See `SimpleDirectoryReader` for more details.

        """
        super().__init__()

        self.path = path
        self.file_extractor = file_extractor

        # opendal service related config.
        self.options = {
            "access_key": access_key_id,
            "secret_key": secret_access_key,
            "endpoint": endpoint,
            "region": region,
            "bucket": bucket,
        }

    def load_data(self) -> List[Document]:
        """Load file(s) from OpenDAL."""
        loader = OpendalReader(
            scheme="s3",
            path=self.path,
            file_extractor=self.file_extractor,
            **self.options,
        )

        return loader.load_data()
