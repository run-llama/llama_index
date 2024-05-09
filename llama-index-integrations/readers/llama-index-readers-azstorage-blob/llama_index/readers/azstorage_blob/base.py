"""Azure Storage Blob file and directory reader.

A loader that fetches a file or iterates through a directory from Azure Storage Blob.

"""
import logging
import math
import os
import tempfile
import time
from typing import Any, Dict, List, Optional, Union

from azure.storage.blob import ContainerClient

from llama_index.core.bridge.pydantic import Field
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.readers.base import BaseReader, BasePydanticReader
from llama_index.core.schema import Document

logger = logging.getLogger(__name__)


class AzStorageBlobReader(BasePydanticReader):
    """General reader for any Azure Storage Blob file or directory.

    Args:
        container_name (str): name of the container for the blob.
        blob (Optional[str]): name of the file to download. If none specified
            this loader will iterate through list of blobs in the container.
        name_starts_with (Optional[str]): filter the list of blobs to download
            to only those whose names begin with the specified string.
        include: (Union[str, List[str], None]): Specifies one or more additional
            datasets to include in the response. Options include: 'snapshots',
            'metadata', 'uncommittedblobs', 'copy', 'deleted',
            'deletedwithversions', 'tags', 'versions', 'immutabilitypolicy',
            'legalhold'.
        file_extractor (Optional[Dict[str, Union[str, BaseReader]]]): A mapping of file
            extension to a BaseReader class that specifies how to convert that file
            to text. See `SimpleDirectoryReader` for more details, or call this path ```llama_index.readers.file.base.DEFAULT_FILE_READER_CLS```.
        connection_string (str): A connection string which can be found under a storage account's "Access keys" security tab. This parameter
        can be used in place of both the account URL and credential.
        account_url (str): URI to the storage account, may include SAS token.
        credential (Union[str, Dict[str, str], AzureNamedKeyCredential, AzureSasCredential, TokenCredential, None] = None):
            The credentials with which to authenticate. This is optional if the account URL already has a SAS token.
    """

    container_name: str
    prefix: Optional[str] = ""
    blob: Optional[str] = None
    name_starts_with: Optional[str] = None
    include: Optional[Any] = None
    file_extractor: Optional[Dict[str, Union[str, BaseReader]]] = Field(
        default=None, exclude=True
    )
    connection_string: Optional[str] = None
    account_url: Optional[str] = None
    credential: Optional[Any] = None
    is_remote: bool = True

    # Not in use. As part of the TODO below. Is part of the kwargs.
    # self.preloaded_data_path = kwargs.get('preloaded_data_path', None)

    @classmethod
    def class_name(cls) -> str:
        return "AzStorageBlobReader"

    def load_data(self) -> List[Document]:
        """Load file(s) from Azure Storage Blob."""
        if self.connection_string:
            container_client = ContainerClient.from_connection_string(
                conn_str=self.connection_string,
                container_name=self.container_name,
            )
        else:
            container_client = ContainerClient(
                self.account_url, self.container_name, credential=self.credential
            )
        total_download_start_time = time.time()
        blob_meta = {}

        with tempfile.TemporaryDirectory() as temp_dir:
            if self.blob:
                blob_client = container_client.get_blob_client(self.blob)
                stream = blob_client.download_blob()
                sanitized_file_name = stream.name.replace("/", "-")
                download_file_path = os.path.join(temp_dir, sanitized_file_name)
                logger.info(f"Start download of {self.blob}")
                start_time = time.time()
                with open(file=download_file_path, mode="wb") as download_file:
                    stream.readinto(download_file)
                blob_meta[download_file_path] = blob_client.get_blob_properties()
                end_time = time.time()
                logger.info(
                    f"{self.blob} downloaded in {end_time - start_time} seconds."
                )
            # TODO: Implement an "elif" for if a pickled dictionary of the Document objects are already stored, to load that in and read into the temp directory.
            # Needed because the loading of a container can take some time, and if everything is already pickled into local environment, loading it from there will be much faster.
            else:
                logger.info("Listing blobs")
                blobs_list = container_client.list_blobs(
                    self.name_starts_with, self.include
                )
                for obj in blobs_list:
                    sanitized_file_name = obj.name.replace("/", "-")
                    download_file_path = os.path.join(temp_dir, sanitized_file_name)
                    logger.info(f"Start download of {obj.name}")
                    start_time = time.time()
                    blob_client = container_client.get_blob_client(obj)
                    stream = blob_client.download_blob()
                    with open(file=download_file_path, mode="wb") as download_file:
                        stream.readinto(download_file)
                    blob_meta[download_file_path] = blob_client.get_blob_properties()
                    end_time = time.time()
                    logger.info(
                        f"{obj.name} downloaded in {end_time - start_time} seconds."
                    )

            total_download_end_time = time.time()
            total_elapsed_time = math.ceil(
                total_download_end_time - total_download_start_time
            )
            logger.info(
                f"Downloading completed in approximately {total_elapsed_time // 60}min"
                f" {total_elapsed_time % 60}s."
            )
            logger.info("Document creation starting")

            def extract_blob_meta(file_path):
                meta: dict = blob_meta[file_path]

                creation_time = meta.get("creation_time")
                creation_time = (
                    creation_time.strftime("%Y-%m-%d") if creation_time else None
                )

                last_modified = meta.get("last_modified")
                last_modified = (
                    last_modified.strftime("%Y-%m-%d") if last_modified else None
                )

                last_accessed_on = meta.get("last_accessed_on")
                last_accessed_on = (
                    last_accessed_on.strftime("%Y-%m-%d") if last_accessed_on else None
                )

                extracted_meta = {
                    "file_name": meta.get("name"),
                    "file_type": meta.get("content_settings", {}).get("content_type"),
                    "file_size": meta.get("size"),
                    "creation_date": creation_time,
                    "last_modified_date": last_modified,
                    "last_accessed_date": last_accessed_on,
                    "container": meta.get("container"),
                }
                extracted_meta.update(meta.get("metadata") or {})
                extracted_meta.update(meta.get("tags") or {})
                return extracted_meta

            loader = SimpleDirectoryReader(
                temp_dir,
                file_extractor=self.file_extractor,
                file_metadata=extract_blob_meta,
            )

            return loader.load_data()
