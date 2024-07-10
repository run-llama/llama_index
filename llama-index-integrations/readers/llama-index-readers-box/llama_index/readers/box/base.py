import logging
import time
from typing import List, Optional, Dict, Any, Literal
from pathlib import Path
import tempfile
import io

from llama_index.core.readers import FileSystemReaderMixin, SimpleDirectoryReader
from llama_index.core.schema import Document
from llama_index.core.readers.base import (
    BasePydanticReader,
    ResourcesReaderMixin,
)
from llama_index.core.bridge.pydantic import PrivateAttr

from box_sdk_gen import BoxClient, BoxDeveloperTokenAuth
from box_sdk_gen.box.errors import BoxAPIError


logger = logging.getLogger(__name__)


class BoxReader(BasePydanticReader, ResourcesReaderMixin, FileSystemReaderMixin):
    """
    A reader class for interacting with Box, a cloud content management and file sharing service.

    This class provides functionality to authenticate with Box, list folder contents,
    and read files from Box, supporting both small and large files.
    """

    is_remote: bool = True

    auth_method: Literal["developer_token"]

    developer_token: Optional[str] = None

    folder_id: Optional[str] = None
    recursive: bool = True
    file_extractor: Optional[Dict[str, Any]] = None
    num_files_limit: Optional[int] = None
    chunk_size: int = 5 * 1024 * 1024  # 5 MB chunk size for streaming
    large_file_threshold: int = 10 * 1024 * 1024  # 10 MB threshold for large files

    _client: Any = PrivateAttr()

    def __init__(
        self,
        auth_method: Literal["developer_token"],
        developer_token: Optional[str] = None,
        folder_id: Optional[str] = None,
        recursive: bool = True,
        file_extractor: Optional[Dict[str, Any]] = None,
        num_files_limit: Optional[int] = None,
        chunk_size: int = 1024 * 1024,
        large_file_threshold: int = 10 * 1024 * 1024,
    ) -> None:
        """
        Initialize the BoxReader with the given parameters.

        Args:
            auth_method (Literal["developer_token"]): The authentication method to use.
            developer_token (Optional[str]): The developer token for authentication.
            folder_id (Optional[str]): The ID of the folder to read from.
            recursive (bool): Whether to recursively read subfolders.
            file_extractor (Optional[Dict[str, Any]]): A dictionary of file extractors.
            num_files_limit (Optional[int]): The maximum number of files to read.
            chunk_size (int): The size of chunks when streaming large files.
            large_file_threshold (int): The threshold size for considering a file as large.
        """
        super().__init__(
            auth_method=auth_method,
            developer_token=developer_token,
            folder_id=folder_id,
            recursive=recursive,
            file_extractor=file_extractor,
            num_files_limit=num_files_limit,
        )
        self.chunk_size = chunk_size
        self.large_file_threshold = large_file_threshold
        self._client = self._get_client()

    @classmethod
    def class_name(cls) -> str:
        return "BoxReader"

    def _get_client(self) -> BoxClient:
        """
        Get an authenticated Box client.

        Returns:
            BoxClient: An authenticated Box client.

        Raises:
            ValueError: If the authentication method is unsupported or required parameters are missing.
        """
        try:
            if self.auth_method == "developer_token":
                if not self.developer_token:
                    raise ValueError(
                        "developer_token is required for DeveloperToken authentication."
                    )
                auth = BoxDeveloperTokenAuth(token=self.developer_token)
            else:
                raise ValueError(f"Unsupported auth method: {self.auth_method}")

            return BoxClient(auth=auth)
        except Exception as e:
            logger.error(f"Failed to initialize Box client: {e!s}")
            raise

    def _list_folder_contents(self, folder_id: str) -> List[Any]:
        """
        List the contents of a Box folder.

        Args:
            folder_id (str): The ID of the folder to list.

        Returns:
            List[Any]: A list of items in the folder.
        """
        items = []
        offset = 0
        # limit = 1000  # Box API limit
        max_retries = 3
        retry_delay = 5  # seconds

        while True:
            for attempt in range(max_retries):
                try:
                    logger.info(
                        f"Fetching items from folder {folder_id}, offset: {offset}"
                    )
                    folder_items = self._client.folders.get_folder_items(
                        folder_id=folder_id, offset=offset
                    )
                    new_items = folder_items.entries
                    items.extend(new_items)

                    # if len(new_items) < limit:
                    #     logger.info(f"Finished fetching items from folder {folder_id}")
                    #     return items

                    offset += len(new_items)

                    if self.num_files_limit and len(items) >= self.num_files_limit:
                        logger.info(f"Reached file limit of {self.num_files_limit}")
                        return items[: self.num_files_limit]
                    break
                except BoxAPIError as e:
                    if "token has expired" in e.message:
                        logger.error(e.message)
                        raise
                    if "rate limit exceeded" in e.message:
                        retry_after = e.response_info.headers.get("retry-after")
                        if retry_after:
                            retry_after = int(retry_after)
                        else:
                            retry_after = retry_delay
                        logger.warning(
                            f"Rate limit exceeded. Retrying after {retry_after} seconds."
                        )
                        if attempt < max_retries - 1:
                            time.sleep(retry_after)
                        else:
                            logger.error(
                                f"Failed to fetch folder contents after {max_retries} attempts"
                            )
                            raise
                except Exception as e:
                    logger.error(f"Unknown Error occurred: {e!s}")
                    raise
            return items

    def _process_item(self, item: Any) -> Optional[Document]:
        """
        Process a single item from Box.

        Args:
            item (Any): The item to process.

        Returns:
            Optional[Document]: A Document object if the item is successfully processed, None otherwise.
        """
        try:
            if item.type == "file":
                logger.info(f"Processing file: {item.name} (ID: {item.id})")

                if item.size > self.large_file_threshold:
                    logger.info(
                        f"Large file detected: {item.name}. Using streaming approach."
                    )
                    return self._process_large_file(item)
                else:
                    return self._process_small_file(item)
            return None
        except Exception as e:
            logger.error(f"Error processing item {item.id}: {e!s}")
            return None

    def _process_small_file(self, item: Any) -> Optional[Document]:
        """
        Process a small file from Box.

        Args:
            item (Any): The file item to process.

        Returns:
            Optional[Document]: A Document object if the file is successfully processed, None otherwise.
        """
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            file_content = self._client.downloads.download_file(file_id=item.id)
            content = file_content.read()
            try:
                temp_file.write(content)
                temp_file.flush()

                reader = SimpleDirectoryReader(
                    input_files=[temp_file.name], file_extractor=self.file_extractor
                )
                docs = reader.load_data()
            finally:
                Path(
                    temp_file.name
                ).unlink()  # Ensure file is deleted even if an error occurs

            if docs:
                doc = docs[0]
                doc.extra_info = self._get_item_metadata(item)
                return doc
        return None

    def _process_large_file(self, item: Any) -> Optional[Document]:
        """
        Process a large file from Box using streaming.

        Args:
            item (Any): The file item to process.

        Returns:
            Optional[Document]: A Document object if the file is successfully processed, None otherwise.
        """
        buffer = io.BytesIO()
        for chunk in self._stream_file(item.id):
            buffer.write(chunk)

        buffer.seek(0)
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            try:
                temp_file.write(buffer.getvalue())
                temp_file.flush()

                reader = SimpleDirectoryReader(
                    input_files=[temp_file.name], file_extractor=self.file_extractor
                )
                docs = reader.load_data()
            finally:
                Path(
                    temp_file.name
                ).unlink()  # Ensure file is deleted even if an error occurs

            if docs:
                doc = docs[0]
                doc.extra_info = self._get_item_metadata(item)
                return doc
        return None

    def _stream_file(self, file_id: str):
        """
        Stream a file from Box in chunks.

        Args:
            file_id (str): The ID of the file to stream.

        Yields:
            bytes: Chunks of the file content.
        """
        logger.info(f"Starting to stream file with ID: {file_id}")
        total_bytes = 0

        try:
            file_stream: io.BufferedIOBase = self._client.downloads.download_file(
                file_id=file_id
            )

            while True:
                chunk = file_stream.read(self.chunk_size)
                if not chunk:
                    break

                chunk_size = len(chunk)
                total_bytes += chunk_size

                logger.debug(f"Read chunk: size={chunk_size}")

                yield chunk

        except Exception as e:
            logger.exception(f"Unexpected error while streaming file {file_id}: {e}")
            raise
        finally:
            logger.info(
                f"Finished streaming file {file_id}. Total bytes: {total_bytes}"
            )

    def _get_item_metadata(self, item: Any) -> Dict[str, Any]:
        """
        Get metadata for a Box item.

        Args:
            item (Any): The Box item.

        Returns:
            Dict[str, Any]: A dictionary containing the item's metadata.
        """
        return {
            "file_id": item.id,
            "file_name": item.name,
            "file_size": item.size,
            "created_at": item.created_at,
            "modified_at": item.modified_at,
        }

    def load_data(self) -> List[Document]:
        """
        Load data from Box.

        Returns:
            List[Document]: A list of Document objects representing the loaded files.
        """
        documents = []
        folders_to_process = (
            [self.folder_id] if self.folder_id else ["0"]
        )  # "0" is the root folder in Box

        while folders_to_process:
            current_folder = folders_to_process.pop(0)
            logger.info(f"Processing folder: {current_folder}")
            try:
                items = self._list_folder_contents(current_folder)

                for item in items:
                    if item.type == "folder" and self.recursive:
                        folders_to_process.append(item.id)
                    elif item.type == "file":
                        file_details = self._client.files.get_file_by_id(
                            file_id=item.id
                        )
                        doc = self._process_item(file_details)
                        if doc:
                            documents.append(doc)

                    if self.num_files_limit and len(documents) >= self.num_files_limit:
                        logger.info(f"Reached file limit of {self.num_files_limit}")
                        return documents

            except Exception as e:
                logger.error(f"Error processing folder {current_folder}: {e!s}")

        logger.info(f"Finished loading data. Total documents: {len(documents)}")
        return documents

    def load_resource(self, resource_id: str) -> List[Document]:
        """
        Load a specific resource from Box.

        Args:
            resource_id (str): The ID of the resource to load.

        Returns:
            List[Document]: A list containing the Document object for the loaded resource.
        """
        try:
            logger.info(f"Loading resource: {resource_id}")
            item = self._client.files.get_file_by_id(file_id=resource_id)
            doc = self._process_item(item)
            return [doc] if doc else []
        except Exception as e:
            logger.error(f"Error loading resource {resource_id}: {e!s}")
            return []

    def get_resource_info(self, resource_id: str) -> Dict:
        """
        Get information about a specific resource from Box.

        Args:
            resource_id (str): The ID of the resource.

        Returns:
            Dict: A dictionary containing information about the resource.
        """
        try:
            logger.info(f"Getting resource info: {resource_id}")
            item = self._client.files.get_file_by_id(file_id=resource_id)
            info_dict = {
                "file_path": item.name,
                "file_size": item.size,
                "created_at": item.created_at,
                "last_modified_date": item.modified_at,
                "etag": item.etag,
                "content_hash": item.sha_1,
            }

            # Ignore None values
            return {
                meta_key: meta_value
                for meta_key, meta_value in info_dict.items()
                if meta_value is not None
            }
        except Exception as e:
            logger.error(f"Error getting resource info for {resource_id}: {e!s}")
            return {}

    def list_resources(self) -> List[str]:
        """
        List all resources (files) in the specified Box folder.

        Returns:
            List[str]: A list of resource IDs.
        """
        try:
            logger.info("Listing resources")
            items = self._list_folder_contents(self.folder_id or "0")
            return [item.id for item in items if item.type == "file"]
        except Exception as e:
            logger.error(f"Error listing resources: {e!s}")
            return []

    def read_file_content(self, input_file: Path, **kwargs) -> bytes:
        """
        Reads the content of a file from Box.

        Args:
            input_file (Path): Path object containing the file ID as a string.

        Returns:
            bytes: Content of the file as bytes.
        """
        try:
            try:
                file_id = input_file.name  # Extract the file ID from the Path object
            except Exception:
                file_id = str(input_file)
            logger.info(f"Reading file content for file ID: {file_id}")

            file_details = self._client.files.get_file_by_id(file_id=file_id)
            doc = self._process_item(file_details)
            logger.info(f"Successfully read content for file ID: {file_id}")
            return doc.text.encode("utf-8")

        except Exception as e:
            logger.error(f"Error reading file content for file ID {file_id}: {e!s}")
            return b""
