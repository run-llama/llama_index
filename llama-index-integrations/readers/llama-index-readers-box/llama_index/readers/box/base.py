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

from box_sdk_gen import BoxClient, BoxOAuth, OAuthConfig, BoxDeveloperTokenAuth


logger = logging.getLogger(__name__)


class BoxReader(BasePydanticReader, ResourcesReaderMixin, FileSystemReaderMixin):
    is_remote: bool = True

    auth_method: Literal["oauth2", "developer_token"]
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    authorization_code: Optional[str] = None

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
        auth_method: Literal["oauth2", "developer_token"],
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        authorization_code: Optional[str] = None,
        developer_token: Optional[str] = None,
        folder_id: Optional[str] = None,
        recursive: bool = True,
        file_extractor: Optional[Dict[str, Any]] = None,
        num_files_limit: Optional[int] = None,
        chunk_size: int = 1024 * 1024,
        large_file_threshold: int = 10 * 1024 * 1024,
    ) -> None:
        super().__init__(
            auth_method=auth_method,
            client_id=client_id,
            client_secret=client_secret,
            authorization_code=authorization_code,
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
        try:
            if self.auth_method == "oauth2":
                if not self.client_id:
                    raise ValueError("client_id is required for OAuth2 authentication.")
                if not self.client_secret:
                    raise ValueError(
                        "client_secret is required for OAuth2 authentication."
                    )
                if not self.authorization_code:
                    raise ValueError(
                        "authorization_code is required for OAuth2 authentication."
                    )
                auth_config = OAuthConfig(
                    client_id=self.client_id, client_secret=self.client_secret
                )
                auth = BoxOAuth(config=auth_config)
                auth.get_tokens_authorization_code_grant(
                    authorization_code=self.authorization_code
                )
            elif self.auth_method == "developer_token":
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
        items = []
        offset = 0
        limit = 1000  # Box API limit
        max_retries = 3
        retry_delay = 5  # seconds

        while True:
            for attempt in range(max_retries):
                try:
                    logger.info(
                        f"Fetching items from folder {folder_id}, offset: {offset}"
                    )
                    folder_items = self._client.folders.get_folder_items(
                        folder_id=folder_id, limit=limit, offset=offset
                    )
                    new_items = folder_items.entries
                    items.extend(new_items)

                    if len(new_items) < limit:
                        logger.info(f"Finished fetching items from folder {folder_id}")
                        return items

                    offset += limit

                    if self.num_files_limit and len(items) >= self.num_files_limit:
                        logger.info(f"Reached file limit of {self.num_files_limit}")
                        return items[: self.num_files_limit]

                    time.sleep(1)  # Rate limiting: wait 1 second between requests
                    break
                except Exception as e:
                    logger.warning(
                        f"Error fetching folder contents (attempt {attempt + 1}): {e!s}"
                    )
                    if "token has expired" in str(e):
                        logger.error(
                            "Authentication failed, provided token is expired, please provide a new one."
                        )
                        raise

                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                    else:
                        logger.error(
                            f"Failed to fetch folder contents after {max_retries} attempts"
                        )
                        raise

    def _process_item(self, item: Any) -> Optional[Document]:
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
        return {
            "file_id": item.id,
            "file_name": item.name,
            "file_size": item.size,
            "created_at": item.created_at,
            "modified_at": item.modified_at,
        }

    def load_data(self) -> List[Document]:
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
        try:
            logger.info(f"Loading resource: {resource_id}")
            item = self._client.files.get_file_by_id(file_id=resource_id)
            doc = self._process_item(item)
            return [doc] if doc else []
        except Exception as e:
            logger.error(f"Error loading resource {resource_id}: {e!s}")
            return []

    def get_resource_info(self, resource_id: str) -> Dict:
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
        try:
            logger.info("Listing resources")
            items = self._list_folder_contents(self.folder_id or "0")
            return [item.id for item in items if item.type == "file"]
        except Exception as e:
            logger.error(f"Error listing resources: {e!s}")
            return []

    def read_file_content(self, input_file: Path, **kwargs) -> bytes:
        logger.warning("read_file_content method is not implemented for BoxReader")
        return b""
