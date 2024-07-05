import logging
import os
import shutil
import tempfile
from typing import List, Optional, Dict, Any, Union

from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.readers.base import (
    BasePydanticReader,
    BaseReader,
)
from llama_index.core.schema import Document
from llama_index.core.bridge.pydantic import BaseModel, Field

from box_sdk_gen import BoxAPIError, BoxClient, ByteStream, File, CCGConfig, BoxCCGAuth

logger = logging.getLogger(__name__)


class _BoxResourcePayload(BaseModel):
    resource_info: Dict[str, Any]
    downloaded_file_path: Optional[str]


# TODO: Implement , ResourcesReaderMixin, FileSystemReaderMixin
class BoxReader(BasePydanticReader):
    box_config: CCGConfig
    file_extractor: Optional[Dict[str, Union[str, BaseReader]]] = Field(
        default=None, exclude=True
    )

    @classmethod
    def class_name(cls) -> str:
        return "BoxReader"

    def _get_box_client(self) -> BoxClient:
        # check what type of object the box_config is:
        if isinstance(self.box_config, CCGConfig):
            auth = BoxCCGAuth(self.box_config)
            if (
                self.box_config.user_id
                and self.box_config.user_id != "YOUR_BOX_CCG_USER_ID (optional)"
            ):
                auth.with_user_subject(self.box_config.user_id)
            return BoxClient(auth)

        raise ValueError("Box config is not a CCGConfig object")

    def load_data(
        self,
        folder_id: Optional[str] = None,
        file_ids: Optional[List[str]] = None,
    ) -> List[Document]:
        client = self._get_box_client()

        # Connect to Box
        try:
            me = client.users.get_user_me()
            logger.info(f"Connected to Box as user: {me.id} {me.name}({me.login})")
        except BoxAPIError as e:
            logger.error(
                f"An error occurred while connecting to Box: {e}", exc_info=True
            )
            raise

        # Get the files
        with tempfile.TemporaryDirectory() as temp_dir:
            if file_ids is not None:
                payloads = self._get_files(client, file_ids, temp_dir)
            elif folder_id is not None:
                payloads = self._get_folder(client, folder_id, temp_dir)
            else:
                payloads = self._get_folder(
                    client,
                    "0",
                    temp_dir,
                )
            file_name_to_metadata = {
                payload.downloaded_file_path: payload.resource_info
                for payload in payloads
            }

            def get_metadata(filename: str) -> Any:
                return file_name_to_metadata[filename]

            simple_loader = SimpleDirectoryReader(
                input_dir=temp_dir,
                file_metadata=get_metadata,
                file_extractor=self.file_extractor,
            )
            return simple_loader.load_data()

    def _get_files(
        self, client: BoxClient, file_ids: List[str], temp_dir: str
    ) -> List[_BoxResourcePayload]:
        payloads = []
        for file_id in file_ids:
            file = client.files.get_file_by_id(file_id)
            logger.info(f"Getting file: {file.id} {file.name} {file.type}")
            local_path = self._download_file_by_id(client, file, temp_dir)
            resource_info = file.to_dict()
            payloads.append(
                _BoxResourcePayload(
                    resource_info=resource_info,
                    downloaded_file_path=local_path,
                )
            )
        return payloads

    def _download_file_by_id(
        self, client: BoxClient, box_file: File, temp_dir: str
    ) -> str:
        # Save the downloaded file to the specified local directory.
        file_path = os.path.join(temp_dir, box_file.name)
        file_stream: ByteStream = client.downloads.download_file(box_file.id)
        with open(file_path, "wb") as file:
            shutil.copyfileobj(file_stream, file)

        return file_path

    def _get_folder(
        self, client: BoxClient, folder_id: str, temp_dir: str
    ) -> List[_BoxResourcePayload]:
        # Make sure folder exists
        folder = client.folders.get_folder_by_id(folder_id)
        logger.info(f"Getting files from folder: {folder.id} {folder.name}")

        # Get the items
        items = client.folders.get_folder_items(folder_id, limit=10000).entries
        payloads = []
        for item in items:
            logger.info(f"Item: {item.id} {item.name} {item.type}")
            if item.type == "file":
                payloads.extend(self._get_files(client, [item.id], temp_dir))
            if item.type == "folder":
                logger.info(f"Skipping folder: {item.id} {item.name}")
                # TODO: Implement Box recursive folder download
        return payloads
