import logging
from typing import List, Optional

from llama_index.core.readers.base import (
    BasePydanticReader,
)
from llama_index.core.schema import Document

from llama_index.readers.box.box_client_ccg import reader_box_client_ccg

from box_sdk_gen import BoxAPIError, BoxClient

logger = logging.getLogger(__name__)


class BoxReader(BasePydanticReader):
    box_client_id: str
    box_client_secret: str
    box_enterprise_id: str
    box_user_id: str = None
    # client: BoxClient

    @classmethod
    def class_name(cls) -> str:
        return "BoxReader"

    def load_data(
        self,
        folder_id: Optional[str] = None,
        file_ids: Optional[List[str]] = None,
    ) -> List[Document]:
        client = reader_box_client_ccg(
            self.box_client_id,
            self.box_client_secret,
            self.box_enterprise_id,
            self.box_user_id,
        )

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
        if file_ids is not None:
            return self._get_files(client, file_ids)
        elif folder_id is not None:
            return self._get_folder(client, folder_id)
        else:
            return self._get_folder(client, "0")

    def _get_folder(self, client: BoxClient, folder_id: str) -> List[Document]:
        # Make sure folder exists
        folder = client.folders.get_folder_by_id(folder_id)
        logger.info(f"Getting files from folder: {folder.id} {folder.name}")

        # Get the items
        items = client.folders.get_folder_items(folder_id, limit=10000)
        for item in items:
            doc = Document()
            # logger.info(f"Item: {item.id} {item.name} {item.type}")

        # Get the files
        files = client.folders.get_folder_items(folder_id, limit=1000)
        return self._get_files(client, [f.id for f in files])
