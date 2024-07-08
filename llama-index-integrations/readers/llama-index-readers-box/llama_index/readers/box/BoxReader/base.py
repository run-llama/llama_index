import logging
import tempfile
from typing import List, Optional, Dict, Any, Union

from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.readers.base import (
    BaseReader,
)
from llama_index.core.schema import Document
from llama_index.core.bridge.pydantic import Field

from llama_index.readers.box.BoxAPI.box_api import (
    _BoxResourcePayload,
    get_box_files_payload,
    get_box_folder_payload,
    download_file_by_id,
)

from box_sdk_gen import (
    BoxAPIError,
    BoxClient,
)

logger = logging.getLogger(__name__)


# TODO: Implement , ResourcesReaderMixin, FileSystemReaderMixin
class BoxReader(BaseReader):
    _box_client: BoxClient
    file_extractor: Optional[Dict[str, Union[str, BaseReader]]] = Field(
        default=None, exclude=True
    )

    @classmethod
    def class_name(cls) -> str:
        return "BoxReader"

    def __init__(
        self,
        box_client: BoxClient,
        file_extractor: Optional[Dict[str, Union[str, BaseReader]]] = None,
    ):
        self._box_client = box_client
        self.file_extractor = file_extractor

    def load_data(
        self,
        folder_id: Optional[str] = None,
        file_ids: Optional[List[str]] = None,
        is_recursive: bool = False,
    ) -> List[Document]:
        # Connect to Box
        try:
            me = self._box_client.users.get_user_me()
            logger.info(f"Connected to Box as user: {me.id} {me.name}({me.login})")
        except BoxAPIError as e:
            logger.error(
                f"An error occurred while connecting to Box: {e}", exc_info=True
            )
            raise

        # Get the file resources
        payloads = []
        if file_ids is not None:
            payloads.extend(
                get_box_files_payload(box_client=self._box_client, file_ids=file_ids)
            )
        elif folder_id is not None:
            payloads.extend(
                get_box_folder_payload(
                    box_client=self._box_client,
                    folder_id=folder_id,
                    is_recursive=is_recursive,
                )
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            payloads = self._download_files(payloads, temp_dir)

            file_name_to_metadata = {
                payload.downloaded_file_path: payload.resource_info.to_dict()
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

    def _download_files(
        self, payloads: List[_BoxResourcePayload], temp_dir: str
    ) -> List[_BoxResourcePayload]:
        for payload in payloads:
            file = payload.resource_info
            local_path = download_file_by_id(
                box_client=self._box_client, box_file=file, temp_dir=temp_dir
            )
            payload.downloaded_file_path = local_path
        return payloads
