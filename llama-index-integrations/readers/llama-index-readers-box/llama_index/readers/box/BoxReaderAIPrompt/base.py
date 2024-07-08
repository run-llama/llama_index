import logging
from typing import List, Optional

from llama_index.core.readers.base import (
    BaseReader,
)
from llama_index.core.schema import Document
from llama_index.readers.box.BoxAPI.box_api import (
    get_box_files_payload,
    get_box_folder_payload,
    get_files_ai_prompt,
)

from box_sdk_gen import (
    BoxAPIError,
    BoxClient,
)


logger = logging.getLogger(__name__)


class BoxReaderAIPrompt(BaseReader):
    _box_client: BoxClient

    @classmethod
    def class_name(cls) -> str:
        return "BoxReaderAIPrompt"

    def __init__(self, box_client: BoxClient):
        self._box_client = box_client

    # def load_data(self, *args: Any, **load_kwargs: Any) -> List[Document]:
    def load_data(
        self,
        ai_prompt: str,
        file_ids: Optional[List[str]] = None,
        folder_id: Optional[str] = None,
        is_recursive: bool = False,
        individual_document_prompt: bool = True,
    ) -> List[Document]:
        # check if the box client is authenticated
        try:
            me = self._box_client.users.get_user_me()
        except BoxAPIError as e:
            logger.error(
                f"An error occurred while connecting to Box: {e}", exc_info=True
            )
            raise

        # return super().load_data(*args, **load_kwargs)

        docs = []

        # get payload information
        if file_ids is not None:
            payloads = get_box_files_payload(
                box_client=self._box_client, file_ids=file_ids
            )
        elif folder_id is not None:
            payloads = get_box_folder_payload(
                box_client=self._box_client,
                folder_id=folder_id,
                is_recursive=is_recursive,
            )

        payloads = get_files_ai_prompt(
            box_client=self._box_client,
            payloads=payloads,
            ai_prompt=ai_prompt,
            individual_document_prompt=individual_document_prompt,
        )

        for payload in payloads:
            file = payload.resource_info
            ai_response = payload.ai_response

            # create a document
            doc = Document(
                # id=file.id,
                extra_info=file.to_dict(),
                metadata=file.to_dict(),
                text=ai_response,
            )
            docs.append(doc)
        return docs
