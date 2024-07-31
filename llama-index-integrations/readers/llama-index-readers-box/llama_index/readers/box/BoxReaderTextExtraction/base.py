import logging
from typing import List, Optional

from llama_index.core.schema import Document
from llama_index.readers.box.BoxAPI.box_api import (
    _BoxResourcePayload,
    box_check_connection,
    get_box_files_payload,
    get_box_folder_payload,
    get_text_representation,
)
from llama_index.readers.box import BoxReaderBase

from box_sdk_gen import (
    BoxClient,
)


logger = logging.getLogger(__name__)


class BoxReaderTextExtraction(BoxReaderBase):
    """
    A reader class for loading text content from Box files.

    This class inherits from the `BaseReader` class and specializes in
    extracting plain text content from Box files. It utilizes the provided
    BoxClient object to interact with the Box API and retrieves the text
    representation of the files.

    Attributes:
        _box_client (BoxClient): An authenticated Box client object used
            for interacting with the Box API.
    """

    @classmethod
    def class_name(cls) -> str:
        return "BoxReaderTextExtraction"

    def __init__(self, box_client: BoxClient):
        super().__init__(box_client=box_client)

    # def load_data(self, *args: Any, **load_kwargs: Any) -> List[Document]:
    def load_data(
        self,
        file_ids: Optional[List[str]] = None,
        folder_id: Optional[str] = None,
        is_recursive: bool = False,
    ) -> List[Document]:
        """
        Extracts text content from Box files and creates Document objects.

        This method utilizes the Box API to retrieve the text representation
        (if available) of the specified Box files. It then creates Document
        objects containing the extracted text and file metadata.

        Args:
            file_ids (Optional[List[str]], optional): A list of Box file IDs
                to extract text from. If provided, folder_id is ignored.
                Defaults to None.
            folder_id (Optional[str], optional): The ID of the Box folder to
                extract text from. If provided, along with is_recursive set to
                True, retrieves data from sub-folders as well. Defaults to None.
            is_recursive (bool, optional): If True and folder_id is provided,
                extracts text from sub-folders within the specified folder.
                Defaults to False.

        Returns:
            List[Document]: A list of Document objects containing the extracted
                text content and file metadata.
        """
        # Connect to Box
        box_check_connection(self._box_client)

        docs = []
        payloads: List[_BoxResourcePayload] = []
        # get payload information
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

        payloads = get_text_representation(
            box_client=self._box_client,
            payloads=payloads,
        )

        for payload in payloads:
            file = payload.resource_info

            # create a document
            doc = Document(
                # id=file.id,
                extra_info=file.to_dict(),
                metadata=file.to_dict(),
                text=payload.text_representation if payload.text_representation else "",
            )
            docs.append(doc)
        return docs
