import logging
from typing import List, Optional

from llama_index.core.schema import Document
from llama_index.readers.box.BoxAPI.box_api import (
    box_check_connection,
    get_box_files_details,
    get_box_folder_files_details,
    get_text_representation,
)
from llama_index.readers.box.BoxAPI.box_llama_adaptors import box_file_to_llama_document
from llama_index.readers.box import BoxReaderBase

from box_sdk_gen import (
    BoxClient,
    File,
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

        docs: List[Document] = []
        box_files: List[File] = []

        # get Box files details
        if file_ids is not None:
            box_files.extend(
                get_box_files_details(box_client=self._box_client, file_ids=file_ids)
            )
        elif folder_id is not None:
            box_files.extend(
                get_box_folder_files_details(
                    box_client=self._box_client,
                    folder_id=folder_id,
                    is_recursive=is_recursive,
                )
            )

        box_files = get_text_representation(
            box_client=self._box_client,
            box_files=box_files,
        )

        for file in box_files:
            doc = box_file_to_llama_document(file)
            doc.text = file.text_representation if file.text_representation else ""
            docs.append(doc)
        return docs
