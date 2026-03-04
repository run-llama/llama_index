import logging
from typing import List, Optional

from llama_index.core.schema import Document
from llama_index.readers.box import BoxReaderBase
from llama_index.readers.box.BoxAPI.box_api import (
    box_check_connection,
    get_box_files_details,
    get_box_folder_files_details,
    get_ai_response_from_box_files,
)

from box_sdk_gen import BoxClient, File

from llama_index.readers.box.BoxAPI.box_llama_adaptors import box_file_to_llama_document


logger = logging.getLogger(__name__)


class BoxReaderAIPrompt(BoxReaderBase):
    """
    A reader class for loading data from Box files using a custom AI prompt.

    This class inherits from the `BaseReader` class and allows specifying a
    custom AI prompt for data extraction. It utilizes the provided BoxClient object
    to interact with the Box API and extracts data based on the prompt.

    Attributes:
        _box_client (BoxClient): An authenticated Box client object used
            for interacting with the Box API.

    """

    _box_client: BoxClient

    @classmethod
    def class_name(cls) -> str:
        return "BoxReaderAIPrompt"

    def __init__(self, box_client: BoxClient):
        super().__init__(box_client=box_client)

    # def load_data(self, *args: Any, **load_kwargs: Any) -> List[Document]:
    def load_data(
        self,
        ai_prompt: str,
        file_ids: Optional[List[str]] = None,
        folder_id: Optional[str] = None,
        is_recursive: bool = False,
        individual_document_prompt: bool = True,
    ) -> List[Document]:
        """
        Extracts data from Box files using a custom AI prompt and creates Document objects.

        This method utilizes a user-provided AI prompt to potentially extract
        more specific data from the Box files compared to pre-configured AI
        services like Box AI Extract. It then creates Document objects containing
        the extracted data along with file metadata.

        Args:
            ai_prompt (str): The custom AI prompt that specifies what data to
                extract from the files.
            file_ids (Optional[List[str]], optional): A list of Box file IDs
                to extract data from. If provided, folder_id is ignored.
                Defaults to None.
            folder_id (Optional[str], optional): The ID of the Box folder to
                extract data from. If provided, along with is_recursive set to
                True, retrieves data from sub-folders as well. Defaults to None.
            is_recursive (bool, optional): If True and folder_id is provided,
                extracts data from sub-folders within the specified folder.
                Defaults to False.
            individual_document_prompt (bool, optional): If True, applies the
                provided AI prompt to each document individually. If False,
                 all documents are used for context to the answer.
                 Defaults to True.

        Returns:
            List[Document]: A list of Document objects containing the extracted
                data and file metadata.

        """
        # Connect to Box
        box_check_connection(self._box_client)

        docs: List[Document] = []
        box_files: List[File] = []

        # get box files information
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

        box_files = get_ai_response_from_box_files(
            box_client=self._box_client,
            box_files=box_files,
            ai_prompt=ai_prompt,
            individual_document_prompt=individual_document_prompt,
        )

        for file in box_files:
            doc = box_file_to_llama_document(file)
            doc.text = file.ai_response if file.ai_response else ""
            doc.metadata["ai_prompt"] = file.ai_prompt
            doc.metadata["ai_response"] = file.ai_response if file.ai_response else ""
            docs.append(doc)
        return docs

    def load_resource(self, box_file_id: str, ai_prompt: str) -> List[Document]:
        """
        Load data from a specific resource.

        Args:
            resource (str): The resource identifier.

        Returns:
            List[Document]: A list of documents loaded from the resource.

        """
        return self.load_data(file_ids=[box_file_id], ai_prompt=ai_prompt)
