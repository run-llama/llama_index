import logging
from typing import List, Optional

from llama_index.core.schema import Document
from llama_index.readers.box import BoxReaderBase
from llama_index.readers.box.BoxAPI.box_api import (
    _BoxResourcePayload,
    box_check_connection,
    get_box_files_payload,
    get_box_folder_payload,
    get_files_ai_prompt,
)

from box_sdk_gen import (
    BoxClient,
)


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

    def load_resource(self, box_file_id: str, ai_prompt: str) -> List[Document]:
        """
        Load data from a specific resource.

        Args:
            resource (str): The resource identifier.

        Returns:
            List[Document]: A list of documents loaded from the resource.
        """
        return self.load_data(file_ids=[box_file_id], ai_prompt=ai_prompt)
