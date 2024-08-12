from llama_index.core.schema import Document
from llama_index.core.tools.tool_spec.base import BaseToolSpec

from box_sdk_gen import (
    BoxClient,
)

from llama_index.readers.box.BoxAPI.box_api import (
    box_check_connection,
    get_box_files_payload,
    get_text_representation,
)


class BoxTextExtractToolSpec(BaseToolSpec):
    """Box Text Extraction Tool Specification.

    This class provides a specification for extracting text content from Box files
    and creating Document objects. It leverages the Box API to retrieve the
    text representation (if available) of specified Box files.

    Attributes:
        _box_client (BoxClient): An instance of the Box client for interacting
            with the Box API.
    """

    spec_functions = ["extract"]
    _box_client: BoxClient

    def __init__(self, box_client: BoxClient) -> None:
        """
        Initializes the Box Text Extraction Tool Specification with the
        provided Box client instance.

        Args:
            box_client (BoxClient): The Box client instance.
        """
        self._box_client = box_client

    def extract(
        self,
        file_id: str,
    ) -> Document:
        """
        Extracts text content from Box files and creates Document objects.

        This method utilizes the Box API to retrieve the text representation
        (if available) of the specified Box files. It then creates Document
        objects containing the extracted text and file metadata.

        Args:
            file_id (str): A of Box file ID
                to extract text from.

        Returns:
            List[Document]: A list of Document objects containing the extracted
                text content and file metadata.
        """
        # Connect to Box
        box_check_connection(self._box_client)

        # get payload information
        payloads = get_box_files_payload(
            box_client=self._box_client, file_ids=[file_id]
        )

        payloads = get_text_representation(
            box_client=self._box_client,
            payloads=payloads,
        )

        for payload in payloads:
            file = payload.resource_info

            # create a document
            doc = Document(
                extra_info=file.to_dict(),
                metadata=file.to_dict(),
                text=payload.text_representation if payload.text_representation else "",
            )

        return doc
