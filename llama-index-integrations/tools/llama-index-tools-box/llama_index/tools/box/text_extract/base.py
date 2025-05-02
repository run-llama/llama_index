from llama_index.core.schema import Document
from llama_index.core.tools.tool_spec.base import BaseToolSpec

from box_sdk_gen import (
    BoxClient,
)

from llama_index.readers.box.BoxAPI.box_api import (
    box_check_connection,
    get_box_files_details,
    get_text_representation,
    add_extra_header_to_box_client,
)

from llama_index.readers.box.BoxAPI.box_llama_adaptors import box_file_to_llama_document


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
        self._box_client = add_extra_header_to_box_client(box_client)

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
        box_file = get_box_files_details(
            box_client=self._box_client, file_ids=[file_id]
        )[0]

        box_file = get_text_representation(
            box_client=self._box_client,
            box_files=[box_file],
        )[0]

        doc = box_file_to_llama_document(box_file)
        doc.text = box_file.text_representation if box_file.text_representation else ""
        return doc
