from llama_index.core.schema import Document
from llama_index.core.tools.tool_spec.base import BaseToolSpec

from box_sdk_gen import BoxClient

from llama_index.readers.box.BoxAPI.box_api import (
    box_check_connection,
    get_box_files_details,
    get_files_ai_extract_data,
    add_extra_header_to_box_client,
)

from llama_index.readers.box.BoxAPI.box_llama_adaptors import box_file_to_llama_document


class BoxAIExtractToolSpec(BaseToolSpec):
    """
    Extracts AI generated content from a Box file.

    Args:
        box_client (BoxClient): A BoxClient instance for interacting with Box API.

    Attributes:
        spec_functions (list): A list of supported functions.
        _box_client (BoxClient): An instance of BoxClient for interacting with Box API.

    Methods:
        ai_extract(file_id, ai_prompt): Extracts AI generated content from a Box file.

    Args:
        file_id (str): The ID of the Box file.
        ai_prompt (str): The AI prompt to use for extraction.

    Returns:
        Document: A Document object containing the extracted AI content.

    """

    spec_functions = ["ai_extract"]

    _box_client: BoxClient

    def __init__(self, box_client: BoxClient) -> None:
        """
        Initializes the BoxAIExtractToolSpec with a BoxClient instance.

        Args:
            box_client (BoxClient): The BoxClient instance to use for interacting with the Box API.

        """
        self._box_client = add_extra_header_to_box_client(box_client)

    def ai_extract(
        self,
        file_id: str,
        ai_prompt: str,
    ) -> Document:
        """
        Extracts AI generated content from a Box file using the provided AI prompt.

        Args:
            file_id (str): The ID of the Box file to process.
            ai_prompt (str): The AI prompt to use for content extraction.

        Returns:
            Document: A Document object containing the extracted AI content,
            including metadata about the original Box file.

        """
        # Connect to Box
        box_check_connection(self._box_client)

        # get payload information
        box_file = get_box_files_details(
            box_client=self._box_client, file_ids=[file_id]
        )[0]

        box_file = get_files_ai_extract_data(
            box_client=self._box_client,
            box_files=[box_file],
            ai_prompt=ai_prompt,
        )[0]

        doc = box_file_to_llama_document(box_file)
        doc.text = box_file.ai_response if box_file.ai_response else ""
        doc.metadata["ai_prompt"] = box_file.ai_prompt
        doc.metadata["ai_response"] = box_file.ai_response

        return doc
