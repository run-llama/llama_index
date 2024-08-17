from llama_index.core.schema import Document
from llama_index.core.tools.tool_spec.base import BaseToolSpec

from box_sdk_gen import (
    BoxClient,
)

from llama_index.readers.box.BoxAPI.box_api import (
    box_check_connection,
    get_box_files_details,
    get_ai_response_from_box_files,
    add_extra_header_to_box_client,
)

from llama_index.readers.box.BoxAPI.box_llama_adaptors import box_file_to_llama_document


class BoxAIPromptToolSpec(BaseToolSpec):
    """
    Generates AI prompts based on a Box file.

    Args:
        box_client (BoxClient): A BoxClient instance for interacting with Box API.

    Attributes:
        spec_functions (list): A list of supported functions.
        _box_client (BoxClient): An instance of BoxClient for interacting with Box API.

    Methods:
        ai_prompt(file_id, ai_prompt): Generates an AI prompt based on a Box file.

    Args:
        file_id (str): The ID of the Box file.
        ai_prompt (str): The base AI prompt to use.

    Returns:
        Document: A Document object containing the generated AI prompt.
    """

    spec_functions = ["ai_prompt"]

    _box_client: BoxClient

    def __init__(self, box_client: BoxClient) -> None:
        """
        Initializes the BoxAIPromptToolSpec with a BoxClient instance.

        Args:
            box_client (BoxClient): The BoxClient instance to use for interacting with the Box API.
        """
        self._box_client = add_extra_header_to_box_client(box_client)

    def ai_prompt(
        self,
        file_id: str,
        ai_prompt: str,
    ) -> Document:
        """
        Generates an AI prompt based on a Box file.

        Retrieves the specified Box file, constructs an AI prompt using the provided base prompt,
        and returns a Document object containing the generated prompt and file metadata.

        Args:
            file_id (str): The ID of the Box file to process.
            ai_prompt (str): The base AI prompt to use as a template.

        Returns:
            Document: A Document object containing the generated AI prompt and file metadata.
        """
        # Connect to Box
        box_check_connection(self._box_client)

        # get box files information
        box_file = get_box_files_details(
            box_client=self._box_client, file_ids=[file_id]
        )[0]

        box_file = get_ai_response_from_box_files(
            box_client=self._box_client,
            box_files=[box_file],
            ai_prompt=ai_prompt,
        )[0]

        doc = box_file_to_llama_document(box_file)
        doc.text = box_file.ai_response if box_file.ai_response else ""
        doc.metadata["ai_prompt"] = box_file.ai_prompt
        doc.metadata["ai_response"] = (
            box_file.ai_response if box_file.ai_response else ""
        )

        return doc
