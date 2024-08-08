from llama_index.core.schema import Document
from llama_index.core.tools.tool_spec.base import BaseToolSpec

from box_sdk_gen import (
    BoxClient,
)

from llama_index.readers.box.BoxAPI.box_api import (
    box_check_connection,
    get_box_files_payload,
    get_files_ai_prompt,
)


class BoxAIPromptToolSpec(BaseToolSpec):
    spec_functions = ["ai_prompt"]

    _box_client: BoxClient

    def __init__(self, box_client: BoxClient) -> None:
        self._box_client = box_client

    def ai_prompt(
        self,
        file_id: str,
        ai_prompt: str,
    ) -> Document:
        # Connect to Box
        box_check_connection(self._box_client)

        # get payload information
        payloads = get_box_files_payload(
            box_client=self._box_client, file_ids=[file_id]
        )

        payloads = get_files_ai_prompt(
            box_client=self._box_client,
            payloads=payloads,
            ai_prompt=ai_prompt,
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

        return doc
