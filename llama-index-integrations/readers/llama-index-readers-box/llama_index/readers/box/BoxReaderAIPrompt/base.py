import logging
from typing import List

from llama_index.core.readers.base import (
    BaseReader,
)
from llama_index.core.schema import Document

from box_sdk_gen import (
    BoxAPIError,
    BoxClient,
    File,
)

from box_sdk_gen.managers.ai import CreateAiAskMode, CreateAiAskItems

logger = logging.getLogger(__name__)


class _BoxResourcePayload:
    resource_info: File
    ai_prompt: str
    ai_response: str

    def __init__(self, resource_info: File) -> None:
        self.resource_info = resource_info


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
        file_ids: List[str],
        ai_prompt: str,
        individual_document_prompt: bool = False,
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
            payloads = self._get_files_payload(file_ids=file_ids)

        payloads = self._get_files_ai_prompt(
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

    def _get_files_payload(self, file_ids: List[str]) -> List[_BoxResourcePayload]:
        payloads = []
        for file_id in file_ids:
            file = self._box_client.files.get_file_by_id(file_id)
            logger.info(f"Getting file: {file.id} {file.name} {file.type}")
            payloads.append(
                _BoxResourcePayload(
                    resource_info=file,
                )
            )
        return payloads

    def _get_files_ai_prompt(
        self,
        payloads: List[_BoxResourcePayload],
        ai_prompt: str,
        individual_document_prompt: bool = True,
    ) -> List[_BoxResourcePayload]:
        if individual_document_prompt:
            mode = CreateAiAskMode.SINGLE_ITEM_QA
            for payload in payloads:
                file = payload.resource_info
                logger.info(f"Getting AI prompt for file: {file.id} {file.name}")

                # get the AI prompt for the file
                ai_response = self._box_client.ai.create_ai_ask(
                    mode=mode, prompt=ai_prompt, items=[file]
                )
                payload.ai_prompt = ai_prompt
                payload.ai_response = ai_response.answer
        else:
            mode = CreateAiAskMode.MULTIPLE_ITEM_QA
            file_ids = [
                CreateAiAskItems(payload.resource_info.id) for payload in payloads
            ]

            # get the AI prompt for the file
            ai_response = self._box_client.ai.create_ai_ask(
                mode=mode, prompt=ai_prompt, items=file_ids
            )
            for payload in payloads:
                payload.ai_prompt = ai_prompt
                payload.ai_response = ai_response.answer

        return payloads
