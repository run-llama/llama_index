import os
import shutil
from typing import List, Optional
import logging
from box_sdk_gen import BoxAPIError, BoxClient, File, ByteStream
from box_sdk_gen.managers.ai import CreateAiAskMode, CreateAiAskItems

logger = logging.getLogger(__name__)


class _BoxResourcePayload:
    resource_info: Optional[File]
    ai_prompt: Optional[str]
    ai_response: Optional[str]
    downloaded_file_path: Optional[str]

    def __init__(self, resource_info: File) -> None:
        self.resource_info = resource_info


def get_box_files_payload(
    box_client: BoxClient, file_ids: List[str]
) -> List[_BoxResourcePayload]:
    payloads = []
    for file_id in file_ids:
        try:
            file = box_client.files.get_file_by_id(file_id)
        except BoxAPIError as e:
            logger.error(
                f"An error occurred while getting file: {e.message}", exc_info=True
            )
            payloads.append(
                _BoxResourcePayload(
                    resource_info=e.message,
                )
            )
        logger.info(f"Getting file: {file.id} {file.name} {file.type}")
        payloads.append(
            _BoxResourcePayload(
                resource_info=file,
            )
        )
    return payloads


def get_box_folder_payload(
    box_client: BoxClient, folder_id: str, is_recursive: bool = False
) -> List[_BoxResourcePayload]:
    payloads = []
    try:
        folder = box_client.folders.get_folder_by_id(folder_id)
    except BoxAPIError as e:
        logger.error(
            f"An error occurred while getting folder: {e.message}", exc_info=True
        )
        return payloads

    for item in box_client.folders.get_folder_items(folder.id).entries:
        if item.type == "file":
            payloads.extend(get_box_files_payload(box_client, [item.id]))
        elif item.type == "folder":
            if is_recursive:
                payloads.extend(
                    get_box_folder_payload(box_client, item.id, is_recursive)
                )
    return payloads


def download_file_by_id(box_client: BoxClient, box_file: File, temp_dir: str) -> str:
    # Save the downloaded file to the specified local directory.
    file_path = os.path.join(temp_dir, box_file.name)

    try:
        file_stream: ByteStream = box_client.downloads.download_file(box_file.id)
    except BoxAPIError as e:
        logger.error(f"An error occurred while downloading file: {e}", exc_info=True)
        return e.message

    logger.info(f"Downloading file: {box_file.id} {box_file.name} ")
    with open(file_path, "wb") as file:
        shutil.copyfileobj(file_stream, file)

    return file_path


def get_files_ai_prompt(
    box_client: BoxClient,
    payloads: List[_BoxResourcePayload],
    ai_prompt: str,
    individual_document_prompt: bool = True,
) -> List[_BoxResourcePayload]:
    if individual_document_prompt:
        mode = CreateAiAskMode.SINGLE_ITEM_QA
        for payload in payloads:
            file = payload.resource_info
            ask_item = CreateAiAskItems(file.id)
            logger.info(f"Getting AI prompt for file: {file.id} {file.name}")

            # get the AI prompt for the file
            try:
                ai_response = box_client.ai.create_ai_ask(
                    mode=mode, prompt=ai_prompt, items=[ask_item]
                )
            except BoxAPIError as e:
                logger.error(
                    f"An error occurred while getting AI response for file: {e}",
                    exc_info=True,
                )
                payload.ai_prompt = ai_prompt
                if e.response_info.status_code == 400:
                    payload.ai_response = "File type not supported by Box AI"
                else:
                    payload.ai_response = e.message
                continue
            payload.ai_prompt = ai_prompt
            payload.ai_response = ai_response.answer

    else:
        mode = CreateAiAskMode.MULTIPLE_ITEM_QA
        file_ids = [CreateAiAskItems(payload.resource_info.id) for payload in payloads]

        # get the AI prompt for the file
        ai_response = box_client.ai.create_ai_ask(
            mode=mode, prompt=ai_prompt, items=file_ids
        )
        for payload in payloads:
            payload.ai_prompt = ai_prompt
            payload.ai_response = ai_response.answer

    return payloads
