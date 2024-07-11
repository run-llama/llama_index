import os
import shutil
from typing import List, Optional, Dict
import logging
import requests
from box_sdk_gen import (
    BoxAPIError,
    BoxClient,
    File,
    ByteStream,
    BoxSDKError,
)
from box_sdk_gen.managers.search import (
    SearchForContentScope,
    SearchForContentContentTypes,
    SearchForContentType,
    SearchResults,
)
from box_sdk_gen.managers.ai import CreateAiAskMode, CreateAiAskItems

from llama_index.readers.box.BoxAPI.box_ai_extract_beta import (
    AiExtractManager,
    CreateAiExtractItems,
)

logger = logging.getLogger(__name__)


class _BoxResourcePayload:
    resource_info: Optional[File]
    ai_prompt: Optional[str]
    ai_response: Optional[str]
    downloaded_file_path: Optional[str]
    text_representation: Optional[str]

    def __init__(self, resource_info: File) -> None:
        self.resource_info = resource_info


def box_check_connection(box_client: BoxClient) -> None:
    """
    Checks if the Box client is connected to the Box API.

    Args:
        box_client (BoxClient): A Box client object.

    Returns:
        bool: True if the Box client is connected to the Box API, False otherwise.
    """
    try:
        box_client.users.get_user_me()
    except BoxAPIError as e:
        logger.error(f"An error occurred while checking connection: {e.message}")
        raise
    return True


def get_box_files_payload(
    box_client: BoxClient, file_ids: List[str]
) -> List[_BoxResourcePayload]:
    """
    This function retrieves payloads for a list of Box files.

    Args:
        box_client (BoxClient): A Box client object.
        file_ids (List[str]): A list of Box file IDs.

    Returns:
        List[_BoxResourcePayload]: A list of _BoxResourcePayload objects.
            - If a file is retrieved successfully, the resource_info attribute
              will contain the corresponding Box file object.
            - If an error occurs while retrieving a file, the resource_info
              attribute will contain the error message.

    Raises:
        BoxAPIError: If an error occurs while interacting with the Box API.
    """
    payloads: List[_BoxResourcePayload] = []
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
    """
    This function retrieves payloads for all files within a Box folder,
    optionally including files from sub-folders.

    Args:
        box_client (BoxClient): A Box client object.
        folder_id (str): The ID of the Box folder.
        is_recursive (bool, optional): If True, retrieves payloads for
            files within sub-folders as well. Defaults to False.

    Returns:
        List[_BoxResourcePayload]: A list of _BoxResourcePayload objects
            containing information about files within the folder
            (and sub-folders if is_recursive is True).
            - If a file is retrieved successfully, the resource_info attribute
              will contain the corresponding Box file object.
            - If an error occurs while retrieving a file or folder,
              the resource_info attribute will contain the error message.

    Raises:
        BoxAPIError: If an error occurs while interacting with the Box API.
    """
    payloads: List[_BoxResourcePayload] = []
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
    """
    Downloads a Box file to the specified temporary directory.

    Args:
        box_client (BoxClient): An authenticated Box client object.
        box_file (BoxFile): The Box file object to download.
        temp_dir (str): The path to the temporary directory where the file
            will be downloaded.

    Returns:
        str: The path to the downloaded file on success.
        If an error occurs during download, returns the error message.

    Raises:
        BoxAPIError: If an error occurs while interacting with the Box API.
    """
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
    """
    Gets AI prompts and responses for a list of Box files.

    Args:
        box_client (BoxClient): An authenticated Box client object.
        payloads (List[_BoxResourcePayload]): A list of _BoxResourcePayload objects
            containing information about Box files.
        ai_prompt (str): The AI prompt to use for generating responses.
        individual_document_prompt (bool, optional): If True, generates an
            individual AI prompt and response for each file in payloads.
            If False, generates a single prompt and response for all files
            combined. Defaults to True.

    Returns:
        List[_BoxResourcePayload]: The updated list of _BoxResourcePayload objects
            with the following attributes added:
            - ai_prompt (str): The AI prompt used for the file.
            - ai_response (str): The AI response generated for the file
              (may be an error message).

    Raises:
        BoxAPIError: If an error occurs while interacting with the Box API.
    """
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


def _do_request(box_client: BoxClient, url: str):
    """
    Performs a GET request to a Box API endpoint using the provided Box client.

    This is an internal helper function and should not be called directly.

    Args:
        box_client (BoxClient): An authenticated Box client object.
        url (str): The URL of the Box API endpoint to make the request to.

    Returns:
        bytes: The content of the response from the Box API.

    Raises:
        BoxSDKError: If an error occurs while retrieving the access token.
        requests.exceptions.RequestException: If the request fails (e.g., network error,
                                             4XX or 5XX status code).
    """
    try:
        access_token = box_client.auth.retrieve_token().access_token
    except BoxSDKError as e:
        logger.error(f"Unable to retrieve access token: {e.message}", exc_info=True)
        raise

    resp = requests.get(url, headers={"Authorization": f"Bearer {access_token}"})
    resp.raise_for_status()
    return resp.content


def get_text_representation(
    box_client: BoxClient, payloads: List[_BoxResourcePayload], token_limit: int = 10000
) -> List[_BoxResourcePayload]:
    """
    Retrieves and stores the text representation for a list of Box files.

    This function attempts to retrieve the pre-generated extracted text for each
    file in the payloads list. If the extracted text is not available or needs
    generation, it initiates the generation process and stores a placeholder
    until the text is ready.

    Args:
        box_client (BoxClient): An authenticated Box client object.
        payloads (List[_BoxResourcePayload]): A list of _BoxResourcePayload objects
            containing information about Box files.
        token_limit (int, optional): The maximum number of tokens (words or
            characters) to store in the text_representation attribute.
            Defaults to 10000.

    Returns:
        List[_BoxResourcePayload]: The updated list of _BoxResourcePayload objects
            with the following attribute added or updated:
            - text_representation (str, optional): The extracted text content
              of the file, truncated to token_limit if applicable. None if
              the text cannot be retrieved or is still being generated.
    """
    for payload in payloads:
        box_file = payload.resource_info

        try:
            # Request the file with the "extracted_text" representation hint
            box_file = box_client.files.get_file_by_id(
                box_file.id,
                x_rep_hints="[extracted_text]",
                fields=["name", "representations"],
            )
        except BoxAPIError as e:
            logger.error(
                f"Error getting file representation {box_file.id}: {e.message}",
                exc_info=True,
            )
            payload.text_representation = None
            continue

        # Check if any representations exist
        if not box_file.representations.entries:
            logger.error(f"No representation for file {box_file.id}")
            payload.text_representation = None
            continue

        # Find the "extracted_text" representation
        extracted_text_entry = next(
            (
                entry
                for entry in box_file.representations.entries
                if entry.representation == "extracted_text"
            ),
            None,
        )
        if not extracted_text_entry:
            payload.text_representation = None
            continue

        # Handle cases where the extracted text needs generation
        if extracted_text_entry.status.state == "none":
            _do_request(extracted_text_entry.info.url)  # Trigger text generation

        # Construct the download URL and sanitize filename
        url = extracted_text_entry.content.url_template.replace("{+asset_path}", "")

        # Download and truncate the raw content
        raw_content = _do_request(box_client, url)
        payload.text_representation = raw_content[:token_limit] if raw_content else None

    return payloads


def get_files_ai_extract_data(
    box_client: BoxClient, payloads: List[_BoxResourcePayload], ai_prompt: str
) -> List[_BoxResourcePayload]:
    """
    Extracts data from Box files using Box AI.

    This function utilizes the Box AI Extract functionality to process each file
    in the payloads list according to the provided prompt. The extracted data
    is then stored in the ai_response attribute of each payload object.

    Args:
        box_client (BoxClient): An authenticated Box client object.
        payloads (List[_BoxResourcePayload]): A list of _BoxResourcePayload objects
            containing information about Box files.
        ai_prompt (str): The AI prompt that specifies what data to extract
            from the files.

    Returns:
        List[_BoxResourcePayload]: The updated list of _BoxResourcePayload objects
            with the following attribute added or updated:
            - ai_response (str, optional): The extracted data from the file
              based on the AI prompt. May be empty if the extraction fails.
    """
    ai_extract_manager = AiExtractManager(
        auth=box_client.auth, network_session=box_client.network_session
    )

    for payload in payloads:
        file = payload.resource_info
        ask_item = CreateAiExtractItems(file.id)
        logger.info(f"Getting AI extracted data for file: {file.id} {file.name}")

        # get the AI extracted data for the file
        try:
            ai_response = ai_extract_manager.create_ai_extract(
                prompt=ai_prompt, items=[ask_item]
            )
        except BoxAPIError as e:
            logger.error(
                f"An error occurred while getting AI extracted data for file: {e}",
                exc_info=True,
            )
            # payload.ai_response = e.message
            continue
        payload.ai_response = ai_response.answer

    return payloads


def search_files(
    box_client: BoxClient,
    query: Optional[str] = None,
    scope: Optional[SearchForContentScope] = None,
    file_extensions: Optional[List[str]] = None,
    created_at_range: Optional[List[str]] = None,
    updated_at_range: Optional[List[str]] = None,
    size_range: Optional[List[int]] = None,
    owner_user_ids: Optional[List[str]] = None,
    recent_updater_user_ids: Optional[List[str]] = None,
    ancestor_folder_ids: Optional[List[str]] = None,
    content_types: Optional[List[SearchForContentContentTypes]] = None,
    # type: Optional[SearchForContentType] = None,
    # trash_content: Optional[SearchForContentTrashContent] = None,
    # mdfilters: Optional[List[MetadataFilter]] = None,
    # sort: Optional[SearchForContentSort] = None,
    # direction: Optional[SearchForContentDirection] = None,
    limit: Optional[int] = None,
    # include_recent_shared_links: Optional[bool] = None,
    # fields: Optional[List[str]] = None,
    offset: Optional[int] = None,
    # deleted_user_ids: Optional[List[str]] = None,
    # deleted_at_range: Optional[List[str]] = None,
    # extra_headers: Optional[Dict[str, Optional[str]]] = None,
) -> List[File]:
    """
    Searches for files in Box based on a query string.

    Args:
        box_client (BoxClient): A Box client object.
        query (str): The search query string.

    Returns:
        List[File]: A list of Box file objects that match the search query.
    """
    # force to return only object type "file"
    type = SearchForContentType.FILE
    # return only the file id
    fields = ["id"]
    try:
        search_results: SearchResults = box_client.search.search_for_content(
            query=query,
            scope=scope,
            file_extensions=file_extensions,
            created_at_range=created_at_range,
            updated_at_range=updated_at_range,
            size_range=size_range,
            owner_user_ids=owner_user_ids,
            recent_updater_user_ids=recent_updater_user_ids,
            ancestor_folder_ids=ancestor_folder_ids,
            content_types=content_types,
            type=type,
            limit=limit,
            fields=fields,
            offset=offset,
        )
    except BoxAPIError as e:
        logger.error(f"An error occurred while searching for files: {e}", exc_info=True)
        return []

    return search_results.entries


def search_files_by_metadata(
    box_client: BoxClient,
    from_: str,
    ancestor_folder_id: str,
    query: Optional[str] = None,
    query_params: Optional[Dict[str, str]] = None,
    # order_by: Optional[List[SearchByMetadataQueryOrderBy]] = None,
    limit: Optional[int] = None,
    marker: Optional[str] = None,
    # fields: Optional[List[str]] = None,
    # extra_headers: Optional[Dict[str, Optional[str]]] = None,
) -> List[File]:
    """
    Searches for files in Box based on metadata filters.

    Args:
        box_client (BoxClient): A Box client object.
        metadata_filters (List[MetadataFilter]): A list of metadata filters to apply to the search.
        limit (int, optional): The maximum number of items to return. Defaults to None.
        offset (int, optional): The offset of the item at which to start the response. Defaults to None.

    Returns:
        List[File]: A list of Box file objects that match the search query.
    """
    # return only the file id
    fields = ["id"]
    try:
        search_results: SearchResults = box_client.search.search_by_metadata_query(
            from_=from_,
            ancestor_folder_id=ancestor_folder_id,
            query=query,
            query_params=query_params,
            limit=limit,
            marker=marker,
            fields=fields,
        )
    except BoxAPIError as e:
        logger.error(f"An error occurred while searching for files: {e}", exc_info=True)
        return []

    # return only files from the entries
    return [file for file in search_results.entries if file.type == "file"]
