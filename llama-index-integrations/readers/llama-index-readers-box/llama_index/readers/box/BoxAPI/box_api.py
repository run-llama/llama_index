import logging
import os
import shutil
from typing import Dict, List, Optional

import requests
from box_sdk_gen import (
    AiItemBase,
    BoxAPIError,
    BoxClient,
    BoxSDKError,
    ByteStream,
    CreateAiAskMode,
    File,
    SearchForContentContentTypes,
    SearchForContentScope,
    SearchForContentType,
    SearchResults,
)

# from llama_index.readers.box.BoxAPI.box_ai_extract_beta import (
#     AiExtractManager,
#     CreateAiExtractItems,
# )

logger = logging.getLogger(__name__)


def add_extra_header_to_box_client(box_client: BoxClient) -> BoxClient:
    """
    Add extra headers to the Box client.

    Args:
        box_client (BoxClient): A Box client object.
        header (Dict[str, str]): A dictionary of extra headers to add to the Box client.

    Returns:
        BoxClient: A Box client object with the extra headers added.
    """
    header = {"x-box-ai-library": "llama-index"}
    return box_client.with_extra_headers(extra_headers=header)


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


def get_box_files_details(box_client: BoxClient, file_ids: List[str]) -> List[File]:
    """
    Retrieves details for multiple Box files identified by their IDs.

    This function takes a Box client and a list of file IDs as input and returns a list of File objects containing details for each requested file.

    Args:
        box_client (BoxClient): An authenticated Box client object.
        file_ids (List[str]): A list of strings representing Box file IDs.

    Returns:
        List[File]: A list of File objects containing details for each requested file.

    Raises:
        BoxAPIError: If an error occurs while retrieving file details from the Box API.
    """
    box_files_details: List[File] = []

    for file_id in file_ids:
        try:
            file = box_client.files.get_file_by_id(file_id)
        except BoxAPIError as e:
            logger.error(
                f"An error occurred while getting file: {e.message}", exc_info=True
            )
            raise

        logger.info(f"Getting file: {file.id} {file.name} {file.type}")
        box_files_details.append(file)

    return box_files_details


def get_box_folder_files_details(
    box_client: BoxClient, folder_id: str, is_recursive: bool = False
) -> List[File]:
    """
    Retrieves details for all files within a Box folder, optionally including nested sub-folders.

    This function takes a Box client, a folder ID, and an optional recursion flag as input. It retrieves details for all files within the specified folder and returns a list of File objects containing details for each file.

    Args:
        box_client (BoxClient): An authenticated Box client object.
        folder_id (str): The ID of the Box folder to retrieve files from.
        is_recursive (bool, optional): A flag indicating whether to recursively traverse sub-folders within the target folder. Defaults to False.

    Returns:
        List[File]: A list of File objects containing details for all files found within the specified folder and its sub-folders (if recursion is enabled).

    Raises:
        BoxAPIError: If an error occurs while retrieving folder or file details from the Box API.
    """
    box_files_details: List[File] = []
    try:
        folder = box_client.folders.get_folder_by_id(folder_id)
    except BoxAPIError as e:
        logger.error(
            f"An error occurred while getting folder: {e.message}", exc_info=True
        )
        raise

    for item in box_client.folders.get_folder_items(folder.id).entries:
        if item.type == "file":
            box_files_details.extend(get_box_files_details(box_client, [item.id]))
        elif item.type == "folder":
            if is_recursive:
                box_files_details.extend(
                    get_box_folder_files_details(box_client, item.id, is_recursive)
                )
    return box_files_details


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


def get_file_content_by_id(box_client: BoxClient, box_file_id: str) -> bytes:
    try:
        file_stream: ByteStream = box_client.downloads.download_file(box_file_id)
        return file_stream.read()
    except BoxAPIError as e:
        logger.error(f"An error occurred while downloading file: {e}", exc_info=True)
        raise


def get_ai_response_from_box_files(
    box_client: BoxClient,
    ai_prompt: str,
    box_files: List[File],
    individual_document_prompt: bool = True,
) -> List[File]:
    """
    Retrieves AI responses for a prompt based on content within Box files.

    This function takes a Box client, an AI prompt string, a list of Box File objects, and an optional flag indicating prompt mode as input. It utilizes the Box API's AI capabilities to generate AI responses based on the prompt and the content of the provided files. The function then updates the File objects with the prompt and AI response information.

    Args:
        box_client (BoxClient): An authenticated Box client object.
        ai_prompt (str): The AI prompt string to be used for generating responses.
        box_files (List[File]): A list of Box File objects representing the files to be analyzed.
        individual_document_prompt (bool, optional): A flag indicating whether to generate individual prompts for each file (True) or a single prompt for all files (False). Defaults to True.

    Returns:
        List[File]: The original list of Box File objects with additional attributes:
            * `ai_prompt`: The AI prompt used for analysis.
            * `ai_response`: The AI response generated based on the prompt and file content.

    Raises:
        BoxAPIError: If an error occurs while interacting with the Box AI API.
    """
    if individual_document_prompt:
        mode = CreateAiAskMode.SINGLE_ITEM_QA
        for file in box_files:
            ask_item = AiItemBase(file.id)
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
                ai_response = None
            file.ai_prompt = ai_prompt
            file.ai_response = ai_response.answer if ai_response else None

    else:
        mode = CreateAiAskMode.MULTIPLE_ITEM_QA
        file_ids = [AiItemBase(file.id) for file in box_files]

        # get the AI prompt for the file
        ai_response = box_client.ai.create_ai_ask(
            mode=mode, prompt=ai_prompt, items=file_ids
        )
        for file in box_files:
            file.ai_prompt = ai_prompt
            file.ai_response = ai_response.answer

    return box_files


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
    box_client: BoxClient, box_files: List[File], token_limit: int = 10000
) -> List[File]:
    """
    Retrieves and populates the text representation for a list of Box files.

    This function takes a Box client, a list of Box File objects, and an optional token limit as input. It attempts to retrieve the extracted text representation for each file using the Box API's representation hints. The function then updates the File objects with the extracted text content, handling cases where text needs generation or is unavailable.

    Args:
        box_client (BoxClient): An authenticated Box client object.
        box_files (List[File]): A list of Box File objects representing the files to extract text from.
        token_limit (int, optional): The maximum number of tokens to include in the text representation. Defaults to 10000.

    Returns:
        List[File]: The original list of Box File objects with an additional attribute:
            * `text_representation`: The extracted text content from the file (truncated to token_limit if applicable).

    Raises:
        BoxAPIError: If an error occurs while interacting with the Box API.
    """
    box_files_text_representations: List[File] = []
    for file in box_files:
        try:
            # Request the file with the "extracted_text" representation hint
            file_text_representation = box_client.files.get_file_by_id(
                file.id,
                x_rep_hints="[extracted_text]",
                fields=["name", "representations"],
            )
        except BoxAPIError as e:
            logger.error(
                f"Error getting file representation {file_text_representation.id}: {e.message}",
                exc_info=True,
            )
            raise

        # Check if any representations exist
        if not file_text_representation.representations.entries:
            logger.warning(f"No representation for file {file_text_representation.id}")
            continue

        # Find the "extracted_text" representation
        extracted_text_entry = next(
            (
                entry
                for entry in file_text_representation.representations.entries
                if entry.representation == "extracted_text"
            ),
            None,
        )
        if not extracted_text_entry:
            file.text_representation = None
            continue

        # Handle cases where the extracted text needs generation
        if extracted_text_entry.status.state == "none":
            _do_request(extracted_text_entry.info.url)  # Trigger text generation

        # Construct the download URL and sanitize filename
        url = extracted_text_entry.content.url_template.replace("{+asset_path}", "")

        # Download and truncate the raw content
        raw_content = _do_request(box_client, url)
        file.text_representation = raw_content[:token_limit] if raw_content else None

        box_files_text_representations.append(file)

    return box_files_text_representations


def get_files_ai_extract_data(
    box_client: BoxClient, ai_prompt: str, box_files: List[File]
) -> List[File]:
    """
    Extracts data from Box files using Box AI features.

    This function takes a Box client, an AI prompt string, and a list of Box File objects as input. It utilizes the Box AI capabilities, specifically the `AiExtractManager`, to extract data from the files based on the provided prompt. The function then updates the File objects with the prompt and AI extracted data information.

    Args:
        box_client (BoxClient): An authenticated Box client object.
        ai_prompt (str): The AI prompt string used to guide data extraction.
        box_files (List[File]): A list of Box File objects representing the files to extract data from.

    Returns:
        List[File]: The original list of Box File objects with additional attributes:
            * `ai_prompt`: The AI prompt used for data extraction.
            * `ai_response`: The extracted data from the file based on the prompt (format may vary depending on Box AI configuration).

    Raises:
        BoxAPIError: If an error occurs while interacting with the Box AI API.
    """
    for file in box_files:
        ask_item = AiItemBase(file.id)
        logger.info(f"Getting AI extracted data for file: {file.id} {file.name}")

        # get the AI extracted data for the file
        try:
            ai_response = box_client.ai.create_ai_extract(
                prompt=ai_prompt, items=[ask_item]
            )
        except BoxAPIError as e:
            logger.error(
                f"An error occurred while getting AI extracted data for file: {e}",
                exc_info=True,
            )
            raise

        file.ai_prompt = ai_prompt
        file.ai_response = ai_response.answer

    return box_files


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

    # return only files from the entries
    return [file for file in search_results.entries if file.type == "file"]
