import logging
from typing import List, Optional, Dict

from llama_index.core.readers.base import (
    BaseReader,
    ResourcesReaderMixin,
)
from llama_index.core.schema import Document
from llama_index.readers.box.BoxAPI.box_api import (
    _BoxResourcePayload,
    box_check_connection,
    get_box_files_payload,
    get_box_folder_payload,
    search_files,
    search_files_by_metadata,
    get_files_ai_prompt,
)

from box_sdk_gen import (
    BoxClient,
    SearchForContentScope,
    SearchForContentContentTypes,
)


logger = logging.getLogger(__name__)


class BoxReaderAIPrompt(BaseReader, ResourcesReaderMixin):
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
        self._box_client = box_client

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

    def list_resources(
        self,
        folder_id: Optional[str] = None,
        file_ids: Optional[List[str]] = None,
        is_recursive: bool = False,
    ) -> List[str]:
        """
        Lists the IDs of Box files based on the specified folder or file IDs.

        This method retrieves a list of Box file identifiers based on the provided
        parameters. You can either specify a list of file IDs or a folder ID with an
        optional `is_recursive` flag to include files from sub-folders as well.

        Args:
            folder_id (Optional[str], optional): The ID of the Box folder to list files
                from. If provided, along with `is_recursive` set to True, retrieves data
                from sub-folders as well. Defaults to None.
            file_ids (Optional[List[str]], optional): A list of Box file IDs to retrieve.
                If provided, this takes precedence over `folder_id`. Defaults to None.
            is_recursive (bool, optional): If True and `folder_id` is provided, retrieves
                resource IDs from sub-folders within the specified folder. Defaults to False.

        Returns:
            List[str]: A list containing the IDs of the retrieved Box files.
        """
        # Connect to Box
        box_check_connection(self._box_client)

        # Get the file resources
        payloads: List[_BoxResourcePayload] = []
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
        return [payload.resource_info.id for payload in payloads]

    def get_resource_info(self, box_file_id: str) -> Dict:
        """
        Get information about a specific resource.

        Args:
            resource_id (str): The resource identifier.

        Returns:
            Dict: A dictionary of information about the resource.
        """
        # Connect to Box
        box_check_connection(self._box_client)

        resource = get_box_files_payload(
            box_client=self._box_client, file_ids=[box_file_id]
        )

        return resource[0].resource_info.to_dict()

    def load_resource(self, box_file_id: str, ai_prompt: str) -> List[Document]:
        """
        Load data from a specific resource.

        Args:
            resource (str): The resource identifier.

        Returns:
            List[Document]: A list of documents loaded from the resource.
        """
        return self.load_data(file_ids=[box_file_id], ai_prompt=ai_prompt)

    def search_resources(
        self,
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
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[str]:
        """
        Searches for Box resources based on specified criteria and returns a list of their IDs.

        This method utilizes the Box API search functionality to find resources
        matching the provided parameters. It then returns a list containing the IDs
        of the found resources.

        Args:
            query (Optional[str], optional): A search query string. Defaults to None.
            scope (Optional[SearchForContentScope], optional): The scope of the search.
                Defaults to None.
            file_extensions (Optional[List[str]], optional): A list of file extensions
                to filter by. Defaults to None.
            created_at_range (Optional[List[str]], optional): A list representing a date
                range for file creation time. Defaults to None.
            updated_at_range (Optional[List[str]], optional): A list representing a date
                range for file update time. Defaults to None.
            size_range (Optional[List[int]], optional): A list representing a size range
                for files. Defaults to None.
            owner_user_ids (Optional[List[str]], optional): A list of user IDs to filter
                by owner. Defaults to None.
            recent_updater_user_ids (Optional[List[str]], optional): A list of user IDs to
                filter by recent updater. Defaults to None.
            ancestor_folder_ids (Optional[List[str]], optional): A list of folder IDs to
                search within. Defaults to None.
            content_types (Optional[List[SearchForContentContentTypes]], optional): A list
                of content types to filter by. Defaults to None.
            limit (Optional[int], optional): The maximum number of results to return.
                Defaults to None.
            offset (Optional[int], optional): The number of results to skip before
                starting to collect. Defaults to None.

        Returns:
            List[str]: A list of Box resource IDs matching the search criteria.
        """
        # Connect to Box
        box_check_connection(self._box_client)

        box_files = search_files(
            box_client=self._box_client,
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
            limit=limit,
            offset=offset,
        )
        return [box_file.id for box_file in box_files]

    def search_resources_by_metadata(
        self,
        from_: str,
        ancestor_folder_id: str,
        query: Optional[str] = None,
        query_params: Optional[Dict[str, str]] = None,
        limit: Optional[int] = None,
        marker: Optional[str] = None,
    ) -> List[str]:
        """
        Searches for Box resources based on metadata and returns a list of their IDs.

        This method utilizes the Box API search functionality to find resources
        matching the provided metadata query. It then returns a list containing the IDs
        of the found resources.

        Args:
            box_client (BoxClient): An authenticated Box client object used
                for interacting with the Box API.
            from_ (str): The metadata template key to search from.
            ancestor_folder_id (str): The ID of the Box folder to search within.
            query (Optional[str], optional): A search query string. Defaults to None.
            query_params (Optional[Dict[str, str]], optional): Additional query parameters
                to filter the search results. Defaults to None.
            limit (Optional[int], optional): The maximum number of results to return.
                Defaults to None.
            marker (Optional[str], optional): The marker for the start of the next page of
                results. Defaults to None.

        Returns:
            List[str]: A list of Box resource IDs matching the search criteria.
        """
        # Connect to Box
        box_check_connection(self._box_client)

        box_files = search_files_by_metadata(
            box_client=self._box_client,
            from_=from_,
            ancestor_folder_id=ancestor_folder_id,
            query=query,
            query_params=query_params,
            limit=limit,
            marker=marker,
        )
        return [box_file.id for box_file in box_files]
