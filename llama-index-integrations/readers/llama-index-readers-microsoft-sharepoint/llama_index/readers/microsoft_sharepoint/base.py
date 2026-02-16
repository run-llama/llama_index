"""SharePoint files reader."""

import html
import logging
import os
import re
from pathlib import Path
import tempfile
from typing import Any, Callable, Dict, List, Union, Optional
from enum import Enum
from urllib.parse import quote

import requests
from pydantic.json_schema import SkipJsonSchema
from llama_index.core.readers import SimpleDirectoryReader, FileSystemReaderMixin
from llama_index.core.readers.base import (
    BaseReader,
    BasePydanticReader,
    ResourcesReaderMixin,
)
from llama_index.core.schema import Document
from llama_index.core.bridge.pydantic import PrivateAttr, Field
from llama_index.core.instrumentation import DispatcherSpanMixin, get_dispatcher
from .event import (
    TotalPagesToProcessEvent,
    PageDataFetchStartedEvent,
    PageDataFetchCompletedEvent,
    PageSkippedEvent,
    PageFailedEvent,
)

logger = logging.getLogger(__name__)
dispatcher = get_dispatcher(__name__)


class SharePointType(Enum):
    """Enum for SharePoint content types."""

    DRIVE = "drive"
    PAGE = "page"


class SharePointReader(
    BasePydanticReader, ResourcesReaderMixin, FileSystemReaderMixin, DispatcherSpanMixin
):
    """
    SharePoint reader.

    Reads folders from the SharePoint site from a folder under documents.
    Also supports reading SharePoint Site Pages when sharepoint_type is set to PAGE.

    Args:
        client_id (str): The Application ID for the app registered in Microsoft Azure Portal.
            The application must also be configured with MS Graph permissions "Files.ReadAll", "Sites.ReadAll" and BrowserSiteLists.Read.All.
        client_secret (str): The application secret for the app registered in Azure.
        tenant_id (str): Unique identifier of the Azure Active Directory Instance.
        sharepoint_site_name (Optional[str]): The name of the SharePoint site to download from.
        sharepoint_folder_path (Optional[str]): The path of the SharePoint folder to download from.
        sharepoint_folder_id (Optional[str]): The ID of the SharePoint folder to download from. Overrides sharepoint_folder_path.
        drive_name (Optional[str]): The name of the drive to download from.
        drive_id (Optional[str]): The ID of the drive to download from. Overrides drive_name.
        required_exts (Optional[List[str]]): List of required extensions. Default is None.
        file_extractor (Optional[Dict[str, BaseReader]]): A mapping of file extension to a BaseReader class that specifies how to convert that
                                                          file to text. See `SimpleDirectoryReader` for more details.
        attach_permission_metadata (bool): If True, the reader will attach permission metadata to the documents. Set to False if your vector store
                                           only supports flat metadata (i.e. no nested fields or lists), or to avoid the additional API calls.
        sharepoint_type (SharePointType): Type of content to load - DRIVE (files) or PAGE (site pages). Default is DRIVE.
        sharepoint_host_name (Optional[str]): The host name of the SharePoint site (e.g., 'contoso.sharepoint.com').
        sharepoint_relative_url (Optional[str]): The relative URL of the SharePoint site (e.g., '/sites/MySite').
        process_document_callback (Optional[Callable[[str], bool]]): Callback to filter documents before processing.
        fail_on_error (bool): If True, raise exceptions on errors. If False, log and continue. Default is True.

    """

    client_id: str = None
    client_secret: str = None
    tenant_id: str = None
    sharepoint_site_name: Optional[str] = None
    sharepoint_host_name: Optional[str] = None
    sharepoint_relative_url: Optional[str] = None
    sharepoint_site_id: Optional[str] = None
    sharepoint_folder_path: Optional[str] = None
    sharepoint_folder_id: Optional[str] = None
    sharepoint_file_id: Optional[str] = None
    required_exts: Optional[List[str]] = None
    file_extractor: Optional[SkipJsonSchema[Dict[str, Union[str, BaseReader]]]] = Field(
        default=None, exclude=True
    )
    attach_permission_metadata: bool = True
    drive_name: Optional[str] = None
    drive_id: Optional[str] = None
    sharepoint_type: SharePointType = SharePointType.DRIVE
    process_document_callback: Optional[SkipJsonSchema[Callable[[str], bool]]] = Field(
        default=None, exclude=True
    )
    fail_on_error: bool = True

    _authorization_headers = PrivateAttr()
    _site_id_with_host_name = PrivateAttr()
    _drive_id_endpoint = PrivateAttr()
    _drive_id = PrivateAttr()

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        tenant_id: str,
        sharepoint_site_name: Optional[str] = None,
        sharepoint_folder_path: Optional[str] = None,
        sharepoint_folder_id: Optional[str] = None,
        required_exts: Optional[List[str]] = None,
        file_extractor: Optional[Dict[str, Union[str, BaseReader]]] = None,
        drive_name: Optional[str] = None,
        drive_id: Optional[str] = None,
        sharepoint_host_name: Optional[str] = None,
        sharepoint_relative_url: Optional[str] = None,
        sharepoint_type: SharePointType = SharePointType.DRIVE,
        process_document_callback: Optional[Callable[[str], bool]] = None,
        fail_on_error: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            client_id=client_id,
            client_secret=client_secret,
            tenant_id=tenant_id,
            sharepoint_site_name=sharepoint_site_name,
            sharepoint_host_name=sharepoint_host_name,
            sharepoint_relative_url=sharepoint_relative_url,
            sharepoint_folder_path=sharepoint_folder_path,
            sharepoint_folder_id=sharepoint_folder_id,
            required_exts=required_exts,
            file_extractor=file_extractor,
            drive_name=drive_name,
            drive_id=drive_id,
            sharepoint_type=sharepoint_type,
            process_document_callback=process_document_callback,
            fail_on_error=fail_on_error,
            **kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        return "SharePointReader"

    def _get_access_token(self) -> str:
        """
        Gets the access_token for accessing file from SharePoint.

        Returns:
            str: The access_token for accessing the file.

        Raises:
            ValueError: If there is an error in obtaining the access_token.

        """
        authority = f"https://login.microsoftonline.com/{self.tenant_id}/oauth2/token"

        payload = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "resource": "https://graph.microsoft.com/",
        }

        response = requests.post(
            url=authority,
            data=payload,
        )

        json_response = response.json()

        if response.status_code == 200 and "access_token" in json_response:
            return json_response["access_token"]

        else:
            error_message = json_response.get("error_description") or json_response.get(
                "error"
            )
            logger.error("Error retrieving access token: %s", json_response["error"])
            raise ValueError(f"Error retrieving access token: {error_message}")

    def _send_request_with_retry(self, request: requests.Request) -> requests.Response:
        """
        Makes a request to the SharePoint API with the provided request object.
        If the request fails with a 401 status code, the access token is refreshed and the request is retried once.
        """
        curr_headers = (request.headers or {}).copy()
        curr_headers.update(self._authorization_headers)
        request.headers = curr_headers
        prepared_request = request.prepare()
        with requests.Session() as session:
            response = session.send(prepared_request)

            if response.status_code == 401:
                # 401 status code indicates that the access token has expired
                # refresh the token and retry once
                logger.debug("Received 401. Refreshing access token.")
                access_token = self._get_access_token()
                self._authorization_headers = {
                    "Authorization": f"Bearer {access_token}"
                }
                curr_headers.update(self._authorization_headers)
                request.headers = curr_headers
                prepared_request = request.prepare()
                response = session.send(prepared_request)

            response.raise_for_status()
            return response

    def _send_get_with_retry(self, url: str) -> requests.Response:
        request = requests.Request(
            method="GET",
            url=url,
        )
        return self._send_request_with_retry(request)

    def _get_all_items_with_pagination(self, endpoint: str) -> List[Dict[str, Any]]:
        """
        Generic helper to fetch all items from a Graph API endpoint with pagination support.
        Automatically follows @odata.nextLink to retrieve all pages.

        Args:
            endpoint (str): The initial Graph API endpoint URL

        Returns:
            List[Dict[str, Any]]: All items from all pages combined

        """
        all_items = []
        next_link = endpoint

        while next_link:
            response = self._send_get_with_retry(next_link)
            response_json = response.json()
            items = response_json.get("value", [])
            all_items.extend(items)
            next_link = response_json.get("@odata.nextLink")

        return all_items

    def _get_site_id_with_host_name(
        self, access_token, sharepoint_site_name: Optional[str]
    ) -> str:
        """
        Retrieves the site ID of a SharePoint site using the provided site name.

        Args:
            sharepoint_site_name (str): The name of the SharePoint site.

        Returns:
            str: The ID of the SharePoint site.

        Raises:
            Exception: If the specified SharePoint site is not found.

        """
        if hasattr(self, "_site_id_with_host_name"):
            return self._site_id_with_host_name

        self._authorization_headers = {"Authorization": f"Bearer {access_token}"}

        if self.sharepoint_site_id:
            return self.sharepoint_site_id

        if not (sharepoint_site_name):
            raise ValueError("The SharePoint site name or ID must be provided.")

        site_information_endpoint = f"https://graph.microsoft.com/v1.0/sites"

        while site_information_endpoint:
            response = self._send_get_with_retry(site_information_endpoint)

            json_response = response.json()
            if response.status_code == 200 and "value" in json_response:
                if (
                    len(json_response["value"]) > 0
                    and "id" in json_response["value"][0]
                ):
                    # find the site with the specified name
                    for site in json_response["value"]:
                        if (
                            "name" in site
                            and site["name"].lower() == sharepoint_site_name.lower()
                        ):
                            return site["id"]
                    site_information_endpoint = json_response.get(
                        "@odata.nextLink", None
                    )
                else:
                    raise ValueError(
                        f"The specified sharepoint site {sharepoint_site_name} is not found."
                    )
            else:
                error_message = json_response.get(
                    "error_description"
                ) or json_response.get("error")
                logger.error("Error retrieving site ID: %s", json_response["error"])
                raise ValueError(f"Error retrieving site ID: {error_message}")

        raise ValueError(
            f"The specified sharepoint site {sharepoint_site_name} is not found."
        )

    def _get_drive_id(self) -> str:
        """
        Retrieves the drive ID of the SharePoint site.

        Returns:
            str: The ID of the SharePoint site drive.

        Raises:
            ValueError: If there is an error in obtaining the drive ID.

        """
        if hasattr(self, "_drive_id"):
            return self._drive_id

        if self.drive_id:
            return self.drive_id

        self._drive_id_endpoint = f"https://graph.microsoft.com/v1.0/sites/{self._site_id_with_host_name}/drives"

        # Use generic pagination helper to get all drives
        drives = self._get_all_items_with_pagination(self._drive_id_endpoint)

        if drives:
            if self.drive_name is not None:
                for drive in drives:
                    if drive["name"].lower() == self.drive_name.lower():
                        return drive["id"]
                raise ValueError(f"The specified drive {self.drive_name} is not found.")

            if len(drives) > 0 and "id" in drives[0]:
                return drives[0]["id"]
            else:
                raise ValueError(
                    "Error occurred while fetching the drives for the sharepoint site."
                )
        else:
            logger.error("No drives found for the SharePoint site.")
            raise ValueError("No drives found for the SharePoint site.")

    def _get_sharepoint_folder_id(self, folder_path: str) -> str:
        """
        Retrieves the folder ID of the SharePoint site.

        Args:
            folder_path (str): The path of the folder in the SharePoint site.

        Returns:
            str: The ID of the SharePoint site folder.

        """
        folder_id_endpoint = (
            f"{self._drive_id_endpoint}/{self._drive_id}/root:/{folder_path}"
        )

        response = self._send_get_with_retry(folder_id_endpoint)

        if response.status_code == 200 and "id" in response.json():
            return response.json()["id"]
        else:
            error_message = response.json().get("error", "Unknown error")
            logger.error("Error retrieving folder ID: %s", error_message)
            raise ValueError(f"Error retrieving folder ID: {error_message}")

    def _download_files_and_extract_metadata(
        self,
        folder_id: str,
        download_dir: str,
        include_subfolders: bool = False,
    ) -> Dict[str, str]:
        """
        Downloads files from the specified folder ID and extracts metadata.

        Args:
            folder_id (str): The ID of the folder from which the files should be downloaded.
            download_dir (str): The directory where the files should be downloaded.
            include_subfolders (bool): If True, files from all subfolders are downloaded.

        Returns:
            Dict[str, str]: A dictionary containing the metadata of the downloaded files.

        Raises:
            ValueError: If there is an error in downloading the files.

        """
        files_path = self.list_resources(
            sharepoint_site_name=self.sharepoint_site_name,
            sharepoint_site_id=self.sharepoint_site_id,
            sharepoint_folder_id=folder_id,
        )

        metadata = {}

        for file_path in files_path:
            item = self._get_item_from_path(file_path)
            metadata.update(self._download_file(item, download_dir))

        return metadata

    def _get_file_content_by_url(self, item: Dict[str, Any]) -> bytes:
        """
        Retrieves the content of the file from the provided URL.

        Args:
            item (Dict[str, Any]): Dictionary containing file metadata.

        Returns:
            bytes: The content of the file.

        """
        file_download_url = item["@microsoft.graph.downloadUrl"]
        response = requests.get(file_download_url)

        if response.status_code != 200:
            json_response = response.json()
            error_message = json_response.get("error_description") or json_response.get(
                "error"
            )
            logger.error("Error downloading file content: %s", json_response["error"])
            raise ValueError(f"Error downloading file content: {error_message}")

        return response.content

    def _download_file_by_url(self, item: Dict[str, Any], download_dir: str) -> str:
        """
        Downloads the file from the provided URL.

        Args:
            item (Dict[str, Any]): Dictionary containing file metadata.
            download_dir (str): The directory where the files should be downloaded.

        Returns:
            str: The path of the downloaded file in the temporary directory.

        """
        # Get the download URL for the file.
        file_name = item["name"]

        content = self._get_file_content_by_url(item)

        # Create the directory if it does not exist and save the file.
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)
        file_path = os.path.join(download_dir, file_name)
        with open(file_path, "wb") as f:
            f.write(content)

        return file_path

    def _get_permissions_info(self, item: Dict[str, Any]) -> Dict[str, str]:
        """
        Extracts the permissions information for the file. For more information, see:
        https://learn.microsoft.com/en-us/graph/api/resources/permission?view=graph-rest-1.0.

        Args:
            item (Dict[str, Any]): Dictionary containing file metadata.

        Returns:
            Dict[str, str]: A dictionary containing the extracted permissions information.

        """
        item_id = item.get("id")
        permissions_info_endpoint = (
            f"{self._drive_id_endpoint}/{self._drive_id}/items/{item_id}/permissions"
        )
        response = self._send_get_with_retry(permissions_info_endpoint)
        permissions = response.json()

        identity_sets = []
        for permission in permissions["value"]:
            # user type permissions
            granted_to = permission.get("grantedToV2", None)
            if granted_to:
                identity_sets.append(granted_to)

            # link type permissions
            granted_to_identities = permission.get("grantedToIdentitiesV2", [])
            for identity in granted_to_identities:
                identity_sets.append(identity)

        # Extract the identity information from each identity set
        # they can be 'application', 'device', 'user', 'group', 'siteUser' or 'siteGroup'
        # 'siteUser' and 'siteGroup' are site-specific, 'group' is for Microsoft 365 groups
        permissions_dict = {}
        for identity_set in identity_sets:
            for identity, identity_info in identity_set.items():
                id = identity_info.get("id")
                display_name = identity_info.get("displayName")
                ids_key = f"allowed_{identity}_ids"
                display_names_key = f"allowed_{identity}_display_names"

                if ids_key not in permissions_dict:
                    permissions_dict[ids_key] = []
                if display_names_key not in permissions_dict:
                    permissions_dict[display_names_key] = []

                permissions_dict[ids_key].append(id)
                permissions_dict[display_names_key].append(display_name)

        # sort to get consistent results, if possible
        for key in permissions_dict:
            try:
                permissions_dict[key] = sorted(permissions_dict[key])
            except TypeError:
                pass

        return permissions_dict

    def _extract_metadata_for_file(self, item: Dict[str, Any]) -> Dict[str, str]:
        """
        Extracts metadata related to the file.

        Parameters
        ----------
        - item (Dict[str, str]): Dictionary containing file metadata.

        Returns
        -------
        - Dict[str, str]: A dictionary containing the extracted metadata.

        """
        # Extract the required metadata for file.
        if self.attach_permission_metadata:
            metadata = self._get_permissions_info(item)
        else:
            metadata = {}

        metadata.update(
            {
                "file_id": item.get("id"),
                "file_name": item.get("name"),
                "url": item.get("webUrl"),
                "file_path": item.get("file_path"),
            }
        )

        return metadata

    def _download_file(
        self,
        item: Dict[str, Any],
        download_dir: str,
    ):
        metadata = {}

        file_path = self._download_file_by_url(item, download_dir)

        metadata[file_path] = self._extract_metadata_for_file(item)
        return metadata

    def _download_files_from_sharepoint(
        self,
        download_dir: str,
        sharepoint_site_name: Optional[str],
        sharepoint_folder_path: Optional[str],
        sharepoint_folder_id: Optional[str],
        recursive: bool,
    ) -> Dict[str, str]:
        """
        Downloads files from the specified folder and returns the metadata for the downloaded files.

        Args:
            download_dir (str): The directory where the files should be downloaded.
            sharepoint_site_name (str): The name of the SharePoint site.
            sharepoint_folder_path (str): The path of the folder in the SharePoint site.
            recursive (bool): If True, files from all subfolders are downloaded.

        Returns:
            Dict[str, str]: A dictionary containing the metadata of the downloaded files.

        """
        access_token = self._get_access_token()

        self._site_id_with_host_name = self._get_site_id_with_host_name(
            access_token, sharepoint_site_name
        )

        self._drive_id = self._get_drive_id()

        if not sharepoint_folder_id and sharepoint_folder_path:
            sharepoint_folder_id = self._get_sharepoint_folder_id(
                sharepoint_folder_path
            )

        return self._download_files_and_extract_metadata(
            sharepoint_folder_id,
            download_dir,
            recursive,
        )

    def _exclude_access_control_metadata(
        self, documents: List[Document]
    ) -> List[Document]:
        """
        Excludes the access control metadata from the documents for embedding and LLM calls.

        Args:
            documents (List[Document]): A list of documents.

        Returns:
            List[Document]: A list of documents with access control metadata excluded.

        """
        for doc in documents:
            access_control_keys = [
                key for key in doc.metadata if key.startswith("allowed_")
            ]

            doc.excluded_embed_metadata_keys.extend(access_control_keys)
            doc.excluded_llm_metadata_keys.extend(access_control_keys)

        return documents

    def _load_documents_with_metadata(
        self,
        files_metadata: Dict[str, Any],
        download_dir: str,
        recursive: bool,
    ) -> List[Document]:
        """
        Loads the documents from the downloaded files.

        Args:
            files_metadata (Dict[str,Any]): A dictionary containing the metadata of the downloaded files.
            download_dir (str): The directory where the files should be downloaded.
            recursive (bool): If True, files from all subfolders are downloaded.

        Returns:
            List[Document]: A list containing the documents with metadata.

        """

        def get_metadata(filename: str) -> Any:
            return files_metadata[filename]

        simple_loader = SimpleDirectoryReader(
            download_dir,
            required_exts=self.required_exts,
            file_extractor=self.file_extractor,
            file_metadata=get_metadata,
            recursive=recursive,
        )
        docs = simple_loader.load_data()
        if self.attach_permission_metadata:
            docs = self._exclude_access_control_metadata(docs)
        return docs

    def load_data(
        self,
        sharepoint_site_name: Optional[str] = None,
        sharepoint_folder_path: Optional[str] = None,
        sharepoint_folder_id: Optional[str] = None,
        recursive: bool = True,
        download_dir: Optional[str] = None,
    ) -> List[Document]:
        """
        Loads data from SharePoint based on sharepoint_type.
        Handles both drive (files/folders) and page types.

        Args:
            sharepoint_site_name (Optional[str]): The name of the SharePoint site.
            sharepoint_folder_path (Optional[str]): The path of the folder in the SharePoint site.
            sharepoint_folder_id (Optional[str]): The ID of the folder in the SharePoint site.
            recursive (bool): If True, files from all subfolders are downloaded.
            download_dir (Optional[str]): Directory to download files to.

        Returns:
            List[Document]: A list containing the documents with metadata.

        Raises:
            Exception: If an error occurs while accessing SharePoint site.

        """
        # If sharepoint_type is 'page', use the page loading functionality
        if self.sharepoint_type == SharePointType.PAGE:
            logger.info(f"Loading pages from site {self.sharepoint_site_name}")
            return self.load_pages_data(download_dir=download_dir)

        # If no arguments are provided to load_data, default to the object attributes
        if not sharepoint_site_name:
            sharepoint_site_name = self.sharepoint_site_name

        if not sharepoint_folder_path:
            sharepoint_folder_path = self.sharepoint_folder_path

        if not sharepoint_folder_id:
            sharepoint_folder_id = self.sharepoint_folder_id

        # TODO: make both of these values optional — and just default to the client ID defaults
        if not (sharepoint_site_name or self.sharepoint_site_id):
            raise ValueError("sharepoint_site_name must be provided.")

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                files_metadata = self._download_files_from_sharepoint(
                    temp_dir,
                    sharepoint_site_name,
                    sharepoint_folder_path,
                    sharepoint_folder_id,
                    recursive,
                )

                # return self.files_metadata
                return self._load_documents_with_metadata(
                    files_metadata, temp_dir, recursive
                )

        except Exception as exp:
            logger.error("An error occurred while accessing SharePoint: %s", exp)
            if self.fail_on_error:
                raise
            return []

    def _list_folder_contents(
        self, folder_id: str, recursive: bool, current_path: str
    ) -> List[Path]:
        """
        Helper method to fetch the contents of a folder.

        Args:
            folder_id (str): ID of the folder whose contents are to be listed.
            recursive (bool): Whether to include subfolders recursively.

        Returns:
            List[Path]: List of file paths.

        """
        folder_contents_endpoint = (
            f"{self._drive_id_endpoint}/{self._drive_id}/items/{folder_id}/children"
        )
        # Use generic pagination helper to get all items
        items = self._get_all_items_with_pagination(folder_contents_endpoint)
        file_paths = []
        for item in items:
            if "folder" in item and recursive:
                # Recursive call for subfolder
                subfolder_id = item["id"]
                subfolder_paths = self._list_folder_contents(
                    subfolder_id, recursive, os.path.join(current_path, item["name"])
                )
                file_paths.extend(subfolder_paths)
            elif "file" in item:
                # Append file path
                file_path = Path(os.path.join(current_path, item["name"]))
                file_paths.append(file_path)

        return file_paths

    def _list_drive_contents(self) -> List[Path]:
        """
        Helper method to fetch the contents of the drive.

        Returns:
            List[Path]: List of file paths.

        """
        drive_contents_endpoint = (
            f"{self._drive_id_endpoint}/{self._drive_id}/root/children"
        )
        file_paths = []

        # Use generic pagination helper to get all items
        items = self._get_all_items_with_pagination(drive_contents_endpoint)

        for item in items:
            if "folder" in item:
                # Append folder path
                folder_paths = self._list_folder_contents(
                    item["id"], recursive=True, current_path=item["name"]
                )
                file_paths.extend(folder_paths)
            elif "file" in item:
                # Append file path
                file_path = Path(item["name"])
                file_paths.append(file_path)

        return file_paths

    def list_resources(
        self,
        sharepoint_site_name: Optional[str] = None,
        sharepoint_folder_path: Optional[str] = None,
        sharepoint_folder_id: Optional[str] = None,
        sharepoint_site_id: Optional[str] = None,
        recursive: bool = True,
    ) -> List[Path]:
        """
        Lists the files in the specified folder in the SharePoint site.

        Args:
            **kwargs: Additional keyword arguments.

        Returns:
            List[Path]: A list of paths of the files in the specified folder.

        Raises:
            Exception: If an error occurs while accessing SharePoint site.

        """
        # If no arguments are provided to load_data, default to the object attributes
        if not sharepoint_site_name:
            sharepoint_site_name = self.sharepoint_site_name

        if not sharepoint_folder_path:
            sharepoint_folder_path = self.sharepoint_folder_path

        if not sharepoint_folder_id:
            sharepoint_folder_id = self.sharepoint_folder_id

        if not sharepoint_site_id:
            sharepoint_site_id = self.sharepoint_site_id

        if not (sharepoint_site_name or sharepoint_site_id):
            raise ValueError(
                "sharepoint_site_name or sharepoint_site_id must be provided."
            )

        file_paths = []
        try:
            access_token = self._get_access_token()
            self._site_id_with_host_name = self._get_site_id_with_host_name(
                access_token, sharepoint_site_name
            )
            self._drive_id = self._get_drive_id()

            if sharepoint_folder_path:
                if not sharepoint_folder_id:
                    sharepoint_folder_id = self._get_sharepoint_folder_id(
                        sharepoint_folder_path
                    )
                # Fetch folder contents
                folder_contents = self._list_folder_contents(
                    sharepoint_folder_id,
                    recursive,
                    os.path.join(sharepoint_site_name, sharepoint_folder_path),
                )
                file_paths.extend(folder_contents)
            else:
                # Fetch drive contents
                drive_contents = self._list_drive_contents()
                file_paths.extend(drive_contents)
        except Exception as exp:
            logger.error("An error occurred while listing files in SharePoint: %s", exp)
            raise

        return file_paths

    def _get_item_from_path(self, input_file: Path) -> Dict[str, Any]:
        """
        Retrieves the item details for a specified file in SharePoint.

        Args:
            input_file (Path): The path of the file in SharePoint.
                Should include the SharePoint site name and the folder path. e.g. "site_name/folder_path/file_name".

        Returns:
            Dict[str, Any]: Dictionary containing the item details.

        """
        # Get the file ID
        # remove the site_name prefix
        parts = [part for part in input_file.parts if part != self.sharepoint_site_name]
        # URI escape each part of the path
        escaped_parts = [quote(part) for part in parts]
        file_path = "/".join(escaped_parts)
        endpoint = f"{self._drive_id_endpoint}/{self._drive_id}/root:/{file_path}"

        response = self._send_get_with_retry(endpoint)

        return response.json()

    def get_permission_info(self, resource_id: str, **kwargs) -> Dict:
        """
        Get a dictionary of information about the permissions of a specific resource.
        """
        try:
            item = self._get_item_from_path(Path(resource_id))
            return self._get_permissions_info(item)
        except Exception as exp:
            logger.error(
                "An error occurred while fetching file information from SharePoint: %s",
                exp,
            )
            raise

    def get_resource_info(self, resource_id: str, **kwargs) -> Dict:
        """
        Retrieves metadata for a specified file in SharePoint without downloading it.

        Args:
            input_file (Path): The path of the file in SharePoint. The path should include
                the SharePoint site name and the folder path. e.g. "site_name/folder_path/file_name".

        """
        try:
            item = self._get_item_from_path(Path(resource_id))

            info_dict = {
                "file_path": resource_id,
                "size": item.get("size"),
                "created_at": item.get("createdDateTime"),
                "modified_at": item.get("lastModifiedDateTime"),
                "etag": item.get("eTag"),
                "url": item.get("webUrl"),
            }

            if (
                self.attach_permission_metadata
            ):  # changes in access control should trigger a reingestion of the file
                permissions = self._get_permissions_info(item)
                info_dict.update(permissions)

            return {
                meta_key: meta_value
                for meta_key, meta_value in info_dict.items()
                if meta_value is not None
            }

        except Exception as exp:
            logger.error(
                "An error occurred while fetching file information from SharePoint: %s",
                exp,
            )
            raise

    def load_resource(self, resource_id: str, **kwargs) -> List[Document]:
        try:
            access_token = self._get_access_token()
            self._site_id_with_host_name = self._get_site_id_with_host_name(
                access_token, self.sharepoint_site_name
            )
            self._drive_id = self._get_drive_id()

            path = Path(resource_id)

            item = self._get_item_from_path(path)

            with tempfile.TemporaryDirectory() as temp_dir:
                metadata = self._download_file(item, temp_dir)
                return self._load_documents_with_metadata(
                    metadata, temp_dir, recursive=False
                )

        except Exception as exp:
            logger.error(
                "An error occurred while reading file from SharePoint: %s", exp
            )
            raise

    def read_file_content(self, input_file: Path, **kwargs) -> bytes:
        try:
            access_token = self._get_access_token()
            self._site_id_with_host_name = self._get_site_id_with_host_name(
                access_token, self.sharepoint_site_name
            )
            self._drive_id = self._get_drive_id()

            item = self._get_item_from_path(input_file)
            return self._get_file_content_by_url(item)

        except Exception as exp:
            logger.error(
                "An error occurred while reading file content from SharePoint: %s", exp
            )
            raise

    # Page-related methods for SharePoint Site Pages support

    def get_site_pages_list_id(self, site_id: str) -> str:
        """
        Gets the ID of the Site Pages list for a SharePoint site.

        Args:
            site_id (str): The ID of the SharePoint site.

        Returns:
            str: The ID of the Site Pages list.

        """
        endpoint = f"https://graph.microsoft.com/v1.0/sites/{site_id}/lists?$filter=displayName eq 'Site Pages'"
        try:
            # Use generic pagination helper to get all lists
            lists = self._get_all_items_with_pagination(endpoint)
            if not lists:
                logger.error("Site Pages list not found for site %s", site_id)
                raise ValueError("Site Pages list not found")
            return lists[0]["id"]
        except Exception as e:
            logger.error(f"Error getting Site Pages list ID: {e}", exc_info=True)
            raise

    def list_pages(self, site_id: str) -> List[Dict[str, Any]]:
        """
        Returns a list of SharePoint site pages with their IDs and names.

        Args:
            site_id (str): The ID of the SharePoint site.

        Returns:
            List[Dict[str, Any]]: List of page dictionaries with id, name, and lastModifiedDateTime.

        """
        try:
            list_id = self.get_site_pages_list_id(site_id)
            endpoint = f"https://graph.microsoft.com/v1.0/sites/{site_id}/lists/{list_id}/items?expand=fields(select=FileLeafRef,CanvasContent1)"

            pages = []

            # Use generic pagination helper to get all items
            items = self._get_all_items_with_pagination(endpoint)

            for item in items:
                fields = item.get("fields", {})
                page_id = item.get("id")
                page_name = fields.get("FileLeafRef")
                last_modified = item.get("lastModifiedDateTime")
                if page_id and page_name:
                    pages.append(
                        {
                            "id": page_id,
                            "name": page_name,
                            "lastModifiedDateTime": last_modified,
                        }
                    )
            return pages
        except Exception as e:
            logger.error(f"Error listing SharePoint pages: {e}", exc_info=True)
            raise

    def get_page_id_by_name(self, site_id: str, page_name: str) -> Optional[str]:
        """
        Get the ID of a SharePoint page by its name.

        Args:
            site_id (str): The ID of the SharePoint site.
            page_name (str): The name of the page to find.

        Returns:
            Optional[str]: The page ID if found, None otherwise.

        """
        try:
            list_id = self.get_site_pages_list_id(site_id)
            endpoint = f"https://graph.microsoft.com/v1.0/sites/{site_id}/lists/{list_id}/items?expand=fields"

            # Use generic pagination helper to get all items
            items = self._get_all_items_with_pagination(endpoint)

            # Search for matching page
            matches = [
                item
                for item in items
                if item.get("fields", {}).get("FileLeafRef") == page_name
            ]
            if matches:
                return matches[0].get("id")

            return None
        except Exception as e:
            logger.error(
                f"Error getting page ID by name {page_name}: {e}", exc_info=True
            )
            raise

    def get_page_text(self, site_id: str, list_id: str, page_id: str) -> Dict[str, Any]:
        """
        Gets the text content of a SharePoint page.

        Args:
            site_id (str): The ID of the SharePoint site.
            list_id (str): The ID of the Site Pages list.
            page_id (str): The ID of the page (can be raw ID or combined listId:itemId).

        Returns:
            Dict[str, Any]: Dictionary containing page id, name, lastModifiedDateTime, textContent, and rawHtml.

        """
        try:
            raw_page_id = page_id
            if ":" in page_id:
                parts = page_id.split(":", 1)
                if len(parts) == 2:
                    list_id, raw_page_id = parts
            if not list_id:
                list_id = self.get_site_pages_list_id(site_id)
            endpoint = f"https://graph.microsoft.com/v1.0/sites/{site_id}/lists/{list_id}/items/{raw_page_id}?expand=fields(select=FileLeafRef,CanvasContent1)"
            response = self._send_get_with_retry(endpoint)
            fields = response.json().get("fields", {})
            last_modified = response.json().get("lastModifiedDateTime")
            if not fields:
                raise ValueError("Page not found")
            raw_html = fields.get("CanvasContent1", "") or ""
            unescaped = html.unescape(raw_html)
            text_content = re.sub(r"<[^>]+>", "", unescaped)
            text_content = re.sub(r"['\"]", "", text_content).strip()
            return {
                "id": f"{list_id}:{raw_page_id}",
                "name": fields.get("FileLeafRef"),
                "lastModifiedDateTime": last_modified,
                "textContent": text_content,
                "rawHtml": raw_html,
            }
        except Exception as e:
            logger.error(
                f"Error getting page text for page {page_id}: {e}", exc_info=True
            )
            raise

    @dispatcher.span
    def load_pages_data(self, download_dir: Optional[str] = None) -> List[Document]:
        """
        Loads SharePoint pages as Documents.

        If self.sharepoint_file_id (combined page id) is provided, only process that page.
        Otherwise, process all pages.

        Args:
            download_dir (Optional[str]): Directory to download files to (not used for pages,
                kept for API compatibility).

        Returns:
            List[Document]: A list of Document objects containing page content.

        """
        logger.info(
            f"Loading page data for site {self.sharepoint_site_name} "
            f"(single_page={bool(self.sharepoint_file_id)})"
        )
        try:
            access_token = self._get_access_token()
            site_id = self._get_site_id_with_host_name(
                access_token, self.sharepoint_site_name
            )
            list_id = self.get_site_pages_list_id(site_id)
            documents: List[Document] = []

            if self.sharepoint_file_id:
                # Specific page requested
                try:
                    page_info = self.get_page_text(
                        site_id=site_id,
                        list_id=list_id,
                        page_id=self.sharepoint_file_id,
                    )
                    combined_id = page_info["id"]
                    page_name = page_info["name"]
                    last_modified_date_time = page_info.get("lastModifiedDateTime", "")
                    url_with_id = f"https://{self.sharepoint_host_name}/{self.sharepoint_relative_url}/SitePages/{page_name}?id={self.sharepoint_file_id}"
                    metadata = {
                        "page_id": combined_id,
                        "page_name": page_name,
                        "site_id": site_id,
                        "site_name": self.sharepoint_site_name,
                        "host_name": self.sharepoint_host_name,
                        "lastModifiedDateTime": last_modified_date_time,
                        "sharepoint_relative_url": self.sharepoint_relative_url,
                        "url": url_with_id,
                        "file_name": page_name,
                        "sharepoint_type": SharePointType.PAGE.value,
                    }
                    text = page_info.get("textContent", "")
                    document = Document(text=text, metadata=metadata, id_=combined_id)
                    dispatcher.event(PageDataFetchStartedEvent(page_id=combined_id))
                    dispatcher.event(
                        PageDataFetchCompletedEvent(
                            page_id=combined_id, document=document
                        )
                    )
                    documents.append(document)
                except Exception as e:
                    dispatcher.event(
                        PageFailedEvent(page_id=self.sharepoint_file_id, error=str(e))
                    )
                    logger.error(
                        f"Error loading SharePoint page {self.sharepoint_file_id}: {e}",
                        exc_info=True,
                    )
                    if self.fail_on_error:
                        raise
                return documents

                # Load all pages
            pages = self.list_pages(site_id)
            dispatcher.event(TotalPagesToProcessEvent(total_pages=len(pages)))

            for page in pages:
                raw_page_id = page["id"]
                combined_id = f"{list_id}:{raw_page_id}"
                page_name = page["name"]
                last_modified_date_time = page.get("lastModifiedDateTime", "")

                try:
                    if (
                        self.process_document_callback
                        and not self.process_document_callback(page_name)
                    ):
                        dispatcher.event(PageSkippedEvent(page_id=combined_id))
                        continue

                    url_with_id = f"https://{self.sharepoint_host_name}/{self.sharepoint_relative_url}/SitePages/{page_name}?id={raw_page_id}"
                    metadata = {
                        "page_id": combined_id,
                        "page_name": page_name,
                        "site_id": site_id,
                        "site_name": self.sharepoint_site_name,
                        "host_name": self.sharepoint_host_name,
                        "lastModifiedDateTime": last_modified_date_time,
                        "sharepoint_relative_url": self.sharepoint_relative_url,
                        "url": url_with_id,
                        "file_name": page_name,
                        "sharepoint_type": SharePointType.PAGE.value,
                    }
                    dispatcher.event(PageDataFetchStartedEvent(page_id=combined_id))
                    page_content = self.get_page_text(
                        site_id=site_id,
                        list_id=list_id,
                        page_id=raw_page_id,
                    )
                    text = page_content.get("textContent", "")
                    metadata["lastModifiedDateTime"] = page_content.get(
                        "lastModifiedDateTime", last_modified_date_time
                    )
                    document = Document(text=text, metadata=metadata, id_=combined_id)
                    dispatcher.event(
                        PageDataFetchCompletedEvent(
                            page_id=combined_id, document=document
                        )
                    )
                    documents.append(document)
                except Exception as e:
                    dispatcher.event(PageFailedEvent(page_id=combined_id, error=str(e)))
                    logger.error(
                        f"Error loading SharePoint page {combined_id}: {e}",
                        exc_info=True,
                    )
                    if self.fail_on_error:
                        raise
            return documents
        except Exception as e:
            error_msg = f"Error loading SharePoint pages: {e}"
            logger.error(f"{error_msg}", exc_info=True)
            if self.fail_on_error:
                raise
            return []
