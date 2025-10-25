"""SharePoint files reader."""

import html
import logging
import os
import re
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
from enum import Enum
from urllib.parse import quote
import requests
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.readers import FileSystemReaderMixin, SimpleDirectoryReader
from llama_index.core.readers.base import (
    BasePydanticReader,
    BaseReader,
    ResourcesReaderMixin,
)
from llama_index.core.instrumentation import DispatcherSpanMixin, get_dispatcher
from llama_index.core.schema import Document
from .event import (
    FileType,
    TotalPagesToProcessEvent,
    PageDataFetchStartedEvent,
    PageDataFetchCompletedEvent,
    PageSkippedEvent,
    PageFailedEvent,
)

logger = logging.getLogger(__name__)
dispatcher = get_dispatcher(__name__)


class SharePointType(Enum):
    DRIVE = "drive"
    PAGE = "page"


class CustomParserManager:
    def __init__(
        self, custom_parsers: Optional[Dict[FileType, BaseReader]], custom_folder: str
    ):
        self.custom_parsers = custom_parsers or {}
        self.custom_folder = custom_folder

    def __remove_custom_file(self, file_path: str):
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            logger.error(f"Error removing file {file_path}: {e}")

    def process_with_custom_parser(
        self, file_type: FileType, file_content: bytes, extension: str
    ) -> Optional[str]:
        if file_type not in self.custom_parsers:
            return None

        file_name = f"{uuid.uuid4().hex}.{extension}"
        custom_file_path = os.path.join(self.custom_folder, file_name)
        with open(custom_file_path, "wb") as f:
            f.write(file_content)

        try:
            markdown_text = "\n".join(
                doc.text
                for doc in self.custom_parsers[file_type].load_data(
                    file_path=custom_file_path
                )
            )
        finally:
            self.__remove_custom_file(custom_file_path)
        return markdown_text


class SharePointReader(
    BasePydanticReader, ResourcesReaderMixin, FileSystemReaderMixin, DispatcherSpanMixin
):
    """
    SharePoint reader.


    Reads folders from the SharePoint site from a folder under documents.

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

    sharepoint_file_name: Optional[str] = None
    sharepoint_file_id: Optional[str] = None

    required_exts: Optional[List[str]] = None
    file_extractor: Optional[Dict[str, Union[str, BaseReader]]] = Field(
        default=None, exclude=True
    )
    attach_permission_metadata: bool = True
    drive_name: Optional[str] = None
    drive_id: Optional[str] = None
    process_document_callback: Optional[Callable[[str], bool]] = None
    process_attachment_callback: Optional[Callable[[str, int], tuple[bool, str]]] = None
    fail_on_error: bool = True
    custom_folder: Optional[str] = None
    custom_parser_manager: Optional[CustomParserManager] = None
    custom_parsers: Optional[Dict[FileType, Any]] = None
    sharepoint_type: Optional[SharePointType] = SharePointType.DRIVE
    page_name: Optional[str] = None

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
        sharepoint_relative_url: Optional[str] = None,
        sharepoint_folder_path: Optional[str] = None,
        sharepoint_folder_id: Optional[str] = None,
        required_exts: Optional[List[str]] = None,
        file_extractor: Optional[Dict[str, Union[str, BaseReader]]] = None,
        drive_name: Optional[str] = None,
        drive_id: Optional[str] = None,
        sharepoint_host_name: Optional[str] = None,
        sharepoint_type: Optional[SharePointType] = SharePointType.DRIVE,
        page_name: Optional[str] = None,
        custom_parsers: Optional[Dict[FileType, Any]] = None,
        process_document_callback: Optional[Callable[[str], bool]] = None,
        process_attachment_callback: Optional[
            Callable[[str, int], tuple[bool, str]]
        ] = None,
        fail_on_error: bool = True,
        custom_folder: Optional[str] = None,
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
            page_name=page_name,
            process_document_callback=process_document_callback,
            process_attachment_callback=process_attachment_callback,
            fail_on_error=fail_on_error,
            **kwargs,
        )
        self.custom_parsers = custom_parsers or {}
        if custom_parsers and custom_folder:
            self.custom_folder = custom_folder
            self.custom_parser_manager = CustomParserManager(
                custom_parsers, custom_folder
            )
        elif custom_parsers:
            self.custom_folder = os.getcwd()
            self.custom_parser_manager = CustomParserManager(
                custom_parsers, self.custom_folder
            )
        elif custom_folder:
            raise ValueError(
                "custom_folder can only be used when custom_parsers are provided"
            )
        else:
            self.custom_folder = None
            self.custom_parser_manager = None
        self.sharepoint_type = sharepoint_type or SharePointType.DRIVE
        self.page_name = page_name

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

        if self.sharepoint_host_name and self.sharepoint_relative_url:
            site_information_endpoint = f"https://graph.microsoft.com/v1.0/sites/{self.sharepoint_host_name}:/{self.sharepoint_relative_url}"

            response = self._send_get_with_retry(site_information_endpoint)
            json_response = response.json()

            if response.status_code == 200 and "id" in json_response:
                self._site_id_with_host_name = json_response["id"]
                if not self.sharepoint_site_id:
                    self.sharepoint_site_id = json_response["id"]
                return json_response["id"]
            else:
                error_message = json_response.get(
                    "error_description"
                ) or json_response.get("error", "Unknown error")
                logger.error("Error retrieving site ID: %s", error_message)
                raise ValueError(f"Error retrieving site ID: {error_message}")

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

        response = self._send_get_with_retry(self._drive_id_endpoint)
        json_response = response.json()

        if response.status_code == 200 and "value" in json_response:
            if len(json_response["value"]) > 0 and self.drive_name is not None:
                for drive in json_response["value"]:
                    if drive["name"].lower() == self.drive_name.lower():
                        return drive["id"]
                    elif (
                        self.drive_name.lower() == "shared documents"
                        and drive["name"].lower() == "documents"
                    ):
                        return drive["id"]
                raise ValueError(f"The specified drive {self.drive_name} is not found.")

            if len(json_response["value"]) > 0 and "id" in json_response["value"][0]:
                return json_response["value"][0]["id"]
            else:
                raise ValueError(
                    "Error occurred while fetching the drives for the sharepoint site."
                )
        else:
            error_message = json_response.get("error_description") or json_response.get(
                "error"
            )
            logger.error("Error retrieving drive ID: %s", json_response["error"])
            raise ValueError(f"Error retrieving drive ID: {error_message}")

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

    @dispatcher.span
    def _download_files_and_extract_metadata(
        self,
        folder_id: str,
        folder_path: Optional[str],
        file_id_to_process: Optional[str],
        download_dir: str,
        include_subfolders: bool = False,
    ) -> Dict[str, str]:
        """
        Downloads files from the specified folder ID and extracts metadata.

        Args:
            folder_id (str): The ID of the folder from which the files should be downloaded.
            folder_path (Optional[str]): The path of the folder in SharePoint (used for resource listing).
            file_id_to_process (Optional[str]): The ID of a specific file to download (if provided, only this file is processed).
            download_dir (str): The directory where the files should be downloaded.
            include_subfolders (bool): If True, files from all subfolders are downloaded.

        Returns:
            Dict[str, str]: A dictionary containing the metadata of the downloaded files.

        Raises:
            ValueError: If there is an error in downloading the files.

        """
        logger.info(
            f"Downloading files from folder_id={folder_id}, folder_path={folder_path}, include_subfolders={include_subfolders}"
        )

        if not file_id_to_process:
            files_path = self.list_resources(
                sharepoint_site_name=self.sharepoint_site_name,
                sharepoint_host_name=self.sharepoint_host_name,
                sharepoint_relative_url=self.sharepoint_relative_url,
                sharepoint_site_id=self.sharepoint_site_id,
                sharepoint_folder_path=folder_path,
                sharepoint_folder_id=folder_id,
                recursive=include_subfolders,
            )
        else:
            file_path, _ = self.get_file_details_by_id(
                file_id_to_process, self.sharepoint_site_name
            )
            files_path = [file_path]
        metadata = {}

        dispatcher.event(TotalPagesToProcessEvent(total_pages=len(files_path)))

        for file_path in files_path:
            try:
                item = self._get_item_from_path(file_path)
                file_id = item.get("id")
                dispatcher.event(PageDataFetchStartedEvent(page_id=file_id))
                file_metadata = self._download_file(item, download_dir)
                metadata.update(file_metadata)
                dispatcher.event(
                    PageDataFetchCompletedEvent(page_id=file_id, document=None)
                )
            except Exception as e:
                dispatcher.event(PageFailedEvent(page_id=str(file_path), error=str(e)))
                logger.error(f"Error processing {file_path}: {e}", exc_info=True)
                if self.fail_on_error:
                    raise

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
                "lastModifiedDateTime": item.get("fileSystemInfo", {}).get(
                    "lastModifiedDateTime"
                ),
                "createdBy": item.get("createdBy", {}).get("user", {}).get("email", ""),
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
        sharepoint_file_id: Optional[str],
        recursive: bool,
    ) -> Dict[str, str]:
        """
        Downloads files from the specified folder and returns the metadata for the downloaded files.

        Args:
            download_dir (str): The directory where the files should be downloaded.
            sharepoint_site_name (str): The name of the SharePoint site.
            sharepoint_folder_path (str): The path of the folder in the SharePoint site.
            sharepoint_folder_id (str): The ID of the folder in the SharePoint site.
            sharepoint_file_id (str): The ID of a specific file to download.
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
            sharepoint_folder_path,
            sharepoint_file_id,
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

        if self.custom_parser_manager:
            docs = self._load_with_custom_parser_manager(
                files_metadata, download_dir, recursive, get_metadata
            )
        else:
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

    def _load_with_custom_parser_manager(
        self,
        files_metadata: Dict[str, Any],
        download_dir: str,
        recursive: bool,
        get_metadata: Callable[[str], Any],
    ) -> List[Document]:
        """
        Loads documents using the custom parser manager if available.

        Args:
            files_metadata (Dict[str,Any]): A dictionary containing the metadata of the downloaded files.
            download_dir (str): The directory where the files should be downloaded.
            recursive (bool): If True, files from all subfolders are downloaded.
            get_metadata (Callable): Function to get metadata for a file.

        Returns:
            List[Document]: A list containing the documents with metadata.

        """
        docs: List[Document] = []
        for file_path in files_metadata:
            file_name = Path(file_path).name
            ext = Path(file_name).suffix.lower().lstrip(".")
            file_type = None
            for ft in FileType:
                if ft.value == ext:
                    file_type = ft
                    break
            if file_type and file_type in self.custom_parser_manager.custom_parsers:
                with open(file_path, "rb") as f:
                    file_content = f.read()
                markdown = self.custom_parser_manager.process_with_custom_parser(
                    file_type, file_content, ext
                )
                if markdown:
                    doc = Document(text=markdown, metadata=files_metadata[file_path])
                    docs.append(doc)
                    continue
            simple_loader = SimpleDirectoryReader(
                download_dir,
                required_exts=self.required_exts,
                file_extractor=self.file_extractor,
                file_metadata=get_metadata,
                recursive=recursive,
            )
            docs.extend(simple_loader.load_data())
        return docs

    @dispatcher.span
    def load_data(
        self,
        sharepoint_site_name: Optional[str] = None,
        sharepoint_folder_path: Optional[str] = None,
        sharepoint_folder_id: Optional[str] = None,
        recursive: bool = True,
        sharepoint_file_id: Optional[str] = None,
        download_dir: Optional[str] = None,
    ) -> List[Document]:
        """
        Loads data from SharePoint based on sharepoint_type.
        Handles both drive (files/folders) and page types.

        Loads the files from the specified folder in the SharePoint site.

        Args:
            sharepoint_site_name (Optional[str]): The name of the SharePoint site.
            sharepoint_folder_path (Optional[str]): The path of the folder in the SharePoint site.
            sharepoint_folder_id (Optional[str]): The ID of the folder in the SharePoint site.
            sharepoint_file_id (Optional[str]): The ID of a specific file to download.
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
            if not download_dir:
                download_dir = self.custom_folder
            return self.load_pages_data(download_dir=download_dir)

        # If no arguments are provided to load_data, default to the object attributes
        if not sharepoint_site_name:
            sharepoint_site_name = self.sharepoint_site_name
        else:
            self.sharepoint_site_name = sharepoint_site_name

        if not sharepoint_folder_path:
            sharepoint_folder_path = self.sharepoint_folder_path

        if not sharepoint_folder_id:
            sharepoint_folder_id = self.sharepoint_folder_id

        if not sharepoint_file_id:
            sharepoint_file_id = self.sharepoint_file_id

        # Ensure at least one identifier is provided
        if not (
            sharepoint_site_name
            or self.sharepoint_site_id
            or (self.sharepoint_host_name and self.sharepoint_relative_url)
        ):
            raise ValueError(
                "One of sharepoint_site_name, sharepoint_site_id, or both sharepoint_host_name and sharepoint_relative_url must be provided."
            )

        try:
            logger.info(f"Starting document download and metadata extraction")
            # Use download_dir if provided, else custom_folder, else fallback to temp dir
            if not download_dir:
                if self.custom_folder:
                    download_dir = self.custom_folder
                else:
                    with tempfile.TemporaryDirectory() as temp_dir:
                        files_metadata = self._download_files_from_sharepoint(
                            temp_dir,
                            sharepoint_site_name,
                            sharepoint_folder_path,
                            sharepoint_folder_id,
                            sharepoint_file_id,
                            recursive,
                        )
                        logger.info(
                            f"Successfully downloaded {len(files_metadata) if files_metadata else 0} files"
                        )
                        return self._load_documents_with_metadata(
                            files_metadata, temp_dir, recursive
                        )
            # If download_dir is set (by user or custom_folder), use it
            files_metadata = self._download_files_from_sharepoint(
                download_dir,
                sharepoint_site_name,
                sharepoint_folder_path,
                sharepoint_folder_id,
                sharepoint_file_id,
                recursive,
            )
            logger.info(
                f"Successfully downloaded {len(files_metadata) if files_metadata else 0} files"
            )
            return self._load_documents_with_metadata(
                files_metadata, download_dir, recursive
            )

        except Exception as exp:
            logger.error(f"Error accessing SharePoint: {exp}", exc_info=True)
            dispatcher.event(
                PageFailedEvent(
                    page_id=str(sharepoint_folder_path or sharepoint_folder_id),
                    error=str(exp),
                )
            )
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
        response = self._send_get_with_retry(folder_contents_endpoint)
        items = response.json().get("value", [])
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

    def get_file_details_by_id(self, file_id: str, sharepoint_site_name: str):
        """
        Retrieve file details and metadata from a SharePoint site by file ID.

        Args:
            file_id (str): The unique identifier of the file in SharePoint.
            sharepoint_site_name (str): The name of the SharePoint site.

        Returns:
            Tuple[Path, dict] or Tuple[None, None]:
                - A tuple containing the file's path (as a pathlib.Path object) and its metadata dictionary if found.
                - (None, None) if the file details could not be retrieved.

        Raises:
            ValueError: If there is an error retrieving file details from SharePoint.

        Notes:
            - The function retrieves the access token, site ID, and drive ID before making the request.
            - The file path is constructed based on the parent reference and file name.
            - Metadata is extracted and augmented with the file's name.

        """
        access_token = self._get_access_token()

        self._site_id_with_host_name = self._get_site_id_with_host_name(
            access_token, sharepoint_site_name
        )
        self._drive_id = self._get_drive_id()

        file_details_endpoint = (
            f"{self._drive_id_endpoint}/{self._drive_id}/items/{file_id}"
        )
        response = self._send_get_with_retry(file_details_endpoint)

        if not response.ok:
            raise ValueError(
                f"Error retrieving file details for file ID {file_id}: {response.text}"
            )

        file_details = response.json()
        metadata = self._extract_metadata_for_file(file_details)
        metadata["name"] = file_details.get("name", "")
        parent_path = file_details.get("parentReference", {}).get("path", "")
        file_name = file_details.get("name", "")
        from pathlib import Path

        if parent_path and file_name:
            if "root:" in parent_path:
                base_path = parent_path.split("root:")[-1].rstrip("/")
                full_path = f"{base_path}/{file_name}" if base_path else f"/{file_name}"
                return Path(full_path.lstrip("/")), metadata
            else:
                return Path(f"{parent_path}/{file_name}".lstrip("/")), metadata
        elif file_name:
            return Path(file_name), metadata
        else:
            return None, None

    def _list_drive_contents(self) -> List[Path]:
        """
        Helper method to fetch the contents of the drive.

        Returns:
            List[Path]: List of file paths.

        """
        drive_contents_endpoint = (
            f"{self._drive_id_endpoint}/{self._drive_id}/root/children"
        )
        response = self._send_get_with_retry(drive_contents_endpoint)
        items = response.json().get("value", [])

        file_paths = []
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
        sharepoint_host_name: Optional[str] = None,
        sharepoint_relative_url: Optional[str] = None,
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

        if not (
            sharepoint_site_name
            or sharepoint_site_id
            or (sharepoint_host_name and sharepoint_relative_url)
        ):
            raise ValueError(
                "sharepoint_site_name or sharepoint_site_id or (sharepoint_host_name and sharepoint_relative_url) must be provided."
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

    def get_site_pages_list_id(self, site_id: str, token: Optional[str] = None) -> str:
        endpoint = f"https://graph.microsoft.com/v1.0/sites/{site_id}/lists?$filter=displayName eq 'Site Pages'"
        try:
            response = self._send_get_with_retry(endpoint)
            lists = response.json().get("value", [])
            if not lists:
                logger.error("Site Pages list not found for site %s", site_id)
                raise ValueError("Site Pages list not found")
            return lists[0]["id"]
        except Exception as e:
            logger.error(f"Error getting Site Pages list ID: {e}", exc_info=True)
            raise

    def list_pages(self, site_id, token):
        """
        Returns a list of SharePoint site pages with their IDs and names.
        """
        try:
            list_id = self.get_site_pages_list_id(site_id, token)
            endpoint = f"https://graph.microsoft.com/v1.0/sites/{site_id}/lists/{list_id}/items?expand=fields(select=FileLeafRef,CanvasContent1)"
            response = self._send_get_with_retry(endpoint)
            items = response.json().get("value", [])
            pages = []
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

    def get_page_id_by_name(
        self, site_id: str, page_name: str, token: Optional[str] = None
    ) -> Optional[str]:
        """
        Get the ID of a SharePoint page by its name.
        Returns None if the page is not found.
        """
        try:
            list_id = self.get_site_pages_list_id(site_id, token)
            endpoint = f"https://graph.microsoft.com/v1.0/sites/{site_id}/lists/{list_id}/items?expand=fields"
            response = self._send_get_with_retry(endpoint)
            items = response.json().get("value", [])
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

    def get_page_text(self, site_id, list_id, page_id, token):
        """
        Accepts either raw page item id, combined listId_itemId, or will combine internally.
        """
        try:
            raw_page_id = page_id
            if "_" in page_id:
                parts = page_id.split("_", 1)
                if len(parts) == 2:
                    list_id, raw_page_id = parts
            if not list_id:
                list_id = self.get_site_pages_list_id(site_id, token)
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
                "id": f"{list_id}_{raw_page_id}",
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
            download_dir (Optional[str]): Directory to download files to.

        Returns:
            List[Document]: A list of Document objects.

        """
        if not download_dir and self.custom_folder:
            download_dir = self.custom_folder
        if not download_dir:
            raise ValueError(
                "No download directory specified for loading SharePoint pages"
            )

        logger.info(
            f"Loading page data for site {self.sharepoint_site_name} "
            f"(single_page={bool(self.sharepoint_file_id)})"
        )

        try:
            access_token = self._get_access_token()
            site_id = self._get_site_id_with_host_name(
                access_token, self.sharepoint_site_name
            )
            list_id = self.get_site_pages_list_id(site_id, access_token)

            documents: List[Document] = []

            if self.sharepoint_file_id:
                # Specific page
                try:
                    page_info = self.get_page_text(
                        site_id=site_id,
                        list_id=list_id,
                        page_id=self.sharepoint_file_id,
                        token=access_token,
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

            # All pages
            pages = self.list_pages(site_id, access_token)
            dispatcher.event(TotalPagesToProcessEvent(total_pages=len(pages)))
            for page in pages:
                raw_page_id = page["id"]
                combined_id = f"{list_id}_{raw_page_id}"
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
                        token=access_token,
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
