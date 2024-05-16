"""SharePoint files reader."""

import logging
import os
import tempfile
from typing import Any, Dict, List, Union, Optional
from typing import Any, Dict, List, Optional

import requests
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.readers.base import BaseReader, BasePydanticReader
from llama_index.core.schema import Document
from llama_index.core.bridge.pydantic import PrivateAttr, Field

logger = logging.getLogger(__name__)


class SharePointReader(BasePydanticReader):
    """SharePoint reader.


    Reads folders from the SharePoint site from a folder under documents.

    Args:
        client_id (str): The Application ID for the app registered in Microsoft Azure Portal.
            The application must also be configured with MS Graph permissions "Files.ReadAll", "Sites.ReadAll" and BrowserSiteLists.Read.All.
        client_secret (str): The application secret for the app registered in Azure.
        tenant_id (str): Unique identifier of the Azure Active Directory Instance.
        sharepoint_site_name (Optional[str]): The name of the SharePoint site to download from.
        sharepoint_folder_path (Optional[str]): The path of the SharePoint folder to download from.
        sharepoint_folder_id (Optional[str]): The ID of the SharePoint folder to download from. Overrides sharepoint_folder_path.
        file_extractor (Optional[Dict[str, BaseReader]]): A mapping of file extension to a BaseReader class that specifies how to convert that
                                                          file to text. See `SimpleDirectoryReader` for more details.
        attach_permission_metadata (bool): If True, the reader will attach permission metadata to the documents. Set to False if your vector store
                                           only supports flat metadata (i.e. no nested fields or lists), or to avoid the additional API calls.
    """

    client_id: str = None
    client_secret: str = None
    tenant_id: str = None
    sharepoint_site_name: Optional[str] = None
    sharepoint_folder_path: Optional[str] = None
    sharepoint_folder_id: Optional[str] = None
    file_extractor: Optional[Dict[str, Union[str, BaseReader]]] = Field(
        default=None, exclude=True
    )
    attach_permission_metadata: bool = True

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
        file_extractor: Optional[Dict[str, Union[str, BaseReader]]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            client_id=client_id,
            client_secret=client_secret,
            tenant_id=tenant_id,
            sharepoint_site_name=sharepoint_site_name,
            sharepoint_folder_path=sharepoint_folder_path,
            sharepoint_folder_id=sharepoint_folder_id,
            file_extractor=file_extractor,
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

        if response.status_code == 200 and "access_token" in response.json():
            return response.json()["access_token"]

        else:
            logger.error(response.json()["error"])
            raise ValueError(response.json()["error_description"])

    def _get_site_id_with_host_name(self, access_token, sharepoint_site_name) -> str:
        """
        Retrieves the site ID of a SharePoint site using the provided site name.

        Args:
            sharepoint_site_name (str): The name of the SharePoint site.

        Returns:
            str: The ID of the SharePoint site.

        Raises:
            Exception: If the specified SharePoint site is not found.
        """
        site_information_endpoint = (
            f"https://graph.microsoft.com/v1.0/sites?search={sharepoint_site_name}"
        )
        self._authorization_headers = {"Authorization": f"Bearer {access_token}"}

        response = requests.get(
            url=site_information_endpoint,
            headers=self._authorization_headers,
        )

        if response.status_code == 200 and "value" in response.json():
            if (
                len(response.json()["value"]) > 0
                and "id" in response.json()["value"][0]
            ):
                return response.json()["value"][0]["id"]
            else:
                raise ValueError(
                    f"The specified sharepoint site {sharepoint_site_name} is not found."
                )
        else:
            if "error_description" in response.json():
                logger.error(response.json()["error"])
                raise ValueError(response.json()["error_description"])
            raise ValueError(response.json()["error"])

    def _get_drive_id(self) -> str:
        """
        Retrieves the drive ID of the SharePoint site.

        Returns:
            str: The ID of the SharePoint site drive.

        Raises:
            ValueError: If there is an error in obtaining the drive ID.
        """
        self._drive_id_endpoint = f"https://graph.microsoft.com/v1.0/sites/{self._site_id_with_host_name}/drives"

        response = requests.get(
            url=self._drive_id_endpoint,
            headers=self._authorization_headers,
        )

        if response.status_code == 200 and "value" in response.json():
            if (
                len(response.json()["value"]) > 0
                and "id" in response.json()["value"][0]
            ):
                return response.json()["value"][0]["id"]
            else:
                raise ValueError(
                    "Error occurred while fetching the drives for the sharepoint site."
                )
        else:
            logger.error(response.json()["error"])
            raise ValueError(response.json()["error_description"])

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

        response = requests.get(
            url=folder_id_endpoint,
            headers=self._authorization_headers,
        )

        if response.status_code == 200 and "id" in response.json():
            return response.json()["id"]
        else:
            raise ValueError(response.json()["error"])

    def _download_files_and_extract_metadata(
        self,
        folder_id: str,
        download_dir: str,
        current_folder_path: str,
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
        folder_info_endpoint = (
            f"{self._drive_id_endpoint}/{self._drive_id}/items/{folder_id}/children"
        )

        response = requests.get(
            url=folder_info_endpoint,
            headers=self._authorization_headers,
        )

        if response.status_code == 200:
            data = response.json()
            metadata = {}
            for item in data["value"]:
                if include_subfolders and "folder" in item:
                    sub_folder_download_dir = os.path.join(download_dir, item["name"])
                    subfolder_metadata = self._download_files_and_extract_metadata(
                        folder_id=item["id"],
                        download_dir=sub_folder_download_dir,
                        current_folder_path=os.path.join(
                            current_folder_path, item["name"]
                        ),
                        include_subfolders=include_subfolders,
                    )

                    metadata.update(subfolder_metadata)

                elif "file" in item:
                    file_metadata = self._download_file(
                        item, download_dir, current_folder_path
                    )
                    metadata.update(file_metadata)
            return metadata
        else:
            logger.error(response.json()["error"])
            raise ValueError(response.json()["error"])

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
        file_download_url = item["@microsoft.graph.downloadUrl"]
        file_name = item["name"]

        response = requests.get(file_download_url)

        # Create the directory if it does not exist and save the file.
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)
        file_path = os.path.join(download_dir, file_name)
        with open(file_path, "wb") as f:
            f.write(response.content)

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
        response = requests.get(
            url=permissions_info_endpoint,
            headers=self._authorization_headers,
        )
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

        return permissions_dict

    def _extract_metadata_for_file(self, item: Dict[str, Any]) -> Dict[str, str]:
        """
        Extracts metadata related to the file.

        Parameters:
        - item (Dict[str, str]): Dictionary containing file metadata.

        Returns:
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
        sharepoint_folder_path: str,
    ):
        metadata = {}

        file_path = self._download_file_by_url(item, download_dir)
        item["file_path"] = os.path.join(sharepoint_folder_path, item["name"])

        metadata[file_path] = self._extract_metadata_for_file(item)
        return metadata

    def _download_files_from_sharepoint(
        self,
        download_dir: str,
        sharepoint_site_name: str,
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

        if not sharepoint_folder_id:
            sharepoint_folder_id = self._get_sharepoint_folder_id(
                sharepoint_folder_path
            )

        return self._download_files_and_extract_metadata(
            sharepoint_folder_id,
            download_dir,
            os.path.join(sharepoint_site_name, sharepoint_folder_path),
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
    ) -> List[Document]:
        """
        Loads the files from the specified folder in the SharePoint site.

        Args:
            sharepoint_site_name (Optional[str]): The name of the SharePoint site.
            sharepoint_folder_path (Optional[str]): The path of the folder in the SharePoint site.
            recursive (bool): If True, files from all subfolders are downloaded.

        Returns:
            List[Document]: A list containing the documents with metadata.

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

        # TODO: make both of these values optional — and just default to the client ID defaults
        if not sharepoint_site_name:
            raise ValueError("sharepoint_site_name must be provided.")

        if not sharepoint_folder_path and not sharepoint_folder_id:
            raise ValueError(
                "sharepoint_folder_path or sharepoint_folder_id must be provided."
            )

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
