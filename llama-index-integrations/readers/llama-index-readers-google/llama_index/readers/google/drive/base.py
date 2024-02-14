"""Google Drive files reader."""

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, List, Optional

from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

logger = logging.getLogger(__name__)

# Scope for reading and downloading google drive files
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


class GoogleDriveReader(BaseReader):
    """Google drive reader."""

    def __init__(
        self,
        credentials_path: str = "credentials.json",
        token_path: str = "token.json",
        pydrive_creds_path: str = "creds.txt",
    ) -> None:
        """Initialize with parameters."""
        self.credentials_path = credentials_path
        self.token_path = token_path
        self.pydrive_creds_path = pydrive_creds_path

        self._creds = None
        self._drive = None

        # Download Google Docs/Slides/Sheets as actual files
        # See https://developers.google.com/drive/v3/web/mime-types
        self._mimetypes = {
            "application/vnd.google-apps.document": {
                "mimetype": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "extension": ".docx",
            },
            "application/vnd.google-apps.spreadsheet": {
                "mimetype": (
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                ),
                "extension": ".xlsx",
            },
            "application/vnd.google-apps.presentation": {
                "mimetype": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                "extension": ".pptx",
            },
        }

    def _get_credentials(self) -> Any:
        """Authenticate with Google and save credentials.
        Download the credentials.json file with these instructions: https://developers.google.com/drive/api/v3/quickstart/python.
            Copy credentials.json file and rename it to client_secrets.json file which will be used by pydrive for downloading files.
            So, we need two files:
                1. credentials.json
                2. client_secrets.json
            Both 1, 2 are esentially same but needed with two different names according to google-api-python-client, google-auth-httplib2, google-auth-oauthlib and pydrive libraries.

        Returns:
            credentials, pydrive object.
        """
        from google_auth_oauthlib.flow import InstalledAppFlow
        from pydrive.auth import GoogleAuth
        from pydrive.drive import GoogleDrive

        from google.auth.transport.requests import Request
        from google.oauth2 import service_account
        from google.oauth2.credentials import Credentials

        # First, we need the Google API credentials for the app
        creds = None
        if os.path.exists(self.token_path):
            creds = Credentials.from_authorized_user_file(self.token_path, SCOPES)
        elif os.path.exists(self.credentials_path):
            creds = service_account.Credentials.from_service_account_file(
                self.credentials_path, scopes=SCOPES
            )
            gauth = GoogleAuth()
            gauth.credentials = creds
            drive = GoogleDrive(gauth)
            return creds, drive
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_path, SCOPES
                )
                creds = flow.run_local_server(port=0)
            # Save the credentials for the next run
            with open(self.token_path, "w") as token:
                token.write(creds.to_json())

        # Next, we need user authentication to download files (via pydrive)
        # Uses client_secrets.json file for authorization.
        gauth = GoogleAuth()
        # Try to load saved client credentials
        gauth.LoadCredentialsFile(self.pydrive_creds_path)
        if gauth.credentials is None:
            # Authenticate if they're not there
            gauth.LocalWebserverAuth()
        elif gauth.access_token_expired:
            # Refresh them if expired
            gauth.Refresh()
        else:
            # Initialize the saved creds
            gauth.Authorize()
        # Save the current credentials to a file so user doesn't have to auth every time
        gauth.SaveCredentialsFile(self.pydrive_creds_path)

        drive = GoogleDrive(gauth)

        return creds, drive

    def _get_fileids_meta(
        self,
        folder_id: Optional[str] = None,
        file_id: Optional[str] = None,
        mime_types: Optional[list] = None,
    ) -> List[List[str]]:
        """Get file ids present in folder/ file id
        Args:
            folder_id: folder id of the folder in google drive.
            file_id: file id of the file in google drive
            mime_types: the mimeTypes you want to allow e.g.: "application/vnd.google-apps.document"
        Returns:
            metadata: List of metadata of filde ids.
        """
        from googleapiclient.discovery import build

        try:
            service = build("drive", "v3", credentials=self._creds)
            fileids_meta = []
            if folder_id:
                folder_mime_type = "application/vnd.google-apps.folder"
                query = "'" + folder_id + "' in parents"

                # Add mimeType filter to query
                if mime_types:
                    if folder_mime_type not in mime_types:
                        mime_types.append(folder_mime_type)  # keep the recursiveness
                    mime_query = " or ".join(
                        [f"mimeType='{mime_type}'" for mime_type in mime_types]
                    )
                    query += f" and ({mime_query})"

                results = (
                    service.files()
                    .list(
                        q=query,
                        includeItemsFromAllDrives=True,
                        supportsAllDrives=True,
                        fields="*",
                    )
                    .execute()
                )
                items = results.get("files", [])
                for item in items:
                    if item["mimeType"] == folder_mime_type:
                        fileids_meta.extend(
                            self._get_fileids_meta(
                                folder_id=item["id"], mime_types=mime_types
                            )
                        )
                    else:
                        # Check if file doesn't belong to a Shared Drive. "owners" doesn't exist in a Shared Drive
                        is_shared_drive = "driveId" in item
                        author = (
                            item["owners"][0]["displayName"]
                            if not is_shared_drive
                            else "Shared Drive"
                        )

                        fileids_meta.append(
                            (
                                item["id"],
                                author,
                                item["name"],
                                item["mimeType"],
                                item["createdTime"],
                                item["modifiedTime"],
                            )
                        )

            else:
                # Get the file details
                file = (
                    service.files()
                    .get(fileId=file_id, supportsAllDrives=True, fields="*")
                    .execute()
                )
                # Get metadata of the file
                # Check if file doesn't belong to a Shared Drive. "owners" doesn't exist in a Shared Drive
                is_shared_drive = "driveId" in file
                author = (
                    file["owners"][0]["displayName"]
                    if not is_shared_drive
                    else "Shared Drive"
                )

                fileids_meta.append(
                    (
                        file["id"],
                        author,
                        file["name"],
                        file["mimeType"],
                        file["createdTime"],
                        file["modifiedTime"],
                    )
                )
            return fileids_meta

        except Exception as e:
            logger.error(f"An error occurred while getting fileids metadata: {e}")

    def _download_file(self, fileid: str, filename: str) -> str:
        """Download the file with fileid and filename
        Args:
            fileid: file id of the file in google drive
            filename: filename with which it will be downloaded
        Returns:
            The downloaded filename, which which may have a new extension.
        """
        from io import BytesIO

        from googleapiclient.discovery import build
        from googleapiclient.http import MediaIoBaseDownload

        try:
            # Get file details
            service = build("drive", "v3", credentials=self._creds)
            file = service.files().get(fileId=fileid, supportsAllDrives=True).execute()

            if file["mimeType"] in self._mimetypes:
                download_mimetype = self._mimetypes[file["mimeType"]]["mimetype"]
                download_extension = self._mimetypes[file["mimeType"]]["extension"]
                new_file_name = filename + download_extension

                # Download and convert file
                request = service.files().export_media(
                    fileId=fileid, mimeType=download_mimetype
                )
            else:
                new_file_name = filename

                # Download file without conversion
                request = service.files().get_media(fileId=fileid)

            # Download file data
            file_data = BytesIO()
            downloader = MediaIoBaseDownload(file_data, request)
            done = False

            while not done:
                status, done = downloader.next_chunk()

            # Save the downloaded file
            with open(new_file_name, "wb") as f:
                f.write(file_data.getvalue())

            return new_file_name
        except Exception as e:
            logger.error(f"An error occurred while downloading file: {e}")

    def _load_data_fileids_meta(self, fileids_meta: List[List[str]]) -> List[Document]:
        """Load data from fileids metadata
        Args:
            fileids_meta: metadata of fileids in google drive.

        Returns:
            Lis[Document]: List of Document of data present in fileids.
        """
        try:
            with tempfile.TemporaryDirectory() as temp_dir:

                def get_metadata(filename):
                    return metadata[filename]

                temp_dir = Path(temp_dir)
                metadata = {}

                for fileid_meta in fileids_meta:
                    filename = fileid_meta[2]
                    filepath = os.path.join(temp_dir, filename)
                    fileid = fileid_meta[0]
                    final_filepath = self._download_file(fileid, filepath)

                    metadata[final_filepath] = {
                        "file id": fileid_meta[0],
                        "author": fileid_meta[1],
                        "file name": fileid_meta[2],
                        "mime type": fileid_meta[3],
                        "created at": fileid_meta[4],
                        "modified at": fileid_meta[5],
                    }
                loader = SimpleDirectoryReader(temp_dir, file_metadata=get_metadata)
                documents = loader.load_data()
                for doc in documents:
                    doc.id_ = doc.metadata.get("file id", doc.id_)

            return documents
        except Exception as e:
            logger.error(f"An error occurred while loading data from fileids meta: {e}")

    def _load_from_file_ids(
        self, file_ids: List[str], mime_types: list
    ) -> List[Document]:
        """Load data from file ids
        Args:
            file_ids: file ids of the files in google drive.

        Returns:
            Document: List of Documents of text.
        """
        try:
            fileids_meta = []
            for file_id in file_ids:
                fileids_meta.extend(
                    self._get_fileids_meta(file_id=file_id, mime_types=mime_types)
                )
            return self._load_data_fileids_meta(fileids_meta)
        except Exception as e:
            logger.error(f"An error occurred while loading with fileid: {e}")

    def _load_from_folder(self, folder_id: str, mime_types: list) -> List[Document]:
        """Load data from folder_id
        Args:
            folder_id: folder id of the folder in google drive.
            mime_types: the mimeTypes you want to allow e.g.: "application/vnd.google-apps.document"
        Returns:
            Document: List of Documents of text.
        """
        try:
            fileids_meta = self._get_fileids_meta(
                folder_id=folder_id, mime_types=mime_types
            )
            return self._load_data_fileids_meta(fileids_meta)
        except Exception as e:
            logger.error(f"An error occurred while loading from folder: {e}")

    def load_data(
        self,
        folder_id: str = None,
        file_ids: List[str] = None,
        mime_types: List[str] = None,
    ) -> List[Document]:
        """Load data from the folder id and file ids.

        Args:
            folder_id: folder id of the folder in google drive.
            file_ids: file ids of the files in google drive.
            mime_types: the mimeTypes you want to allow e.g.: "application/vnd.google-apps.document"
        Returns:
            List[Document]: A list of documents.
        """
        self._creds, self._drive = self._get_credentials()

        if folder_id:
            return self._load_from_folder(folder_id, mime_types)
        else:
            return self._load_from_file_ids(file_ids, mime_types)
