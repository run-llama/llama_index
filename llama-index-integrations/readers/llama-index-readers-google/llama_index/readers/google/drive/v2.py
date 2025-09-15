"""Google Drive files reader V2 - Optimized for batch operations."""

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.schema import Document

from .base import GoogleDriveReader

logger = logging.getLogger(__name__)


class GoogleDriveReaderV2(GoogleDriveReader):
    """
    Google Drive Reader V2 - Optimized Version.

    This version optimizes the original GoogleDriveReader by:
    1. Reducing API calls through batch operations
    2. Eliminating redundant file metadata retrievals
    3. Caching file information to avoid repeated API calls
    4. Building file paths from list responses instead of individual get calls

    Inherits all the same parameters and functionality as GoogleDriveReader
    but with improved performance for large numbers of files.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize with same parameters as base GoogleDriveReader."""
        super().__init__(**kwargs)
        # Cache for file metadata to avoid redundant API calls
        self._file_metadata_cache: Dict[str, Dict[str, Any]] = {}
        self._folder_path_cache: Dict[str, str] = {}

    @classmethod
    def class_name(cls) -> str:
        return "GoogleDriveReaderV2"

    def _build_folder_path_cache(
        self, service, root_folder_id: Optional[str] = None
    ) -> None:
        """
        Build a cache of folder paths to avoid individual API calls.
        This fetches all folders in one or few API calls and builds the path hierarchy.
        """
        if not root_folder_id:
            return

        try:
            # Query to get all folders that are descendants of root_folder_id
            query = f"'{root_folder_id}' in parents and mimeType='application/vnd.google-apps.folder'"

            folders = []
            page_token = ""

            # Get all folders in the hierarchy
            while True:
                if self.drive_id:
                    results = (
                        service.files()
                        .list(
                            q=query,
                            driveId=self.drive_id,
                            corpora="drive",
                            includeItemsFromAllDrives=True,
                            supportsAllDrives=True,
                            fields="files(id, name, parents)",
                            pageToken=page_token,
                        )
                        .execute()
                    )
                else:
                    results = (
                        service.files()
                        .list(
                            q=query,
                            includeItemsFromAllDrives=True,
                            supportsAllDrives=True,
                            fields="files(id, name, parents)",
                            pageToken=page_token,
                        )
                        .execute()
                    )

                folders.extend(results.get("files", []))
                page_token = results.get("nextPageToken", None)
                if page_token is None:
                    break

            # Build the path cache by resolving parent relationships
            # Start with root folder
            try:
                root_folder = (
                    service.files()
                    .get(fileId=root_folder_id, supportsAllDrives=True, fields="name")
                    .execute()
                )
                self._folder_path_cache[root_folder_id] = root_folder["name"]
            except Exception as e:
                logger.debug(f"Could not get root folder name: {e}")
                self._folder_path_cache[root_folder_id] = ""

            # Recursively build paths for all folders
            def build_path(folder_id: str, visited: set) -> str:
                if folder_id in visited:
                    return self._folder_path_cache.get(folder_id, "")

                if folder_id in self._folder_path_cache:
                    return self._folder_path_cache[folder_id]

                visited.add(folder_id)

                # Find the folder in our list
                folder = next((f for f in folders if f["id"] == folder_id), None)
                if not folder:
                    return ""

                parents = folder.get("parents", [])
                if not parents or parents[0] == root_folder_id:
                    # Direct child of root
                    path = f"{self._folder_path_cache[root_folder_id]}/{folder['name']}"
                else:
                    # Build parent path first
                    parent_path = build_path(parents[0], visited)
                    path = (
                        f"{parent_path}/{folder['name']}"
                        if parent_path
                        else folder["name"]
                    )

                self._folder_path_cache[folder_id] = path
                return path

            # Build paths for all folders
            for folder in folders:
                if folder["id"] not in self._folder_path_cache:
                    build_path(folder["id"], set())

        except Exception as e:
            logger.debug(f"Could not build folder path cache: {e}")

    def _get_relative_path_optimized(
        self,
        file_id: str,
        file_name: str,
        parents: List[str],
        root_folder_id: Optional[str] = None,
    ) -> str:
        """
        Get relative path using cached folder information instead of individual API calls.
        """
        if not root_folder_id or not parents:
            return file_name

        parent_id = parents[0]

        # If parent is root folder, return just the filename
        if parent_id == root_folder_id:
            return file_name

        # Use cached path if available
        if parent_id in self._folder_path_cache:
            parent_path = self._folder_path_cache[parent_id]
            return f"{parent_path}/{file_name}"

        # Fallback to original method if not in cache
        return file_name

    def _get_fileids_meta_optimized(
        self,
        drive_id: Optional[str] = None,
        folder_id: Optional[str] = None,
        file_id: Optional[str] = None,
        mime_types: Optional[List[str]] = None,
        query_string: Optional[str] = None,
        current_path: Optional[str] = None,
    ) -> List[List[str]]:
        """
        Optimized version that reduces API calls by fetching comprehensive file information
        in fewer requests and using cached data.
        """
        from googleapiclient.discovery import build

        fileids_meta = []
        try:
            service = build("drive", "v3", credentials=self._creds)

            if folder_id and not file_id:
                # Build folder path cache first to avoid individual get calls
                self._build_folder_path_cache(service, folder_id)

                try:
                    folder = (
                        service.files()
                        .get(fileId=folder_id, supportsAllDrives=True, fields="name")
                        .execute()
                    )
                    current_path = (
                        f"{current_path}/{folder['name']}"
                        if current_path
                        else folder["name"]
                    )
                except Exception as e:
                    logger.warning(f"Could not get folder name: {e}")

                folder_mime_type = "application/vnd.google-apps.folder"
                query = "('" + folder_id + "' in parents)"

                # Add mimeType filter to query
                if mime_types:
                    if folder_mime_type not in mime_types:
                        mime_types.append(folder_mime_type)  # keep the recursiveness
                    mime_query = " or ".join(
                        [f"mimeType='{mime_type}'" for mime_type in mime_types]
                    )
                    query += f" and ({mime_query})"

                # Add query string filter
                if query_string:
                    query += (
                        f" and ((mimeType='{folder_mime_type}') or ({query_string}))"
                    )

                items = []
                page_token = ""

                # Fetch ALL file information in one go with comprehensive fields
                # This eliminates the need for individual get() calls later
                while True:
                    if drive_id:
                        results = (
                            service.files()
                            .list(
                                q=query,
                                driveId=drive_id,
                                corpora="drive",
                                includeItemsFromAllDrives=True,
                                supportsAllDrives=True,
                                fields="files(id, name, mimeType, parents, owners, createdTime, modifiedTime, driveId, size)",
                                pageToken=page_token,
                            )
                            .execute()
                        )
                    else:
                        results = (
                            service.files()
                            .list(
                                q=query,
                                includeItemsFromAllDrives=True,
                                supportsAllDrives=True,
                                fields="files(id, name, mimeType, parents, owners, createdTime, modifiedTime, driveId, size)",
                                pageToken=page_token,
                            )
                            .execute()
                        )
                    items.extend(results.get("files", []))
                    page_token = results.get("nextPageToken", None)
                    if page_token is None:
                        break

                # Cache all file metadata to avoid future API calls
                for item in items:
                    self._file_metadata_cache[item["id"]] = item

                # Process items using cached data
                for item in items:
                    # Use optimized path calculation
                    item_path = self._get_relative_path_optimized(
                        item["id"], item["name"], item.get("parents", []), folder_id
                    )

                    if current_path and item_path != item["name"]:
                        # Only prepend current_path if we don't already have a full path
                        if not item_path.startswith(current_path):
                            item_path = f"{current_path}/{item['name']}"
                    elif current_path:
                        item_path = f"{current_path}/{item['name']}"

                    if item["mimeType"] == folder_mime_type:
                        # Recursive call for folders
                        if drive_id:
                            fileids_meta.extend(
                                self._get_fileids_meta_optimized(
                                    drive_id=drive_id,
                                    folder_id=item["id"],
                                    mime_types=mime_types,
                                    query_string=query_string,
                                    current_path=current_path,
                                )
                            )
                        else:
                            fileids_meta.extend(
                                self._get_fileids_meta_optimized(
                                    folder_id=item["id"],
                                    mime_types=mime_types,
                                    query_string=query_string,
                                    current_path=current_path,
                                )
                            )
                    else:
                        # Process regular files
                        is_shared_drive = "driveId" in item
                        author = (
                            item["owners"][0]["displayName"]
                            if not is_shared_drive and item.get("owners")
                            else "Shared Drive"
                        )
                        fileids_meta.append(
                            (
                                item["id"],
                                author,
                                item_path,
                                item["mimeType"],
                                item["createdTime"],
                                item["modifiedTime"],
                                self._get_drive_link(item["id"]),
                            )
                        )
            else:
                # Handle single file case - use cached data if available
                if file_id in self._file_metadata_cache:
                    file = self._file_metadata_cache[file_id]
                else:
                    # Only make API call if not cached
                    file = (
                        service.files()
                        .get(fileId=file_id, supportsAllDrives=True, fields="*")
                        .execute()
                    )
                    self._file_metadata_cache[file_id] = file

                # Get metadata of the file
                is_shared_drive = "driveId" in file
                author = (
                    file["owners"][0]["displayName"]
                    if not is_shared_drive and file.get("owners")
                    else "Shared Drive"
                )

                # Use optimized path calculation
                file_path = self._get_relative_path_optimized(
                    file_id,
                    file["name"],
                    file.get("parents", []),
                    folder_id or self.folder_id,
                )

                fileids_meta.append(
                    (
                        file["id"],
                        author,
                        file_path,
                        file["mimeType"],
                        file["createdTime"],
                        file["modifiedTime"],
                        self._get_drive_link(file["id"]),
                    )
                )
            return fileids_meta

        except Exception as e:
            if self.raise_errors:
                raise
            else:
                logger.error(
                    f"An error occurred while getting fileids metadata: {e}",
                    exc_info=True,
                )
            return fileids_meta

    def _download_file_optimized(self, fileid: str, filename: str) -> str:
        """
        Optimized file download that uses cached file metadata instead of making
        an additional API call to get file details.
        """
        from io import BytesIO
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaIoBaseDownload

        try:
            service = build("drive", "v3", credentials=self._creds)

            # Use cached file metadata if available, otherwise fetch it
            if fileid in self._file_metadata_cache:
                file = self._file_metadata_cache[fileid]
            else:
                file = (
                    service.files().get(fileId=fileid, supportsAllDrives=True).execute()
                )
                self._file_metadata_cache[fileid] = file

            if file["mimeType"] in self._mimetypes:
                download_mimetype = self._mimetypes[file["mimeType"]]["mimetype"]
                download_extension = self._mimetypes[file["mimeType"]]["extension"]
                new_file_name = filename + download_extension

                # Download and convert file
                request = service.files().export_media(
                    fileId=fileid, mimeType=download_mimetype
                )
            else:
                # we should have a file extension to allow the readers to work
                _, download_extension = os.path.splitext(file.get("name", ""))
                new_file_name = filename + download_extension

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
            if self.raise_errors:
                raise
            else:
                logger.error(
                    f"An error occurred while downloading file: {e}", exc_info=True
                )

    def _load_data_fileids_meta(self, fileids_meta: List[List[str]]) -> List[Document]:
        """
        Optimized version that uses cached file metadata and optimized download.
        """
        try:
            with tempfile.TemporaryDirectory() as temp_dir:

                def get_metadata(filename):
                    return metadata[filename]

                temp_dir = Path(temp_dir)
                metadata = {}

                for fileid_meta in fileids_meta:
                    # Download files and name them with their fileid
                    fileid = fileid_meta[0]
                    filepath = os.path.join(temp_dir, fileid)
                    final_filepath = self._download_file_optimized(fileid, filepath)

                    # Add metadata of the file to metadata dictionary
                    metadata[final_filepath] = {
                        "file id": fileid_meta[0],
                        "author": fileid_meta[1],
                        "file path": fileid_meta[2],
                        "mime type": fileid_meta[3],
                        "created at": fileid_meta[4],
                        "modified at": fileid_meta[5],
                    }
                loader = SimpleDirectoryReader(
                    temp_dir,
                    file_extractor=self.file_extractor,
                    file_metadata=get_metadata,
                )
                documents = loader.load_data()
                for doc in documents:
                    doc.id_ = doc.metadata.get("file id", doc.id_)

            return documents
        except Exception as e:
            if self.raise_errors:
                raise
            else:
                logger.error(
                    f"An error occurred while loading data from fileids meta: {e}",
                    exc_info=True,
                )

    def _load_from_folder(
        self,
        drive_id: Optional[str],
        folder_id: str,
        mime_types: Optional[List[str]],
        query_string: Optional[str],
    ) -> List[Document]:
        """
        Optimized folder loading using the new optimized methods.
        """
        try:
            fileids_meta = self._get_fileids_meta_optimized(
                drive_id=drive_id,
                folder_id=folder_id,
                mime_types=mime_types,
                query_string=query_string,
            )
            return self._load_data_fileids_meta(fileids_meta)
        except Exception as e:
            if self.raise_errors:
                raise
            else:
                logger.error(
                    f"An error occurred while loading from folder: {e}", exc_info=True
                )

    def _load_from_file_ids(
        self,
        drive_id: Optional[str],
        file_ids: List[str],
        mime_types: Optional[List[str]],
        query_string: Optional[str],
    ) -> List[Document]:
        """
        Optimized file IDs loading using the new optimized methods.
        """
        try:
            fileids_meta = []
            for file_id in file_ids:
                fileids_meta.extend(
                    self._get_fileids_meta_optimized(
                        drive_id=drive_id,
                        file_id=file_id,
                        mime_types=mime_types,
                        query_string=query_string,
                    )
                )
            return self._load_data_fileids_meta(fileids_meta)
        except Exception as e:
            if self.raise_errors:
                raise
            else:
                logger.error(
                    f"An error occurred while loading with fileid: {e}", exc_info=True
                )

    def list_resources(self, **kwargs) -> List[str]:
        """Optimized resource listing using cached metadata."""
        self._creds = self._get_credentials()

        drive_id = kwargs.get("drive_id", self.drive_id)
        folder_id = kwargs.get("folder_id", self.folder_id)
        file_ids = kwargs.get("file_ids", self.file_ids)
        query_string = kwargs.get("query_string", self.query_string)

        if folder_id:
            fileids_meta = self._get_fileids_meta_optimized(
                drive_id, folder_id, query_string=query_string
            )
        elif file_ids:
            fileids_meta = []
            for file_id in file_ids:
                fileids_meta.extend(
                    self._get_fileids_meta_optimized(
                        drive_id, file_id=file_id, query_string=query_string
                    )
                )
        else:
            raise ValueError("Either 'folder_id' or 'file_ids' must be provided.")

        return [meta[0] for meta in fileids_meta]  # Return list of file IDs

    def get_resource_info(self, resource_id: str, **kwargs) -> Dict:
        """Optimized resource info retrieval using cached metadata."""
        self._creds = self._get_credentials()

        # Try to use cached data first
        if resource_id in self._file_metadata_cache:
            file = self._file_metadata_cache[resource_id]
            is_shared_drive = "driveId" in file
            author = (
                file["owners"][0]["displayName"]
                if not is_shared_drive and file.get("owners")
                else "Shared Drive"
            )
            return {
                "file_path": file["name"],  # Use name since we may not have full path
                "file_size": file.get("size"),
                "last_modified_date": file["modifiedTime"],
                "content_hash": None,
                "content_type": file["mimeType"],
                "author": author,
                "created_date": file["createdTime"],
                "drive_link": self._get_drive_link(resource_id),
            }

        # Fallback to original method
        fileids_meta = self._get_fileids_meta_optimized(file_id=resource_id)
        if not fileids_meta:
            raise ValueError(f"Resource with ID {resource_id} not found.")

        meta = fileids_meta[0]
        return {
            "file_path": meta[2],
            "file_size": None,
            "last_modified_date": meta[5],
            "content_hash": None,
            "content_type": meta[3],
            "author": meta[1],
            "created_date": meta[4],
            "drive_link": meta[6],
        }

    def load_resource(self, resource_id: str, **kwargs) -> List[Document]:
        """Load a specific resource using optimized methods."""
        return self._load_from_file_ids(
            self.drive_id, [resource_id], None, self.query_string
        )

    def read_file_content(self, file_path: Union[str, Path], **kwargs) -> bytes:
        """Read file content using optimized download method."""
        self._creds = self._get_credentials()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = os.path.join(temp_dir, "temp_file")
            downloaded_file = self._download_file_optimized(str(file_path), temp_file)
            with open(downloaded_file, "rb") as file:
                return file.read()

    def clear_cache(self) -> None:
        """Clear internal caches. Useful for long-running applications."""
        self._file_metadata_cache.clear()
        self._folder_path_cache.clear()
