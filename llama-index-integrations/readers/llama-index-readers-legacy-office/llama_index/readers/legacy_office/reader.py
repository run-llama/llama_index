"""Legacy Office Reader for LlamaIndex."""

import os
import logging
import requests
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

from fsspec import AbstractFileSystem


from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

logger = logging.getLogger(__name__)


class LegacyOfficeReader(BaseReader):
    """
    Legacy Office Reader for parsing old Office documents (.doc, etc.) using Apache Tika.

    This reader uses Apache Tika to parse legacy Office documents like Word 97 (.doc) files.
    It can use either a local Tika server or connect to a remote one.

    Args:
        tika_server_jar_path (Optional[str]): Path to the Tika server JAR file.
            If not provided, will download and use the default Tika server JAR.
        tika_server_url (Optional[str]): URL of remote Tika server.
            If provided, will use remote server instead of starting local one.
        cache_dir (Optional[str]): Directory to cache the Tika server JAR.
            Defaults to ~/.cache/llama_index/tika
        excluded_embed_metadata_keys (Optional[List[str]]): Metadata keys to exclude from embedding.
        excluded_llm_metadata_keys (Optional[List[str]]): Metadata keys to exclude from LLM.

    """

    def __init__(
        self,
        tika_server_jar_path: Optional[str] = None,
        tika_server_url: Optional[str] = None,
        cache_dir: Optional[str] = None,
        excluded_embed_metadata_keys: Optional[List[str]] = None,
        excluded_llm_metadata_keys: Optional[List[str]] = None,
    ) -> None:
        """Initialize with parameters."""
        super().__init__()

        try:
            import tika
            from tika import parser
        except ImportError as err:
            raise ImportError(
                "`tika` package not found, please run `pip install tika`"
            ) from err

        self.parser = parser
        self.excluded_embed_metadata_keys = excluded_embed_metadata_keys or []
        self.excluded_llm_metadata_keys = excluded_llm_metadata_keys or []

        # Set up cache directory
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.cache/llama_index/tika")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Handle remote server configuration
        if tika_server_url:
            logger.info(f"Using remote Tika server at {tika_server_url}")
            os.environ["TIKA_SERVER_ENDPOINT"] = tika_server_url
            return

        # Set up local Tika server
        if tika_server_jar_path:
            os.environ["TIKA_SERVER_JAR"] = tika_server_jar_path
        else:
            # Use cached JAR if available
            cached_jar = self.cache_dir / "tika-server.jar"
            if cached_jar.exists():
                logger.info("Using cached Tika server JAR")
                os.environ["TIKA_SERVER_JAR"] = str(cached_jar)
            else:
                logger.info("Downloading Tika server JAR (this may take a while)...")
                os.environ["TIKA_SERVER_JAR"] = str(cached_jar)

        # Check if Tika server is already running
        try:
            response = requests.get("http://localhost:9998/version")
            if response.status_code == 200:
                logger.info("Using existing Tika server on port 9998")
                os.environ["TIKA_SERVER_ENDPOINT"] = "http://localhost:9998"
                return
        except requests.RequestException:
            # Server not running, will start it
            pass

        # Initialize Tika
        logger.info("Initializing Tika server...")
        tika.initVM()

        # Set server endpoint
        os.environ["TIKA_SERVER_ENDPOINT"] = "http://localhost:9998"
        logger.info("Tika server will run on port 9998")

    def _process_metadata(
        self, tika_metadata: Dict[str, Any], file_path: str
    ) -> Dict[str, Any]:
        """
        Process Tika metadata into LlamaIndex format.

        Args:
            tika_metadata: Raw metadata from Tika
            file_path: Path to the document

        Returns:
            Processed metadata dictionary with essential information only

        """
        # Start with basic metadata
        metadata = {
            "file_path": file_path,
            "file_name": Path(file_path).name,
            "file_type": Path(file_path).suffix.lower(),
        }

        # Whitelist of metadata keys to keep
        essential_keys = {
            # Document properties
            "title": "title",
            "dc:title": "title",
            "dc:creator": "author",
            "meta:author": "author",
            "meta:word-count": "words",
            "meta:character-count": "chars",
            "meta:page-count": "pages",
            "xmptpg:npages": "pages",
            # Dates
            "dcterms:created": "created",
            "dcterms:modified": "modified",
        }

        for key, orig_value in tika_metadata.items():
            # Skip if not an essential key
            normalized_key = essential_keys.get(key.lower())
            if not normalized_key:
                continue

            # Skip empty values
            if not orig_value:
                continue

            # Handle lists by joining with semicolon
            processed_value = orig_value
            if isinstance(orig_value, list):
                processed_value = "; ".join(str(v) for v in orig_value)

            # Convert to string and clean up
            processed_value = str(processed_value).strip()
            if processed_value and ":" in processed_value:
                processed_value = processed_value.split(":", 1)[1].strip()

            if processed_value:
                metadata[normalized_key] = processed_value

        return metadata

    def load_data(
        self,
        file: Path,
        extra_info: Optional[Dict] = None,
        fs: Optional[AbstractFileSystem] = None,
    ) -> List[Document]:
        """
        Load data from legacy Office documents.

        Args:
            file (Path): Path to the legacy Office document.
            extra_info (Optional[Dict]): Optional dictionary of extra metadata to add.
            fs (Optional[AbstractFileSystem]): Optional filesystem to use.

        Returns:
            List[Document]: List of documents parsed from the file.

        Raises:
            ValueError: If document parsing fails or content is empty.

        """
        try:
            logger.info(f"Parsing document: {file}")
            # Parse the document using Tika
            if fs:
                with fs.open(file) as f:
                    parsed = cast(Dict[str, Any], self.parser.from_buffer(f.read()))
            else:
                parsed = cast(Dict[str, Any], self.parser.from_file(str(file)))

            if parsed is None:
                raise ValueError(f"Failed to parse document: {file}")

            content = str(parsed.get("content", "")).strip()
            if not content:
                raise ValueError(f"No content found in document: {file}")

            # Process metadata
            tika_metadata = parsed.get("metadata", {})
            if not isinstance(tika_metadata, dict):
                tika_metadata = {}

            metadata = self._process_metadata(tika_metadata, str(file))
            if extra_info:
                metadata.update(extra_info)

            # Create document with content and metadata
            doc = Document(
                text=content,
                metadata=metadata,
                excluded_embed_metadata_keys=self.excluded_embed_metadata_keys,
                excluded_llm_metadata_keys=self.excluded_llm_metadata_keys,
            )

            logger.info(f"Successfully parsed document: {file}")
            return [doc]

        except Exception as e:
            logger.error(f"Error processing document {file}: {e!s}")
            raise ValueError(f"Error processing document {file}: {e!s}")
