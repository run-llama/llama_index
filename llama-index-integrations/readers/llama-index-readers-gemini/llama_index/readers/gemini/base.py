"""Main GeminiReader implementation for processing PDFs with Gemini API."""

import logging
import os
import tempfile
import time
import warnings
import nest_asyncio
from io import BufferedIOBase
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from llama_index.core.async_utils import asyncio_run
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document
from pydantic import Field, field_validator

from llama_index.readers.gemini.api import GeminiAPI
from llama_index.readers.gemini.cache import CacheManager
from llama_index.readers.gemini.processor import PDFProcessor
from llama_index.readers.gemini.types import FileInput, ProcessingStats
from llama_index.readers.gemini.utils import is_web_url, download_from_url

# Configure logging
logger = logging.getLogger(__name__)

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()


class GeminiReader(BasePydanticReader):
    """
    A PDF reader that uses Google's Gemini API for OCR and intelligent
    document chunking.

    This reader processes PDFs by converting them to images and using Gemini's
    multimodal capabilities to extract text, handle complex layouts, recognize
    tables, forms, and other document structures.
    It produces semantically meaningful chunks optimized for retrieval.

    Features:
    - High-quality OCR with layout preservation
    - Intelligent semantic chunking
    - Table and form recognition
    - Mathematical formula extraction
    - Multi-language support
    - Caching for repeated processing
    - Parallel processing for improved performance
    """

    # API access configuration
    api_key: str = Field(
        description="The API key for the Google Gemini API", validate_default=True
    )
    model_name: str = Field(
        default="gemini-2.0-flash",
        description="The model name to use for OCR and chunking",
    )

    # Processing configuration
    split_by_page: bool = Field(
        default=True, description="Whether to split the document by page"
    )
    verbose: bool = Field(
        default=False, description="Whether to print progress information"
    )
    ignore_errors: bool = Field(
        default=True, description="Whether to ignore errors and continue processing"
    )
    dpi: int = Field(default=300, description="DPI for PDF to image conversion")
    language: str = Field(default="en", description="Primary language of the documents")
    max_workers: int = Field(
        default=4,
        gt=0,
        lt=11,
        description="Maximum number of workers for parallel processing",
    )
    continuous_mode: bool = Field(
        default=False,
        description="Parse documents continuously, maintaining content across\
            page boundaries",
    )
    chunk_size: Union[str, int] = Field(
        default="256-512",
        description="Target size range for chunks in words (e.g., '256-512').\
            Can also be a single number (e.g., 256).",
    )

    # Caching configuration
    enable_caching: bool = Field(
        default=True, description="Whether to cache processed results"
    )
    cache_dir: str = Field(
        default=os.path.join(tempfile.gettempdir(), "gemini_pdf_cache"),
        description="Directory for caching processed results",
    )
    cache_ttl: int = Field(
        default=24 * 60 * 60,  # 24 hours
        description="Time-to-live for cache entries in seconds",
    )

    # Document content extraction options
    extract_forms: bool = Field(
        default=True, description="Whether to apply special handling for forms"
    )
    extract_tables: bool = Field(
        default=True, description="Whether to extract and format tables"
    )
    extract_math_formulas: bool = Field(
        default=True, description="Whether to extract and format mathematical formulas"
    )

    # Retry configuration
    max_retries: int = Field(
        default=3, description="Maximum number of retries for API calls"
    )
    retry_delay: float = Field(
        default=1.0, description="Delay between retries in seconds"
    )

    # Prompt configuration
    system_prompt: str = Field(
        default="",
        description="Template for the prompt to be used with the Gemini API",
    )

    # Private attributes
    _api: Optional[GeminiAPI] = PrivateAttr(default=None)
    _cache_manager: Optional[CacheManager] = PrivateAttr(default=None)
    _pdf_processor: Optional[PDFProcessor] = PrivateAttr(default=None)
    _progress_callback: Optional[Callable[[int, int], None]] = PrivateAttr(default=None)
    _stats: ProcessingStats = PrivateAttr(
        default_factory=lambda: ProcessingStats(start_time=time.time())
    )

    @field_validator("api_key", mode="before")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """
        Validate the API key from parameter or environment variable.

        Args:
            v: API key value

        Returns:
            Valid API key

        Raises:
            ValueError: If API key is not provided and not in environment
        """
        if not v:
            api_key = os.getenv("GOOGLE_API_KEY")
            if api_key is None:
                raise ValueError(
                    "API key not provided",
                    "GOOGLE_API_KEY environment variable not set",
                )
            return api_key
        return v

    def __init__(
        self,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        **kwargs: Any,
    ):
        """
        Initialize the reader with parameters.

        Args:
            progress_callback: Optional callback function for reporting
                progress
            **kwargs: Additional parameters passed to the base class
        """
        super().__init__(**kwargs)

        if self.continuous_mode:
            warnings.warn(
                "Warning: continuous_mode is enabled.\
                    This may result in larger chunk sizes."
            )
        self._progress_callback = progress_callback
        self._initialize_components()

    def _initialize_components(self) -> None:
        """Initialize the reader components."""
        # Initialize API client
        self._api = GeminiAPI(
            api_key=self.api_key,
            model_name=self.model_name,
            extract_forms=self.extract_forms,
            extract_tables=self.extract_tables,
            extract_math_formulas=self.extract_math_formulas,
            language=self.language,
            chunk_size=self.chunk_size,
            max_retries=self.max_retries,
            retry_delay=self.retry_delay,
            verbose=self.verbose,
            prompt=self.system_prompt,
        )

        # Initialize cache manager
        self._cache_manager = CacheManager(
            enable_caching=self.enable_caching,
            cache_dir=self.cache_dir,
            cache_ttl=self.cache_ttl,
            verbose=self.verbose,
        )

        # Initialize PDF processor
        # (will be updated with stats for each processing job)
        self._initialize_pdf_processor()

    def _initialize_pdf_processor(self) -> None:
        """Initialize or reinitialize the PDF processor with current stats."""
        self._pdf_processor = PDFProcessor(
            dpi=self.dpi,
            verbose=self.verbose,
            max_workers=self.max_workers,
            ignore_errors=self.ignore_errors,
            stats=self._stats,
            progress_callback=self._progress_callback,
        )

    def process_pdf(self, pdf_path: str) -> List[Document]:
        """
        Process a PDF file and return a list of Document objects.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of Document objects with extracted text

        Raises:
            FileNotFoundError: If the PDF file cannot be found
            RuntimeError: If processing fails and ignore_errors is False
        """
        # Check cache first
        if self.enable_caching:
            file_hash = self._cache_manager.compute_file_hash(pdf_path)
            cached_docs = self._cache_manager.load_from_cache(file_hash)
            if cached_docs:
                return cached_docs

        # Reset statistics
        self._stats = ProcessingStats(start_time=time.time())
        self._initialize_pdf_processor()  # Reinitialize with new stats

        try:
            # Convert PDF to images
            images = self._pdf_processor.convert_pdf_to_images(pdf_path)

            # Process based on mode
            if self.continuous_mode:
                documents = self._pdf_processor.process_continuous_mode(
                    images, pdf_path, self._api.process_image
                )
            else:
                documents = self._pdf_processor.process_pages_parallel(
                    images, pdf_path, self.split_by_page, self._api.process_image
                )

            # Update and finalize statistics
            self._stats.end_time = time.time()

            if self.verbose:
                logger.info(
                    f"PDF processing complete. Extracted {len(documents)} documents."
                )
                logger.info(f"Processing time: {self._stats.duration:.2f} seconds")
                logger.info(
                    f"Average processing time per page: {self._stats.seconds_per_page:.2f} seconds"
                )
                logger.info(
                    f"Average chunks per page: {self._stats.chunks_per_page:.2f}"
                )
                if self._stats.errors:
                    logger.warning(
                        f"Encountered {len(self._stats.errors)} errors during processing"
                    )

            # Cache the results if enabled
            if self.enable_caching:
                file_hash = self._cache_manager.compute_file_hash(pdf_path)
                self._cache_manager.save_to_cache(file_hash, documents, self._stats)

            return documents

        except Exception as e:
            error_msg = f"Error processing PDF {pdf_path}: {e!s}"
            logger.error(error_msg)
            if self.ignore_errors:
                return []
            else:
                raise RuntimeError(error_msg) from e

    def load_data(
        self,
        file_path: Union[FileInput, List[FileInput]],
        extra_info: Optional[Dict] = None,
    ) -> List[Document]:
        """
        Asynchronously load data from the input file(s).

        Args:
            file_path: Path to PDF file(s), list of paths, or web URL(s)
            extra_info: Extra information to add to document metadata
        Returns:
            List of documents with extracted text.
        """
        return asyncio_run(self.aload_data(file_path, extra_info=extra_info))

    async def aload_data(
        self,
        file_path: Union[FileInput, List[FileInput]],
        extra_info: Optional[Dict] = None,
    ) -> List[Document]:
        """
        Load data from the input file(s).

        Args:
            file_path: Path to PDF file(s), list of paths, or web URL(s)
            extra_info: Extra information to add to document metadata

        Returns:
            List of documents with extracted text
        """
        metadata = extra_info or {}

        # Handle list of files/URLs
        if isinstance(file_path, list):
            all_documents = []
            for i, single_path in enumerate(file_path):
                if self.verbose:
                    logger.info(f"Processing file {i+1}/{len(file_path)}")

                try:
                    file_docs = self.load_data(single_path, extra_info)
                    all_documents.extend(file_docs)
                except Exception as e:
                    if self.verbose:
                        logger.error(
                            f"Error processing file {i+1}/{len(file_path)}: {e!s}"
                        )
                    if not self.ignore_errors:
                        raise

            return all_documents

        # Handle different input types for a single file
        if isinstance(file_path, (str, Path)):
            file_str = str(file_path)

            # Check if the input is a web URL
            if isinstance(file_path, str) and is_web_url(file_path):
                try:
                    # Download the PDF from the URL
                    temp_path = download_from_url(file_path, self.verbose)

                    # Process the downloaded PDF
                    documents = self.process_pdf(temp_path)

                    # Add source URL to metadata
                    for doc in documents:
                        doc.metadata["source_url"] = file_path
                        doc.metadata["source_type"] = "web"
                        if metadata:
                            doc.metadata.update(metadata)

                    # Clean up
                    try:
                        os.unlink(temp_path)
                    except Exception:
                        pass

                    return documents
                except Exception as e:
                    if self.verbose:
                        logger.error(f"Error loading data from URL {file_path}: {e!s}")
                    if self.ignore_errors:
                        return []
                    else:
                        raise

            # Regular file path
            try:
                documents = self.process_pdf(file_str)

                # Add extra metadata if provided
                if metadata:
                    for doc in documents:
                        doc.metadata.update(metadata)

                return documents
            except Exception as e:
                if self.verbose:
                    logger.error(f"Error loading data from {file_str}: {e!s}")
                if self.ignore_errors:
                    return []
                else:
                    raise

        elif isinstance(file_path, (bytes, BufferedIOBase)):
            # For bytes or buffer, save to a temporary file first
            try:
                with tempfile.NamedTemporaryFile(
                    suffix=".pdf", delete=False
                ) as temp_file:
                    temp_path = temp_file.name

                    if isinstance(file_path, bytes):
                        temp_file.write(file_path)
                    else:
                        # It's a buffer
                        temp_file.write(file_path.read())

                documents = self.process_pdf(temp_path)

                # Add file source to metadata
                for doc in documents:
                    doc.metadata["source_type"] = "buffer"
                    if metadata:
                        doc.metadata.update(metadata)

                # Clean up
                os.unlink(temp_path)

                return documents

            except Exception as e:
                if self.verbose:
                    logger.error(f"Error loading data from buffer: {e!s}")
                if self.ignore_errors:
                    return []
                else:
                    raise
        else:
            raise ValueError(
                "file_path must be a string, Path, bytes, BufferedIOBase,\
                    web URL, or a list of these types"
            )

    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the last PDF processing operation.

        Returns:
            Dictionary of processing statistics
        """
        stats = {
            "duration_seconds": self._stats.duration,
            "total_pages": self._stats.total_pages,
            "processed_pages": self._stats.processed_pages,
            "total_chunks": self._stats.total_chunks,
            "seconds_per_page": self._stats.seconds_per_page,
            "chunks_per_page": self._stats.chunks_per_page,
            "error_count": len(self._stats.errors),
        }

        if self._stats.errors:
            # Limit to first 10 errors
            stats["errors"] = self._stats.errors[:10]

        return stats
