"""PDF processing and image conversion logic for Gemini PDF Reader."""

import asyncio
import logging
import os
from typing import Callable, List, Optional, Tuple

import PIL.Image
from pdf2image import convert_from_path

from llama_index.core.schema import Document
from llama_index.readers.gemini.types import ProcessingStats

# Configure logging
logger = logging.getLogger(__name__)


class PDFProcessor:
    """Handles PDF to image conversion and document processing."""

    def __init__(
        self,
        dpi: int,
        verbose: bool,
        max_workers: int,
        ignore_errors: bool,
        stats: ProcessingStats,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ):
        """
        Initialize the PDF processor.

        Args:
            dpi: DPI for PDF to image conversion
            verbose: Whether to print verbose output
            max_workers: Maximum number of parallel workers
            ignore_errors: Whether to ignore errors
            stats: Processing statistics object
            progress_callback: Optional callback function for reporting
                progress
        """
        self.dpi = dpi
        self.verbose = verbose
        self.max_workers = max_workers
        self.ignore_errors = ignore_errors
        self._stats = stats
        self._progress_callback = progress_callback

    def _update_progress(self, current: int, total: int) -> None:
        """
        Update progress and call the progress callback if provided.

        Args:
            current: Current progress value
            total: Total expected operations
        """
        if self._progress_callback:
            self._progress_callback(current, total)

        if self.verbose:
            progress_pct = int(100 * current / total) if total > 0 else 0
            logger.info(f"Progress: {progress_pct}% ({current}/{total})")

    def convert_pdf_to_images(self, pdf_path: str) -> List[PIL.Image.Image]:
        """
        Convert a PDF file to a list of PIL images.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of PIL images

        Raises:
            FileNotFoundError: If the PDF file cannot be found
            RuntimeError: If PDF conversion fails
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        if self.verbose:
            logger.info(f"Converting PDF to images: {pdf_path} (DPI: {self.dpi})")

        try:
            # Convert PDF to images using pdf2image
            images = convert_from_path(
                pdf_path=pdf_path,
                dpi=self.dpi,
                fmt="png",
                thread_count=os.cpu_count() or 4,
                use_cropbox=True,
                use_pdftocairo=True,
                grayscale=False,
                strict=False,
            )
            self._stats.total_pages = len(images)
            if self.verbose:
                logger.info(f"Converted PDF to {len(images)} images")
            return images
        except Exception as e:
            logger.error(f"PDF conversion error: {e!s}")
            self._stats.errors.append(f"PDF conversion error: {e!s}")
            raise RuntimeError(f"Failed to convert PDF: {e!s}") from e

    def update_stats(self, chunks_count: int) -> None:
        """
        Update processing statistics.

        Args:
            chunks_count: Number of chunks processed
        """
        self._stats.processed_pages += 1
        self._stats.total_chunks += chunks_count

        # Update progress
        self._update_progress(self._stats.processed_pages, self._stats.total_pages)

    async def process_pages_parallel(
        self,
        images: List[PIL.Image.Image],
        pdf_path: str,
        split_by_page: bool,
        process_image_func: Callable,
    ) -> List[Document]:
        """
        Process multiple pages in parallel.

        Args:
            images: List of page images
            pdf_path: Path to the PDF file
            split_by_page: Whether to split by page
            process_image_func: Function to process each image

        Returns:
            Documents extracted from all pages
        """
        total_pages = len(images)

        # Create arguments for each page
        page_args = [
            (image, pdf_path, i, total_pages) for i, image in enumerate(images, start=1)
        ]

        all_documents = []

        semaphore = asyncio.Semaphore(self.max_workers)
        tasks = []

        async def process_page_with_semaphore(args):
            async with semaphore:
                return await asyncio.to_thread(
                    self._process_page, args, split_by_page, process_image_func
                )

        tasks = [process_page_with_semaphore(args) for args in page_args]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                if not self.ignore_errors:
                    raise RuntimeError(f"Error processing page: {result!s}") from result
            else:
                all_documents.extend(result)

        # with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
        #     future_to_page = {
        #         executor.submit(
        #             self._process_page, args, split_by_page, process_image_func
        #         ): args
        #         for args in page_args
        #     }

        #     for future in as_completed(future_to_page):
        #         try:
        #             page_documents = future.result()
        #             all_documents.extend(page_documents)
        #         except Exception as e:
        #             if not self.ignore_errors:
        #                 raise RuntimeError(f"Error processing page: {e!s}") from e

        # Sort documents by page number and chunk number if needed
        if split_by_page:
            all_documents.sort(
                key=lambda doc: (
                    doc.metadata.get("page_number", 0),
                    doc.metadata.get("chunk_number_in_page", 0),
                )
            )
        else:
            all_documents.sort(key=lambda doc: doc.metadata.get("page_number", 0))

        return all_documents

    def _process_page(
        self,
        args: Tuple[PIL.Image.Image, str, int, int],
        split_by_page: bool,
        process_image_func: Callable,
    ) -> List[Document]:
        """
        Process a single page of the PDF.

        Args:
            args: Tuple containing (image, pdf_path, page_number, total_pages)
            split_by_page: Whether to split by page
            process_image_func: Function to process each image

        Returns:
            Documents extracted from the page

        Raises:
            RuntimeError: If page processing fails and ignore_errors is False
        """
        image, pdf_path, page_number, total_pages = args
        documents = []

        try:
            # Process the image
            text_chunks = process_image_func(image)

            # Create Document objects
            if split_by_page:
                # Create a separate Document for each chunk
                for chunk_number, chunk in enumerate(text_chunks, start=1):
                    # Create metadata
                    metadata = {
                        "file_path": pdf_path,
                        "file_name": os.path.basename(pdf_path),
                        "page_number": page_number,
                        "chunk_number_in_page": chunk_number,
                        "total_pages": total_pages,
                        "total_chunks_in_page": len(text_chunks),
                        "source_type": "pdf",
                    }

                    doc = Document(
                        text=chunk.text,
                        metadata=metadata,
                        excluded_llm_metadata_keys=[
                            "file_path",
                            "file_name",
                            "page_number",
                            "chunk_number_in_page",
                            "total_pages",
                            "total_chunks_in_page",
                            "source_type",
                        ],
                    )
                    documents.append(doc)
            else:
                # Create a single Document per page with all chunks
                combined_text = "\n\n".join([chunk.text for chunk in text_chunks])

                # Create metadata
                metadata = {
                    "file_path": pdf_path,
                    "file_name": os.path.basename(pdf_path),
                    "page_number": page_number,
                    "total_pages": total_pages,
                    "source_type": "pdf",
                }

                doc = Document(
                    text=combined_text,
                    metadata=metadata,
                    excluded_llm_metadata_keys=[
                        "file_path",
                        "file_name",
                        "page_number",
                        "total_pages",
                        "source_type",
                    ],
                )
                documents.append(doc)

            self.update_stats(len(text_chunks))

            return documents

        except Exception as e:
            error_msg = f"Error processing page {page_number}: {e!s}"
            logger.error(error_msg)
            self._stats.errors.append(error_msg)

            if self.ignore_errors:
                return []
            else:
                raise RuntimeError(error_msg) from e

    async def process_continuous_mode(
        self,
        images: List[PIL.Image.Image],
        pdf_path: str,
        process_image_func: Callable,
    ) -> List[Document]:
        """
        Process the PDF in continuous mode with parallel extraction.

        In continuous mode, content that spans across pages is treated
            as continuous.

        Args:
            images: List of page images
            pdf_path: Path to the PDF file
            process_image_func: Function to process each image

        Returns:
            Documents extracted from the PDF
        """
        if self.verbose:
            logger.info(f"Processing PDF in continuous mode with {len(images)} pages")

        total_pages = len(images)

        # Create arguments for each page
        page_args = [
            (image, pdf_path, i, total_pages) for i, image in enumerate(images, start=1)
        ]

        # Process all pages in parallel to extract chunks
        all_page_chunks = []

        semaphore = asyncio.Semaphore(self.max_workers)

        # Replace threading.Lock with asyncio.Lock
        stats_lock = asyncio.Lock()

        # Define a function to process a single page with semaphore control
        async def extract_page_chunks_async(args):
            async with semaphore:
                image, pdf_path, page_num, total_pages = args

                if self.verbose:
                    logger.info(f"Processing page {page_num}/{total_pages}")

                # Run the CPU-intensive process in a thread pool
                text_chunks = await asyncio.to_thread(process_image_func, image)

                # Update statistics with async lock
                async with stats_lock:
                    self._stats.processed_pages += 1
                    self._stats.total_chunks += len(text_chunks)
                    self._update_progress(
                        self._stats.processed_pages, self._stats.total_pages
                    )

                # Return the chunks with their page number
                return [(page_num, chunk) for chunk in text_chunks]

        # Create tasks for all pages
        tasks = [extract_page_chunks_async(args) for args in page_args]

        # Gather results, handling exceptions
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for result in results:
            if isinstance(result, Exception):
                error_msg = f"Error in continuous mode processing: {result!s}"
                logger.error(error_msg)
                self._stats.errors.append(error_msg)
                if not self.ignore_errors:
                    raise RuntimeError(error_msg) from result
            else:
                all_page_chunks.extend(result)

        # Sort all chunks by page number to ensure proper ordering
        all_page_chunks.sort(key=lambda x: x[0])

        # Merge chunks that might have continuous content
        documents = []
        current_text = ""
        current_pages = []

        for page_num, chunk in all_page_chunks:
            # Check if the current chunk should be merged with previous content
            is_continuation = current_text and (
                # Hyphenated word continuing
                current_text.rstrip().endswith("-")
                or current_text.rstrip().endswith("\u2014")  # Em dash
                or not current_text.rstrip().endswith(".")
            )  # No period ending

            if is_continuation or not current_text:
                # Continue the current document or start a new one
                if current_text:
                    current_text += f"\n\n{chunk.text}"
                else:
                    current_text = chunk.text

                if page_num not in current_pages:
                    current_pages.append(page_num)
            else:
                # Create a document for the previous content
                if current_text:
                    metadata = {
                        "file_path": pdf_path,
                        "file_name": os.path.basename(pdf_path),
                        "page_range": f"{min(current_pages)}-{max(current_pages)}",
                        "pages": current_pages,
                        "total_pages": total_pages,
                        "source_type": "pdf_continuous",
                    }

                    doc = Document(
                        text=current_text,
                        metadata=metadata,
                        excluded_llm_metadata_keys=[
                            "file_path",
                            "file_name",
                            "page_range",
                            "pages",
                            "total_pages",
                            "source_type",
                        ],
                    )
                    documents.append(doc)

                # Start a new document with the current chunk
                current_text = chunk.text
                current_pages = [page_num]

        # Add the last document if any
        if current_text:
            metadata = {
                "file_path": pdf_path,
                "file_name": os.path.basename(pdf_path),
                "page_range": f"{min(current_pages)}-{max(current_pages)}",
                "pages": current_pages,
                "total_pages": total_pages,
                "source_type": "pdf_continuous",
            }

            doc = Document(
                text=current_text,
                metadata=metadata,
                excluded_llm_metadata_keys=[
                    "file_path",
                    "file_name",
                    "page_range",
                    "pages",
                    "total_pages",
                    "source_type",
                ],
            )
            documents.append(doc)

        return documents
