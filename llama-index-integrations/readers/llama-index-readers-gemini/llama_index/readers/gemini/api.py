"""Gemini API integration for PDF OCR and document processing."""

import asyncio
import logging
import time
from typing import List, Union

import PIL.Image

try:
    import google.genai as genai
except ImportError:
    import_err_msg = """ `google-genai` package not found.
    Install google-genai package by running:
    ```pip install google-genai```
    """
    raise ImportError(import_err_msg)

from llama_index.readers.gemini.types import Chunks, TextChunk

# Configure logging
logger = logging.getLogger(__name__)


class GeminiAPI:
    """Wrapper for the Gemini API for document processing."""

    # System prompt template for document extraction and chunking
    SYSTEM_PROMPT_TEMPLATE: str = """
ANALYZE THIS DOCUMENT IMAGE AND EXTRACT CHUNK TEXTS:

STEP 1: OCR - Optical Character Recognition
- Extract all visible text from the document image with high accuracy
- Preserve the original document structure in clean markdown format
- Maintain proper reading order and paragraph flow
- Correctly identify and format headings, subheadings, and sections
- Preserve footnotes, citations, and references when present
- Distinguish between main content, captions, and sidebars
{form_instructions}
- Include text from tables with proper structural formatting
{math_formula_instructions}

STEP 2: Document Chunking
- Identify natural content boundaries in the document
- Create semantic chunks of approximately {chunk_instruction} words each
- Respect the document's logical structure (sections, subsections)
- Keep coherent content together in the same chunk to avoid fragmentation of meaning
- Preserve the relationship between:
  * Questions and answers
  * Headers and their content
  * Lists and their context
  * Citations and their references
  * Images and their captions
- Never split tables, figures, or equations across chunks
- Optimize chunks for semantic relevance in a retrieval system

General Processing Guidelines:
1. Preserve the document's logical flow and reading order
2. Handle various layout types (single-column, multi-column, complex layouts)
3. Distinguish between different textual elements (body text, headers, captions, footnotes)
4. Ensure complete text extraction with no content omissions
5. Maintain semantic coherence within each chunk
6. Preserve the context and meaning of the original document
7. Handle specialized formatting appropriately for the document type
{table_instructions}
{checkbox_instructions}
10. For hierarchical or nested content:
   - Maintain the hierarchical structure
   - Preserve parent-child relationships between elements
   - Keep related components together
{language_instructions}

For each document image, provide:
1. A list of "chunks" with:
   - "text": The extracted text content formatted in clean markdown

Format your response matching the specified schema.
"""

    # Form-specific instructions
    FORM_INSTRUCTIONS: str = """
- For forms and structured documents:
  * Preserve the relationship between labels and their fields
  * Maintain the logical structure of forms and questionnaires
  * Keep related form elements together
  * Preserve the state of selection elements (checkboxes, radio buttons)
  * Maintain the distinction between questions and their possible answers
  * If form has multiple questions, keep them in the same chunk
"""

    # Table-specific instructions
    TABLE_INSTRUCTIONS: str = """
8. For tables and tabular data:
   - Preserve the complete table structure using markdown table format
   - Maintain column headers with their respective data
   - Properly align and format all table cells
   - Handle merged cells, spanning cells, and nested tables appropriately
   - Preserve numerical precision and formatting in table cells
   - Include table captions or titles when present
   - For complex tables, maintain the relationship between data cells
   - Ensure all table content is included in the same chunk
"""

    # Checkbox-specific instructions
    CHECKBOX_INSTRUCTIONS: str = """
9. For documents with selection elements:
   - Represent checkboxes using standard Unicode symbols:
     * ☐ (U+2610) for unchecked boxes
     * ☑ (U+2611) or ☒ (U+2612) for checked boxes
   - Maintain consistent formatting for selection indicators
   - Preserve the relationship between selection elements and their labels
   - Correctly identify selected vs. unselected options
   - Keep related checkbox options together in the same chunk
"""

    # Mathematical formula instructions
    MATH_FORMULA_INSTRUCTIONS: str = """
- For documents with mathematical content:
  * Extract equations and formulas accurately
  * Represent mathematical notation using appropriate markdown syntax
  * Maintain the relationship between formulas and surrounding text
  * Preserve subscripts, superscripts, and special mathematical symbols
"""

    # Language-specific instructions template
    LANGUAGE_INSTRUCTIONS_TEMPLATE: str = """
11. This document appears to be primarily in {language}.
Optimize text extraction for this language, paying attention to:
    - Character sets and special symbols specific to {language}
    - Reading order conventions for {language}
    - Proper handling of language-specific punctuation and formatting
"""

    def __init__(
        self,
        api_key: str,
        model_name: str,
        extract_forms: bool,
        extract_tables: bool,
        extract_math_formulas: bool,
        language: str,
        chunk_size: Union[str, int],
        max_retries: int,
        retry_delay: float,
        verbose: bool,
        prompt: str = None,
    ):
        """
        Initialize the Gemini API wrapper.

        Args:
            api_key: Google Gemini API key
            model_name: Gemini model name
            extract_forms: Whether to extract forms
            extract_tables: Whether to extract tables
            extract_math_formulas: Whether to extract math formulas
            language: Primary language of documents
            chunk_size: Target size for chunks
            max_retries: Max number of retries
            retry_delay: Delay between retries in seconds
            verbose: Whether to log detailed messages
        """
        self.api_key = api_key
        self.model_name = model_name
        self.extract_forms = extract_forms
        self.extract_tables = extract_tables
        self.extract_math_formulas = extract_math_formulas
        self.language = language
        self.chunk_size = chunk_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.verbose = verbose
        self._client = None
        self.prompt = prompt

    @property
    def client(self) -> genai.Client:
        """Get the Gemini API client, initializing if needed."""
        if self._client is None:
            self._client = genai.Client(api_key=self.api_key)
            if self.verbose:
                logger.info(
                    f"Initialized Gemini API client with model: {self.model_name}"
                )
        return self._client

    @property
    def system_prompt(self) -> str:
        """Generate the system prompt based on configuration."""
        if self.prompt:
            # Use the provided system prompt
            return self.prompt
        else:
            # Use the default system prompt template
            # Fill in template sections based on configuration
            form_instructions = self.FORM_INSTRUCTIONS if self.extract_forms else ""
            table_instructions = self.TABLE_INSTRUCTIONS if self.extract_tables else ""
            checkbox_instructions = (
                self.CHECKBOX_INSTRUCTIONS if self.extract_forms else ""
            )
            language_instructions = self.LANGUAGE_INSTRUCTIONS_TEMPLATE.format(
                language=self.language
            )
            math_formula_instructions = (
                self.MATH_FORMULA_INSTRUCTIONS if self.extract_math_formulas else ""
            )
            return self.SYSTEM_PROMPT_TEMPLATE.format(
                form_instructions=form_instructions,
                table_instructions=table_instructions,
                checkbox_instructions=checkbox_instructions,
                language_instructions=language_instructions,
                math_formula_instructions=math_formula_instructions,
                chunk_instruction=self.chunk_size,
            )

    async def process_image_async(
        self, image: PIL.Image.Image, retry_count: int = 0
    ) -> List[TextChunk]:
        """
        Process a single image for OCR and document chunking asynchronously.

        Args:
            image: Image to process
            retry_count: Current retry attempt (for internal use)

        Returns:
            List of text chunks extracted from the image

        Raises:
            RuntimeError: If processing fails after retries
        """
        try:
            if self.verbose and retry_count == 0:
                logger.info(f"Processing image ({image.width}x{image.height})")
            elif self.verbose and retry_count > 0:
                logger.info(
                    f"Retrying image processing (attempt\
                        {retry_count}/{self.max_retries})"
                )

            # Generate OCR and chunking
            response = self.client.models.generate_content(
                contents=[image, self.system_prompt],
                model=self.model_name,
                config={
                    "temperature": 0,
                    "response_mime_type": "application/json",
                    "response_schema": Chunks,
                },
            )

            chunks = response.parsed.chunks
            if self.verbose:
                logger.info(f"Processed document with {len(chunks)} chunks")

            return chunks

        except Exception as e:
            # Handle retries
            if retry_count < self.max_retries:
                if self.verbose:
                    logger.warning(f"Processing error: {e!s}. Retrying...")

                # Wait before retrying
                # (using asyncio.sleep for non-blocking wait)
                await asyncio.sleep(self.retry_delay * (retry_count + 1))

                # Retry with incremented count
                return await self.process_image_async(image, retry_count + 1)
            else:
                logger.error(
                    f"Processing error after {self.max_retries} retries: {e!s}"
                )
                raise RuntimeError(f"Failed to process image: {e!s}") from e

    def process_image(
        self, image: PIL.Image.Image, retry_count: int = 0
    ) -> List[TextChunk]:
        """
        Process a single image for OCR and document chunking.

        Synchronous version for backward compatibility.

        Args:
            image: Image to process
            retry_count: Current retry attempt (for internal use)

        Returns:
            List of text chunks extracted from the image

        Raises:
            RuntimeError: If processing fails after retries
        """
        # For synchronous calls, we have to call the API directly
        try:
            if self.verbose and retry_count == 0:
                logger.info(f"Processing image ({image.width}x{image.height})")
            elif self.verbose and retry_count > 0:
                logger.info(
                    f"Retrying image processing (attempt\
                        {retry_count}/{self.max_retries})"
                )

            # Generate OCR and chunking
            response = self.client.models.generate_content(
                contents=[image, self.system_prompt],
                model=self.model_name,
                config={
                    "temperature": 0,
                    "response_mime_type": "application/json",
                    "response_schema": Chunks,
                },
            )

            chunks = response.parsed.chunks
            if self.verbose:
                logger.info(f"Processed document with {len(chunks)} chunks")

            return chunks

        except Exception as e:
            # Handle retries
            if retry_count < self.max_retries:
                if self.verbose:
                    logger.warning(f"Processing error: {e!s}. Retrying...")

                # Wait before retrying
                time.sleep(self.retry_delay * (retry_count + 1))

                # Retry with incremented count
                return self.process_image(image, retry_count + 1)
            else:
                logger.error(
                    f"Processing error after {self.max_retries} retries: {e!s}"
                )
                raise RuntimeError(f"Failed to process image: {e!s}") from e
