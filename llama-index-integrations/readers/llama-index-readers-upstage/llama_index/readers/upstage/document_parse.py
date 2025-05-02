import io
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Union
from fitz import Document as fitzDocument

import fitz  # type: ignore
from llama_index.core.readers.base import BaseReader
import requests
from llama_index.core.schema import Document


DOCUMENT_PARSE_BASE_URL = "https://api.upstage.ai/v1/document-ai/document-parse"
DEFAULT_NUMBER_OF_PAGE = 10
DOCUMENT_PARSE_DEFAULT_MODEL = "document-parse"

OutputFormat = Literal["text", "html", "markdown"]
OCR = Literal["auto", "force"]
SplitType = Literal["none", "page", "element"]
Category = Literal[
    "paragraph",
    "table",
    "figure",
    "header",
    "footer",
    "caption",
    "equation",
    "heading1",
    "list",
    "index",
    "footnote",
    "chart",
]


def validate_file_path(file_path: str) -> None:
    """
    Validates if a file exists at the given file path.

    Args:
        file_path (str): The path to the file.

    Raises:
        FileNotFoundError: If the file does not exist at the given file path.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")


def parse_output(data: dict, output_format: OutputFormat) -> str:
    """
    Parse the output data based on the specified output type.

    Args:
        data (dict): The data to be parsed.
        output_type (OutputFormat): The output type to parse the element data
                                               into.

    Returns:
        str: The parsed output.

    Raises:
        ValueError: If the output type is invalid.
    """
    content = data["content"]
    if output_format == "text":
        return content["text"]
    elif output_format == "html":
        return content["html"]
    elif output_format == "markdown":
        return content["markdown"]
    else:
        raise ValueError(f"Invalid output type: {output_format}")


def get_from_param_or_env(
    key: str,
    param: Optional[str] = None,
    env_key: Optional[str] = None,
    default: Optional[str] = None,
) -> str:
    """Get a value from a param or an environment variable."""
    if param is not None:
        return param
    elif env_key and env_key in os.environ and os.environ[env_key]:
        return os.environ[env_key]
    elif default is not None:
        return default
    else:
        raise ValueError(
            f"Did not find {key}, please add an environment variable"
            f" `{env_key}` which contains it, or pass"
            f"  `{key}` as a named parameter."
        )


class UpstageDocumentParseReader(BaseReader):
    """
    Upstage Document Parse Reader.

    To use, you should have the environment variable `UPSTAGE_API_KEY`
    set with your API key or pass it as a named parameter to the constructor.

    Example:
        .. code-block:: python

            from llama_index.readers.file import UpstageDocumentParseReader

            reader = UpstageDocumentParseReader()

            docs = reader.load_data("path/to/file.pdf")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DOCUMENT_PARSE_BASE_URL,
        model: str = DOCUMENT_PARSE_DEFAULT_MODEL,
        split: SplitType = "none",
        ocr: OCR = "auto",
        output_format: OutputFormat = "html",
        coordinates: bool = True,
        base64_encoding: List[Category] = [],
    ):
        """
        Initializes an instance of the Upstage Document Parse Reader class.

        Args:
            api_key (str, optional): The API key for accessing the Upstage API.
                                     Defaults to None, in which case it will be
                                     fetched from the environment variable
                                     `UPSTAGE_API_KEY`.
            base_url (str, optional): The base URL for accessing the Upstage API.
            split (SplitType, optional): The type of splitting to be applied.
                                         Defaults to "none" (no splitting).
            model (str): The model to be used for the document parse.
                         Defaults to "document-parse".
            ocr (OCRMode, optional): Extract text from images in the document using OCR.
                                     If the value is "force", OCR is used to extract
                                     text from an image. If the value is "auto", text is
                                     extracted from a PDF. (An error will occur if the
                                     value is "auto" and the input is NOT in PDF format)
            output_format (OutputFormat, optional): Format of the inference results.
            coordinates (bool, optional): Whether to include the coordinates of the
                                          OCR in the output.
            base64_encoding (List[Category], optional): The category of the elements to
                                                        be encoded in base64.

        """
        self.api_key = get_from_param_or_env(
            "UPSTAGE_API_KEY", api_key, "UPSTAGE_API_KEY"
        )
        self.base_url = base_url
        self.model = model
        self.split = split
        self.ocr = ocr
        self.output_format = output_format
        self.coordinates = coordinates
        self.base64_encoding = base64_encoding

    def _get_response(
        self,
        files: Dict,
    ) -> List:
        """
        Sends a POST request to the API endpoint with the provided files and
        returns the response.

        Args:
            files (dict): A dictionary containing the files to be sent in the request.

        Returns:
            dict: The JSON response from the API.

        Raises:
            ValueError: If there is an error in the API call.
        """
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
            }
            response = requests.post(
                self.base_url,
                headers=headers,
                files=files,
                data={
                    "ocr": self.ocr,
                    "model": self.model,
                    "output_formats": f"['{self.output_format}']",
                    "coordinates": self.coordinates,
                    "base64_encoding": f"{self.base64_encoding}",
                },
            )
            response.raise_for_status()
            return response.json().get("elements", [])

        except requests.HTTPError as e:
            raise ValueError(f"HTTP error: {e.response.text}")
        except requests.RequestException as e:
            # Handle any request-related exceptions
            raise ValueError(f"Failed to send request: {e}")
        except json.JSONDecodeError as e:
            # Handle JSON decode errors
            raise ValueError(f"Failed to decode JSON response: {e}")
        except Exception as e:
            # Handle any other exceptions
            raise ValueError(f"An error occurred: {e}")

    def _split_and_request(
        self,
        full_docs: fitzDocument,
        start_page: int,
        num_pages: int,
    ) -> List:
        """
        Splits the full pdf document into partial pages and sends a request to the
        server.

        Args:
            full_docs (str): The full document to be split and requested.
            start_page (int): The starting page number for splitting the document.
            num_pages (int, optional): The number of pages to split the document
                                       into.
                                       Defaults to DEFAULT_NUMBER_OF_PAGE.

        Returns:
            response: The response from the server.
        """
        with fitz.open() as chunk_pdf:
            chunk_pdf.insert_pdf(
                full_docs,
                from_page=start_page,
                to_page=start_page + num_pages - 1,
            )
            pdf_bytes = chunk_pdf.write()

        with io.BytesIO(pdf_bytes) as buffer:
            return self._get_response({"document": buffer})

    def _element_document(self, element: Dict) -> Document:
        """
        Converts an elements into a Document object.

        Args:
            element (Dict): The element to be converted into a Document object.

        Returns:
            Document: A Document object representing the element with its content
                      and metadata.

        """
        extra_info = {
            "page": element["page"],
            "id": element["id"],
            "output_format": self.output_format,
            "split": self.split,
            "category": element.get("category"),
        }
        if element.get("coordinates") is not None:
            extra_info["coordinates"] = element.get("coordinates")
        if element.get("base64_encoding") is not None:
            extra_info["base64_encoding"] = element.get("base64_encoding")

        return Document(
            text=(parse_output(element, self.output_format)), extra_info=extra_info
        )

    def _page_document(self, elements: List) -> List[Document]:
        """
        Combines elements with the same page number into a single Document object.

        Args:
            elements (List): A list of elements containing page numbers.

        Returns:
            List[Document]: A list of Document objects, each representing a page
                            with its content and metadata.
        """
        _docs = []
        pages = sorted({x["page"] for x in elements})

        page_group = [
            [element for element in elements if element["page"] == x] for x in pages
        ]

        for group in page_group:
            page_content = " ".join(
                [parse_output(element, self.output_format) for element in group]
            )

            coordinates = [
                element.get("coordinates")
                for element in group
                if element.get("coordinates") is not None
            ]

            base64_encodings = [
                element.get("base64_encoding")
                for element in group
                if element.get("base64_encoding") is not None
            ]
            extra_info = {
                "page": group[0]["page"],
                "output_format": self.output_format,
                "split": self.split,
            }

            if coordinates:
                extra_info["coordinates"] = coordinates

            if base64_encodings:
                extra_info["base64_encodings"] = base64_encodings

            _docs.append(
                Document(
                    text=page_content.strip(),
                    extra_info=extra_info,
                )
            )

        return _docs

    def lazy_load_data(
        self,
        file_path: Union[str, Path, List[str], List[Path]],
    ) -> Iterable[Document]:
        """
        Load data from a file or list of files lazily.

        Args:
            file_path (Union[str, Path, List[str], List[Path]]): The path or list of paths to the file(s) to load.

        Returns:
            List[Document]: A list of Document objects containing the loaded data.

        Raises:
            ValueError: If an invalid split type is provided or if file_path is required.
        """
        # Check if the file path is a list of paths
        if isinstance(file_path, list):
            for path in file_path:
                docs = self.load_data(path)
                yield from docs

        else:
            num_pages = DEFAULT_NUMBER_OF_PAGE

            if not file_path:
                raise ValueError("file_path is required.")

            validate_file_path(file_path)

            full_docs = fitz.open(file_path)
            number_of_pages = full_docs.page_count

            if self.split == "none":
                if full_docs.is_pdf:
                    result = ""
                    start_page = 0
                    for _ in range(number_of_pages):
                        if start_page >= number_of_pages:
                            break

                        elements = self._split_and_request(
                            full_docs, start_page, num_pages
                        )
                        for element in elements:
                            result += parse_output(element, self.output_format)

                        start_page += num_pages

                else:
                    with open(file_path, "rb") as f:
                        elements = self._get_response({"document": f})

                    result = ""
                    for element in elements:
                        result += parse_output(element, self.output_format)

                yield Document(
                    text=result,
                    extra_info={
                        "total_pages": number_of_pages,
                        "type": self.output_format,
                        "split": self.split,
                    },
                )

            elif self.split == "element":
                if full_docs.is_pdf:
                    start_page = 0
                    for _ in range(number_of_pages):
                        if start_page >= number_of_pages:
                            break

                        elements = self._split_and_request(
                            full_docs, start_page, num_pages
                        )
                        for element in elements:
                            yield self._element_document(element)

                        start_page += num_pages

                else:
                    with open(file_path, "rb") as f:
                        elements = self._get_response({"document": f})

                    for element in elements:
                        yield self._element_document(element)

            elif self.split == "page":
                if full_docs.is_pdf:
                    start_page = 0
                    for _ in range(number_of_pages):
                        if start_page >= number_of_pages:
                            break

                        elements = self._split_and_request(
                            full_docs, start_page, num_pages
                        )
                        yield from self._page_document(elements)

                        start_page += num_pages
                else:
                    with open(file_path, "rb") as f:
                        elements = self._get_response({"document": f})

                    yield from self._page_document(elements)

            else:
                raise ValueError(f"Invalid split type: {self.split}")
