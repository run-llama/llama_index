import io
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Union

import fitz  # type: ignore
import requests
from fitz import Document as fitzDocument
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

LAYOUT_ANALYSIS_URL = "https://api.upstage.ai/v1/document-ai/layout-analysis"

DEFAULT_NUMBER_OF_PAGE = 10

OutputType = Literal["text", "html"]
SplitType = Literal["none", "element", "page"]


def validate_api_key(api_key: str) -> None:
    """
    Validates the provided API key.

    Args:
        api_key (str): The API key to be validated.

    Raises:
        ValueError: If the API key is empty or None.

    Returns:
        None

    """
    if not api_key:
        raise ValueError("API Key is required for Upstage Document Reader.")


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


def parse_output(data: dict, output_type: Union[OutputType, dict]) -> str:
    """
    Parse the output data based on the specified output type.

    Args:
        data (dict): The data to be parsed.
        output_type (Union[OutputType, dict]): The output type to parse the element data
                                               into.

    Returns:
        str: The parsed output.

    Raises:
        ValueError: If the output type is invalid.

    """
    if isinstance(output_type, dict):
        if data["category"] in output_type:
            return data[output_type[data["category"]]]
        else:
            return data["html"]
    elif isinstance(output_type, str):
        if output_type == "text":
            return data["text"]
        elif output_type == "html":
            return data["html"]
        else:
            raise ValueError(f"Invalid output type: {output_type}")
    else:
        raise ValueError(f"Invalid output type: {output_type}")


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


class UpstageLayoutAnalysisReader(BaseReader):
    """
    Upstage Layout Analysis Reader.

    To use, you should have the environment variable `UPSTAGE_API_KEY`
    set with your API key or pass it as a named parameter to the constructor.

    Example:
        .. code-block:: python

            from llama_index.readers.file import UpstageLayoutAnalysisReader

            reader = UpstageLayoutAnalysisReader()

            docs = reader.load_data("path/to/file.pdf")

    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        use_ocr: bool = False,
        exclude: list = ["header", "footer"],
    ):
        """
        Initializes an instance of the Upstage class.

        Args:
            api_key (str, optional): The API key for accessing the Upstage API.
                                     Defaults to None, in which case it will be
                                     fetched from the environment variable
                                     `UPSTAGE_API_KEY`.
            use_ocr (bool, optional): Extract text from images in the document.
                                      Defaults to False. (Use text info in PDF file)
            exclude (list, optional): Exclude specific elements from the output.
                                      Defaults to [] (all included).

        """
        self.api_key = get_from_param_or_env(
            "UPSTAGE_API_KEY", api_key, "UPSTAGE_API_KEY"
        )
        self.use_ocr = use_ocr
        self.exclude = exclude

        validate_api_key(self.api_key)

    def _get_response(self, files: Dict) -> List:
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
            headers = {"Authorization": f"Bearer {self.api_key}"}
            options = {"ocr": self.use_ocr}
            response = requests.post(
                LAYOUT_ANALYSIS_URL, headers=headers, files=files, data=options
            )
            response.raise_for_status()

            result = response.json().get("elements", [])

            return [
                element for element in result if element["category"] not in self.exclude
            ]

        except requests.RequestException as req_err:
            # Handle any request-related exceptions
            print(f"Request Exception: {req_err}")
            raise ValueError(f"Failed to send request to Upstage API: {req_err}")
        except json.JSONDecodeError as json_err:
            # Handle JSON decode errors
            print(f"JSON Decode Error: {json_err}")
            raise ValueError(f"Failed to decode JSON response: {json_err}")

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

        with io.BytesIO(pdf_bytes) as f:
            return self._get_response({"document": f})

    def _element_document(
        self, element: Dict, output_type: OutputType, split: SplitType
    ) -> Document:
        """
        Converts an elements into a Document object.

        Args:
            element (Dict): The element to be converted into a Document object.
            output_type (OutputType): The output type of the document.
            split (SplitType): The split type of the document.

        Returns:
            Document: A Document object representing the element with its content
                      and metadata.

        """
        return Document(
            text=(parse_output(element, output_type)),
            extra_info={
                "page": element["page"],
                "id": element["id"],
                "type": output_type,
                "split": split,
                "bounding_box": json.dumps(element["bounding_box"]),
            },
        )

    def _page_document(
        self, elements: List, output_type: OutputType, split: SplitType
    ) -> List[Document]:
        """
        Combines elements with the same page number into a single Document object.

        Args:
            elements (List): A list of elements containing page numbers.
            output_type (OutputType): The output type of the document.
            split (SplitType): The split type of the document.

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
                [parse_output(element, output_type) for element in group]
            )

            _docs.append(
                Document(
                    text=page_content.strip(),
                    extra_info={
                        "page": group[0]["page"],
                        "type": output_type,
                        "split": split,
                    },
                )
            )

        return _docs

    def lazy_load_data(
        self,
        file_path: Union[str, Path, List[str], List[Path]],
        output_type: Union[OutputType, dict] = "html",
        split: SplitType = "none",
    ) -> Iterable[Document]:
        """
        Load data from a file or list of files lazily.

        Args:
            file_path (Union[str, Path, List[str], List[Path]]): The path or list of paths to the file(s) to load.
            output_type (Union[OutputType, dict], optional): The desired output type. Defaults to "html".
                - If a dict is provided, it should be in the format {"category": "output_type", ...}.
                - The category could possibly include the following:
                    - "paragraph"
                    - "caption"
                    - "table"
                    - "figure"
                    - "equation"
                    - "footer"
                    - "header"
                - The output_type can be "text" or "html".
            split (SplitType, optional): The type of splitting to apply. Defaults to "none".

        Returns:
            List[Document]: A list of Document objects containing the loaded data.

        Raises:
            ValueError: If an invalid split type is provided or if file_path is required.

        """
        # Check if the file path is a list of paths
        if isinstance(file_path, list):
            for path in file_path:
                docs = self.load_data(path, output_type, split)
                yield from docs

        else:
            num_pages = DEFAULT_NUMBER_OF_PAGE

            if not file_path:
                raise ValueError("file_path is required.")

            validate_file_path(file_path)

            full_docs = fitz.open(file_path)
            number_of_pages = full_docs.page_count

            if split == "none":
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
                            result += parse_output(element, output_type)

                        start_page += num_pages

                else:
                    with open(file_path, "rb") as f:
                        elements = self._get_response({"document": f})

                    result = ""
                    for element in elements:
                        result += parse_output(element, output_type)

                yield Document(
                    text=result,
                    extra_info={
                        "total_pages": number_of_pages,
                        "type": output_type,
                        "split": split,
                    },
                )

            elif split == "element":
                if full_docs.is_pdf:
                    start_page = 0
                    for _ in range(number_of_pages):
                        if start_page >= number_of_pages:
                            break

                        elements = self._split_and_request(
                            full_docs, start_page, num_pages
                        )
                        for element in elements:
                            yield self._element_document(element, output_type, split)

                        start_page += num_pages

                else:
                    with open(file_path, "rb") as f:
                        elements = self._get_response({"document": f})

                    for element in elements:
                        yield self._element_document(element, output_type, split)

            elif split == "page":
                if full_docs.is_pdf:
                    start_page = 0
                    for _ in range(number_of_pages):
                        if start_page >= number_of_pages:
                            break

                        elements = self._split_and_request(
                            full_docs, start_page, num_pages
                        )
                        yield from self._page_document(elements, output_type, split)

                        start_page += num_pages
                else:
                    with open(file_path, "rb") as f:
                        elements = self._get_response({"document": f})

                    yield from self._page_document(elements, output_type, split)

            else:
                raise ValueError(f"Invalid split type: {split}")
