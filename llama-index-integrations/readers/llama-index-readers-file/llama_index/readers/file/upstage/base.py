import io
import os
from pathlib import Path
from typing import Dict, List, Union

import fitz
import requests
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from llama_index.legacy.llms.generic_utils import get_from_param_or_env
from typing_extensions import Literal

LAYOUT_ANALYZER_URL = "https://api.upstage.ai/v1/document-ai/layout-analyzer"
LIMIT_OF_PAGE_REQUEST = 10

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
        raise ValueError("API Key is required for Upstage Document Reader")


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


def parse_output(data: dict, output_type: OutputType) -> str:
    """
    Parse the output data based on the specified output type.

    Args:
        data (dict): The data to be parsed.
        output_type (str): The type of output to parse.

    Returns:
        str: The parsed output.

    Raises:
        ValueError: If the output type is invalid.
    """
    if output_type == "text":
        return data["text"]
    elif output_type == "html":
        return data["html"]
    else:
        raise ValueError(f"Invalid output type: {output_type}")


class UpstageDocumentReader(BaseReader):
    def __init__(
        self,
        api_key: str = None,
        api_base: str = None,
    ):
        """
        Initializes a new instance of the Base class.

        Args:
            api_key (str, optional): The API key to be used for authentication. Defaults to an empty string.
            api_base (str, optional): The base URL for the API. Defaults to LAYOUT_ANALYZER_URL.
        """
        self.api_key = get_from_param_or_env(
            "UPSTAGE_API_KEY", api_key, "UPSTAGE_API_KEY"
        )
        self.api_base = get_from_param_or_env(
            "UPSTAGE_API_BASE", api_base, "UPSTAGE_API_BASE", LAYOUT_ANALYZER_URL
        )

        validate_api_key(self.api_key)

    def _get_response(self, files) -> Dict:
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
            response = requests.post(self.api_base, headers=headers, files=files)
        except requests.exceptions.RequestException as e:
            raise ValueError(f"API call error: {e}")
        finally:
            files["document"].close()

        return response.json()

    def _split_and_request(
        self, full_docs, start_page: int, split_pages: int = LIMIT_OF_PAGE_REQUEST
    ) -> Dict:
        """
        Splits the full document into chunks and sends a request for each chunk.

        Args:
            full_docs: The full document to be split and requested.
            start_page (int): The starting page number for splitting the document.
            split_pages (int, optional): The number of pages to split the document into.
                                         Defaults to LIMIT_OF_PAGE_REQUEST.

        Returns:
            dict: The response from the request.

        """
        with fitz.open() as chunk_pdf:
            chunk_pdf.insert_pdf(
                full_docs,
                from_page=start_page,
                to_page=start_page + split_pages - 1,
            )
            pdf_bytes = chunk_pdf.write()

        files = {"document": io.BytesIO(pdf_bytes)}
        return self._get_response(files)

    def _element_document(
        self, element: Dict, output_type: OutputType, split: SplitType = "element"
    ) -> List[Document]:
        return Document(
            text=(parse_output(element, output_type)),
            extra_info={
                "page": element["page"],
                "id": element["id"],
                "type": output_type,
                "split": split,
            },
        )

    def _page_document(
        self, element: Dict, output_type: OutputType, split: SplitType = "page"
    ) -> List[Document]:
        """
        Generate a list of Document objects based on the provided element, output type, and split type.

        Args:
            element (Dict): The element to process.
            output_type (OutputType): The type of output to generate.
            split (SplitType, optional): The type of split to apply. Defaults to "page".

        Returns:
            List[Document]: A list of Document objects.
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

    def load_data(
        self,
        file_path: Union[Path, str],
        output_type: OutputType = "text",
        split: SplitType = "none",
    ) -> List[Document]:
        """
        Load a document from a file.

        Args:
            file_path (Union[Path, str]): The path to the file.
            output_type (str): The desired output type of the document. Defaults to 'text'.
            split (str): The desired split type of the document. Defaults to 'none'.

        Returns:
            List[Document]: A list of Document objects representing the loaded document.

        Raises:
            TypeError: If file_path is not a string or Path object.
            ValueError: If there is an API call error or an invalid split type.

        """
        # check if file_path is a string or Path
        if not isinstance(file_path, str) and not isinstance(file_path, Path):
            raise TypeError("file_path must be a string or Path.")

        file_name = os.path.basename(file_path)
        validate_file_path(file_path)

        full_docs = fitz.open(file_path)
        number_of_pages = full_docs.page_count

        if split == "none":
            if full_docs.is_pdf:
                result = ""
                start_page = 0
                split_pages = LIMIT_OF_PAGE_REQUEST
                for _ in range(number_of_pages):
                    if start_page >= number_of_pages:
                        break

                    response = self._split_and_request(
                        full_docs, start_page, split_pages
                    )
                    result += parse_output(response, output_type)

                    start_page += split_pages

            else:
                files = {"document": open(file_path, "rb")}
                response = self._get_response(files)
                result = parse_output(response, output_type)

            return [
                Document(
                    text=(parse_output(response, output_type)),
                    extra_info={
                        "total_pages": response["billed_pages"],
                        "type": output_type,
                        "split": split,
                    },
                )
            ]

        elif split == "element":
            docs = []
            if full_docs.is_pdf:
                start_page = 0
                split_pages = LIMIT_OF_PAGE_REQUEST
                for _ in range(number_of_pages):
                    if start_page >= number_of_pages:
                        break

                    response = self._split_and_request(
                        full_docs, start_page, split_pages
                    )
                    for element in response["elements"]:
                        docs.append(self._element_document(element, output_type, split))

                    start_page += split_pages

            else:
                files = {"document": open(file_path, "rb")}
                response = self._get_response(files)

                for element in response["elements"]:
                    docs.append(self._element_document(element, output_type, split))

            return docs

        elif split == "page":
            docs = []
            if full_docs.is_pdf:
                start_page = 0
                split_pages = LIMIT_OF_PAGE_REQUEST
                for _ in range(number_of_pages):
                    if start_page >= number_of_pages:
                        break

                    response = self._split_and_request(
                        full_docs, start_page, split_pages
                    )
                    elements = response["elements"]
                    docs.extend(self._page_document(elements, output_type, split))

                    start_page += split_pages
            else:
                files = {"document": open(file_path, "rb")}
                response = self._get_response(files)
                elements = response["elements"]
                docs.extend(self._page_document(elements, output_type, split))

            return docs

        else:
            # Invalid split type
            raise ValueError(f"Invalid split type: {split}")
