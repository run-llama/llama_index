import os
from enum import Enum
from pathlib import Path
from typing import List, Union

import requests
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

LAYOUT_ANALYZER_URL = "https://api.upstage.ai/v1/document-ai/layout-analyzer"


class OutputType(Enum):
    """
    Represents the output type for a document reader.
    """

    TEXT = "text"
    HTML = "html"


class SplitType(Enum):
    """
    Enum class representing the type of split for a document.

    Attributes:
        NONE (str): Represents no split.
        ELEMENT (str): Represents splitting by element.
        PAGE (str): Represents splitting by page.
    """

    NONE = "none"
    ELEMENT = "element"
    PAGE = "page"


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


def parse_output(data: dict, output_type: str) -> str:
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
    if (output_type) == OutputType.TEXT.value:
        return data["text"]
    elif (output_type) == OutputType.HTML.value:
        return data["html"]
    else:
        raise ValueError(f"Invalid output type: {output_type}")


class UpstageDocumentReader(BaseReader):
    """
    A class for loading documents using the Upstage Document Reader.

    Args:
        api_key (str): The API key for authentication.
        url (str): The URL for the layout analyzer.

    Attributes:
        api_key (str): The API key for authentication.
        url (str): The URL for the layout analyzer.

    """

    def __init__(
        self,
        api_key: str = "",
        url: str = LAYOUT_ANALYZER_URL,
    ):
        self.api_key = api_key
        self.url = url

        validate_api_key(self.api_key)

    def load(
        self,
        file_path: Union[Path, str],
        output_type: str = OutputType.TEXT.value,
        split: str = SplitType.NONE.value,
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

        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            files = {"document": open(file_path, "rb")}
            response = requests.post(self.url, headers=headers, files=files)
        except requests.exceptions.RequestException as e:
            raise ValueError(f"API call error: {e}")
        finally:
            files["document"].close()

        if response.status_code != 200:
            raise ValueError(f"API call error: {response.status_code}")

        json = response.json()

        if (split) == SplitType.NONE.value:
            # Split by document (NONE)
            docs = []
            docs.append(
                Document(
                    text=(parse_output(json, output_type)),
                    extra_info={
                        "total_pages": json["billed_pages"],
                        "type": output_type,
                        "split": split,
                    },
                )
            )
            return docs

        elif (split) == SplitType.ELEMENT.value:
            # Split by element
            docs = []
            for element in json["elements"]:
                docs.append(
                    Document(
                        text=(parse_output(element, output_type)),
                        extra_info={
                            "page": element["page"],
                            "id": element["id"],
                            "type": output_type,
                            "split": split,
                        },
                    )
                )

            return docs

        elif (split) == SplitType.PAGE.value:
            # Split by page
            elements = json["elements"]
            pages = sorted({x["page"] for x in elements})

            page_group = [
                [element for element in elements if element["page"] == x] for x in pages
            ]

            docs = []
            for group in page_group:
                page_content = ""
                for element in group:
                    page_content += parse_output(element, output_type) + " "
                docs.append(
                    Document(
                        text=page_content.strip(),
                        extra_info={
                            "page": group[0]["page"],
                            "type": output_type,
                            "split": split,
                        },
                    )
                )

            return docs

        else:
            # Invalid split type
            raise ValueError(f"Invalid split type: {split}")

        return []
