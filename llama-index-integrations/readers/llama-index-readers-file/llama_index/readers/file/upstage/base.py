import os
from pathlib import Path
from typing import List, Union

import requests
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from llama_index.legacy.llms.generic_utils import get_from_param_or_env
from typing_extensions import Literal

LAYOUT_ANALYZER_URL = "https://api.upstage.ai/v1/document-ai/layout-analyzer"

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

    def _get_response(self, file_path) -> requests.Response:
        """
        Sends a POST request to the specified URL with the document file
        and returns the response.

        Returns:
            requests.Response: The response object from the API call.

        Raises:
            ValueError: If there is an error in the API call.
        """
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            files = {"document": open(file_path, "rb")}
            response = requests.post(self.api_base, headers=headers, files=files)
        except requests.exceptions.RequestException as e:
            raise ValueError(f"API call error: {e}")
        finally:
            files["document"].close()

        return response.json()

    def load(
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

        response = self._get_response(file_path)

        if split == "none":
            # Split by document (NONE)
            docs = []
            docs.append(
                Document(
                    text=(parse_output(response, output_type)),
                    extra_info={
                        "total_pages": response["billed_pages"],
                        "type": output_type,
                        "split": split,
                    },
                )
            )
            return docs

        elif split == "element":
            # Split by element
            docs = []
            for element in response["elements"]:
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

        elif split == "page":
            # Split by page
            elements = response["elements"]
            pages = sorted({x["page"] for x in elements})

            page_group = [
                [element for element in elements if element["page"] == x] for x in pages
            ]

            docs = []
            for group in page_group:
                page_content = " ".join([parse_output(x, output_type) for x in group])
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
