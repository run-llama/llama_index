"""Unstructured file reader.

A parser for unstructured text files using Unstructured.io.
Supports .txt, .docx, .pptx, .jpg, .png, .eml, .html, and .pdf documents.

"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class UnstructuredReader(BaseReader):
    """General unstructured text reader for a variety of files."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Init params."""
        super().__init__(*args)  # not passing kwargs to parent bc it cannot accept it

        self.api = False  # we default to local
        if "url" in kwargs:
            self.server_url = str(kwargs["url"])
            self.api = True  # is url was set, switch to api
        else:
            self.server_url = "http://localhost:8000"

        if "api" in kwargs:
            self.api = kwargs["api"]

        self.api_key = ""
        if "api_key" in kwargs:
            self.api_key = kwargs["api_key"]

        # Prerequisite for Unstructured.io to work
        import nltk

        if not nltk.data.find("tokenizers/punkt"):
            nltk.download("punkt")
        if not nltk.data.find("taggers/averaged_perceptron_tagger"):
            nltk.download("averaged_perceptron_tagger")

    """ Loads data using Unstructured.io py

        Depending on the constructin if url is set or api = True
        it'll parse file using API call, else parse it locally
        extra_info is extended by the returned metadata if
        split_documents is True

        Returns list of documents
    """

    def load_data(
        self,
        file: Path,
        extra_info: Optional[Dict] = None,
        split_documents: Optional[bool] = False,
    ) -> List[Document]:
        """If api is set, parse through api."""
        if self.api:
            from unstructured.partition.api import partition_via_api

            elements = partition_via_api(
                filename=str(file),
                api_key=self.api_key,
                api_url=self.server_url + "/general/v0/general",
            )
        else:
            """Parse file locally"""
            from unstructured.partition.auto import partition

            elements = partition(filename=str(file))

        """ Process elements """
        docs = []
        if split_documents:
            for node in elements:
                metadata = {}
                if hasattr(node, "metadata"):
                    """Load metadata fields"""
                    for field, val in vars(node.metadata).items():
                        if field == "_known_field_names":
                            continue
                        # removing coordinates because it does not serialize
                        # and dont want to bother with it
                        if field == "coordinates":
                            continue
                        # removing bc it might cause interference
                        if field == "parent_id":
                            continue
                        metadata[field] = val

                if extra_info is not None:
                    metadata.update(extra_info)

                metadata["filename"] = str(file)
                docs.append(Document(text=node.text, extra_info=metadata))

        else:
            text_chunks = [" ".join(str(el).split()) for el in elements]

            metadata = {}

            if extra_info is not None:
                metadata.update(extra_info)

            metadata["filename"] = str(file)
            # Create a single document by joining all the texts
            docs.append(Document(text="\n\n".join(text_chunks), extra_info=metadata))

        return docs
