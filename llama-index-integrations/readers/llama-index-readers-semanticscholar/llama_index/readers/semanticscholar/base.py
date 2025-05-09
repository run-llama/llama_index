import logging
import os
from typing import List

import requests
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class SemanticScholarReader(BaseReader):
    """
    A class to read and process data from Semantic Scholar API
    ...

    Methods
    -------
    __init__():
       Instantiate the SemanticScholar object

    load_data(query: str, limit: int = 10, returned_fields: list = ["title", "abstract", "venue", "year", "paperId", "citationCount", "openAccessPdf", "authors"]) -> list:
        Loads data from Semantic Scholar based on the query and returned_fields

    """

    def __init__(self, timeout=10, api_key=None, base_dir="pdfs") -> None:
        """
        Instantiate the SemanticScholar object.
        """
        import arxiv

        from semanticscholar import SemanticScholar

        self.arxiv = arxiv
        self.base_dir = base_dir
        self.s2 = SemanticScholar(timeout, api_key)
        # check for base dir
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

    def _clear_cache(self):
        """
        Delete the .citation* folder.
        """
        import shutil

        shutil.rmtree("./.citation*")

    def _download_pdf(self, paper_id, url: str, base_dir="pdfs"):
        logger = logging.getLogger()
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML,"
                " like Gecko) Chrome/58.0.3029.110 Safari/537.3"
            )
        }
        # Making a GET request
        response = requests.get(url, headers=headers, stream=True)
        content_type = response.headers["Content-Type"]

        # As long as the content-type is application/pdf, this will download the file
        if "application/pdf" in content_type:
            os.makedirs(base_dir, exist_ok=True)
            file_path = os.path.join(base_dir, f"{paper_id}.pdf")
            # check if the file already exists
            if os.path.exists(file_path):
                logger.info(f"{file_path} already exists")
                return file_path
            with open(file_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        file.write(chunk)
            logger.info(f"Downloaded pdf from {url}")
            return file_path
        else:
            logger.warning(f"{url} was not downloaded: protected")
            return None

    def _get_full_text_docs(self, documents: List[Document]) -> List[Document]:
        from PyPDF2 import PdfReader

        """
        Gets the full text of the documents from Semantic Scholar

        Parameters
        ----------
        documents: list
            The list of Document object that contains the search results

        Returns
        -------
        list
            The list of Document object that contains the search results with full text

        Raises
        ------
        Exception
            If there is an error while getting the full text

        """
        full_text_docs = []
        for paper in documents:
            metadata = paper.extra_info
            url = metadata["openAccessPdf"]
            externalIds = metadata["externalIds"]
            paper_id = metadata["paperId"]
            file_path = None
            persist_dir = os.path.join(self.base_dir, f"{paper_id}.pdf")
            if url and not os.path.exists(persist_dir):
                # Download the document first
                file_path = self._download_pdf(metadata["paperId"], url, persist_dir)

            if (
                not url
                and externalIds
                and "ArXiv" in externalIds
                and not os.path.exists(persist_dir)
            ):
                # download the pdf from arxiv
                file_path = self._download_pdf_from_arxiv(
                    paper_id, externalIds["ArXiv"]
                )

            # Then, check if it's a valid PDF. If it's not, skip to the next document.
            if file_path:
                try:
                    pdf = PdfReader(open(file_path, "rb"))
                except Exception as e:
                    logging.error(
                        f"Failed to read pdf with exception: {e}. Skipping document..."
                    )
                    continue

                text = ""
                for page in pdf.pages:
                    text += page.extract_text()
                full_text_docs.append(Document(text=text, extra_info=metadata))

        return full_text_docs

    def _download_pdf_from_arxiv(self, paper_id, arxiv_id):
        paper = next(self.arxiv.Search(id_list=[arxiv_id], max_results=1).results())
        paper.download_pdf(dirpath=self.base_dir, filename=paper_id + ".pdf")
        return os.path.join(self.base_dir, f"{paper_id}.pdf")

    def load_data(
        self,
        query,
        limit,
        full_text=False,
        returned_fields=[
            "title",
            "abstract",
            "venue",
            "year",
            "paperId",
            "citationCount",
            "openAccessPdf",
            "authors",
            "externalIds",
        ],
    ) -> List[Document]:
        """
        Loads data from Semantic Scholar based on the entered query and returned_fields.

        Parameters
        ----------
        query: str
            The search query for the paper
        limit: int, optional
            The number of maximum results returned (default is 10)
        returned_fields: list, optional
            The list of fields to be returned from the search

        Returns
        -------
        list
            The list of Document object that contains the search results

        Raises
        ------
        Exception
            If there is an error while performing the search

        """
        try:
            results = self.s2.search_paper(query, limit=limit, fields=returned_fields)
        except (requests.HTTPError, requests.ConnectionError, requests.Timeout) as e:
            logging.error(
                "Failed to fetch data from Semantic Scholar with exception: %s", e
            )
            raise
        except Exception as e:
            logging.error("An unexpected error occurred: %s", e)
            raise

        documents = []

        for item in results[:limit]:
            openAccessPdf = getattr(item, "openAccessPdf", None)
            abstract = getattr(item, "abstract", None)
            title = getattr(item, "title", None)
            text = None
            # concat title and abstract
            if abstract and title:
                text = title + " " + abstract
            elif not abstract:
                text = title

            metadata = {
                "title": title,
                "venue": getattr(item, "venue", None),
                "year": getattr(item, "year", None),
                "paperId": getattr(item, "paperId", None),
                "citationCount": getattr(item, "citationCount", None),
                "openAccessPdf": openAccessPdf.get("url") if openAccessPdf else None,
                "authors": [author["name"] for author in getattr(item, "authors", [])],
                "externalIds": getattr(item, "externalIds", None),
            }
            documents.append(Document(text=text, extra_info=metadata))

        if full_text:
            full_text_documents = self._get_full_text_docs(documents)
            documents.extend(full_text_documents)
        return documents
