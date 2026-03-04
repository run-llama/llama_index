"""Read Arxiv Papers."""

import hashlib
import logging
import os
from typing import List, Optional, Tuple

from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class ArxivReader(BaseReader):
    """
    Arxiv Reader.

    Gets a search query, return a list of Documents of the top corresponding scientific papers on Arxiv.
    """

    def __init__(
        self,
    ) -> None:
        """Initialize with parameters."""
        super().__init__()

    def _hacky_hash(self, some_string):
        return hashlib.md5(some_string.encode("utf-8")).hexdigest()

    def load_data(
        self,
        search_query: str,
        papers_dir: Optional[str] = ".papers",
        max_results: Optional[int] = 10,
    ) -> List[Document]:
        """
        Search for a topic on Arxiv, download the PDFs of the top results locally, then read them.

        Args:
            search_query (str): A topic to search for (e.g. "Artificial Intelligence").
            papers_dir (Optional[str]): Locally directory to store the papers
            max_results (Optional[int]): Maximum number of papers to fetch.

        Returns:
            List[Document]: A list of Document objects.

        """
        import arxiv

        arxiv_search = arxiv.Search(
            query=search_query,
            id_list=[],
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        search_results = list(arxiv_search.results())
        logging.debug(f"> Successfully fetched {len(search_results)} paperes")

        if not os.path.exists(papers_dir):
            os.makedirs(papers_dir)

        paper_lookup = {}
        for paper in search_results:
            # Hash filename to avoid bad characters in file path
            hashed_name = self._hacky_hash(f"{paper.title}{paper.entry_id}")
            filename = f"{hashed_name}.pdf"
            paper_lookup[filename] = {
                "Title of this paper": paper.title,
                "Authors": (", ").join([a.name for a in paper.authors]),
                "Date published": paper.published.strftime("%m/%d/%Y"),
                "URL": paper.entry_id,
                # "summary": paper.summary
            }
            paper.download_pdf(dirpath=papers_dir, filename=filename)
            logging.debug(f"> Downloading {filename}...")

        def get_paper_metadata(filename):
            return paper_lookup[os.path.basename(filename)]

        arxiv_documents = SimpleDirectoryReader(
            papers_dir,
            file_metadata=get_paper_metadata,
            exclude_hidden=False,  # default directory is hidden ".papers"
        ).load_data()
        # Include extra documents containing the abstracts
        abstract_documents = []
        for paper in search_results:
            d = (
                f"The following is a summary of the paper: {paper.title}\n\nSummary:"
                f" {paper.summary}"
            )
            abstract_documents.append(Document(text=d))

        # Delete downloaded papers
        try:
            for f in os.listdir(papers_dir):
                os.remove(os.path.join(papers_dir, f))
                logging.debug(f"> Deleted file: {f}")
            os.rmdir(papers_dir)
            logging.debug(f"> Deleted directory: {papers_dir}")
        except OSError:
            print("Unable to delete files or directory")

        return arxiv_documents + abstract_documents

    def load_papers_and_abstracts(
        self,
        search_query: str,
        papers_dir: Optional[str] = ".papers",
        max_results: Optional[int] = 10,
    ) -> Tuple[List[Document], List[Document]]:
        """
        Search for a topic on Arxiv, download the PDFs of the top results locally, then read them.

        Args:
            search_query (str): A topic to search for (e.g. "Artificial Intelligence").
            papers_dir (Optional[str]): Locally directory to store the papers
            max_results (Optional[int]): Maximum number of papers to fetch.

        Returns:
            List[Document]: A list of Document objects representing the papers themselves
            List[Document]: A list of Document objects representing abstracts only

        """
        import arxiv

        arxiv_search = arxiv.Search(
            query=search_query,
            id_list=[],
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        search_results = list(arxiv_search.results())
        logging.debug(f"> Successfully fetched {len(search_results)} paperes")

        if not os.path.exists(papers_dir):
            os.makedirs(papers_dir)

        paper_lookup = {}
        for paper in search_results:
            # Hash filename to avoid bad characters in file path
            hashed_name = self._hacky_hash(f"{paper.title}{paper.entry_id}")
            filename = f"{hashed_name}.pdf"
            paper_lookup[filename] = {
                "Title of this paper": paper.title,
                "Authors": (", ").join([a.name for a in paper.authors]),
                "Date published": paper.published.strftime("%m/%d/%Y"),
                "URL": paper.entry_id,
                # "summary": paper.summary
            }
            paper.download_pdf(dirpath=papers_dir, filename=filename)
            logging.debug(f"> Downloading {filename}...")

        def get_paper_metadata(filename):
            return paper_lookup[os.path.basename(filename)]

        arxiv_documents = SimpleDirectoryReader(
            papers_dir,
            file_metadata=get_paper_metadata,
            exclude_hidden=False,  # default directory is hidden ".papers"
        ).load_data()
        # Include extra documents containing the abstracts
        abstract_documents = []
        for paper in search_results:
            d = (
                f"The following is a summary of the paper: {paper.title}\n\nSummary:"
                f" {paper.summary}"
            )
            abstract_documents.append(Document(text=d))

        # Delete downloaded papers
        try:
            for f in os.listdir(papers_dir):
                os.remove(os.path.join(papers_dir, f))
                logging.debug(f"> Deleted file: {f}")
            os.rmdir(papers_dir)
            logging.debug(f"> Deleted directory: {papers_dir}")
        except OSError:
            print("Unable to delete files or directory")

        return arxiv_documents, abstract_documents
