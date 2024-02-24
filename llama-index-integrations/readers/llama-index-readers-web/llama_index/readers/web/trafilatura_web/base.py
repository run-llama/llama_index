from typing import List

from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document


class TrafilaturaWebReader(BasePydanticReader):
    """Trafilatura web page reader.

    Reads pages from the web.
    Requires the `trafilatura` package.

    """

    is_remote: bool = True

    @classmethod
    def class_name(cls) -> str:
        """Get the name identifier of the class."""
        return "TrafilaturaWebReader"

    def load_data(
        self,
        urls: List[str],
        include_comments=True,
        output_format="txt",
        include_tables=True,
        include_images=False,
        include_formatting=False,
        include_links=False,
    ) -> List[Document]:
        """Load data from the urls.

        Args:
            urls (List[str]): List of URLs to scrape.
            include_comments (bool, optional): Include comments in the output. Defaults to True.
            output_format (str, optional): Output format. Defaults to 'txt'.
            include_tables (bool, optional): Include tables in the output. Defaults to True.
            include_images (bool, optional): Include images in the output. Defaults to False.
            include_formatting (bool, optional): Include formatting in the output. Defaults to False.
            include_links (bool, optional): Include links in the output. Defaults to False.

        Returns:
            List[Document]: List of documents.

        """
        import trafilatura

        if not isinstance(urls, list):
            raise ValueError("urls must be a list of strings.")
        documents = []
        for url in urls:
            downloaded = trafilatura.fetch_url(url)
            response = trafilatura.extract(
                downloaded,
                include_comments=include_comments,
                output_format=output_format,
                include_tables=include_tables,
                include_images=include_images,
                include_formatting=include_formatting,
                include_links=include_links,
            )
            documents.append(Document(text=response, id_=url))

        return documents
