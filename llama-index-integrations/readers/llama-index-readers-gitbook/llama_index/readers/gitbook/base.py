
from typing import Dict, Iterable, List, Optional, Union

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

from llama_index.readers.gitbook.gitbook_client import GitbookClient

class SimpleGitbookReader(BaseReader):
    """Simple gitbook reader.

    Convert each gitbook page into Document used by LlamaIndex.

    Args:
        api_token (str): Gitbook API Token.
        api_url (str): Gitbook API Endpoint.
    """

    def __init__(
        self, api_token:str, api_url:Optional[str] = None
    ) -> None:
        """Initialize with parameters."""
        self.client = GitbookClient(api_token, api_url)

    def load_data(
        self,
        space_id: str,
        metadata_names: Optional[List[str]] = None,
        show_progress=False,
    ) -> List[Document]:
        """Load data from the input directory.

        Args:
            space_id (str): Gitbook space id
            metadata_names (Optional[List[str]]): names of the fields to be added
                to the metadata attribute of the Document. 
                only 'path', 'title', 'description', 'parent' are available
                Defaults to None
            show_progress (bool, optional): Show progress bar. Defaults to False

        Returns:
            List[Document]: A list of documents.

        """
        documents = []
        pages = self.client.list_pages(space_id)

        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(pages, desc="Downloading pages")
        else:
            iterator = pages

        for page in iterator:
            id = page.get("id")
            content = self.client.get_page_markdown(space_id, id)
            if metadata_names is None:
                documents.append(Document(text=content, id_=id, metadata={'path': page.get("path")}))
            else:
                try:
                    metadata = {name: page.get(name) for name in metadata_names}
                except KeyError as err:
                    raise ValueError(
                        f"{err.args[0]} field not available."
                    ) from err
                documents.append(Document(text=content, id_=id, metadata=metadata))
        
        return documents