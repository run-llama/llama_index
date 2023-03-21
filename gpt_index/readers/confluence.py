"""Confluence reader."""
import os
from typing import List, Optional

from gpt_index.readers.base import BaseReader
from gpt_index.readers.schema.base import Document

CONFLUENCE_USERNAME = "CONFLUENCE_USERNAME"
CONFLUENCE_API_TOKEN = "CONFLUENCE_API_TOKEN"


class ConfluenceReader(BaseReader):
    """Confluence reader.

    Reads a set of confluence pages given a space id and optionally a list of page ids

    Args:
        user_name (str): your confluence username.
        api_token (str): api token for your account (https://support.atlassian.com/atlassian-account/docs/manage-api-tokens-for-your-atlassian-account/)
        base_url (str): 'base_url' for confluence instance, this is suffixed with '/wiki', eg 'https://yoursite.atlassian.com/wiki'

    """

    def __init__(self, user_name: Optional[str] = None, api_token: Optional[str] = None, base_url: str = None) -> None:

        try:
            from atlassian import Confluence
        except ImportError:
            raise ImportError("`atlassian` package not found, please run `pip install atlassian-python-api`")

        if user_name is None:
            user_name = os.getenv(CONFLUENCE_USERNAME)
            if user_name is None:
                raise ValueError(
                    "Must specify `user_name` or set environment "
                    "variable `CONFLUENCE_USERNAME`."
                )
        if api_token is None:
            api_token = os.getenv(CONFLUENCE_API_TOKEN)
            if api_token is None:
                raise ValueError(
                    "Must specify `api_token` or set environment "
                    "variable `CONFLUENCE_API_TOKEN`."
                )

        self.confluence = Confluence(url=base_url, username=user_name, password=api_token, cloud=True)

    def load_data(self, space_id: Optional[str] = None, page_ids: Optional[List[str]] = []) -> List[Document]:
        """Load data from the confluence instance.

        Args:
            space_id (Optional[str]): confluence space id - all pages from this space will be loaded
            page_ids (Optional[List[str]]): list of page ids from the given confluence site to load

            if both are specified, the union of both sets will be returned.

        Returns:
            List[Document]: List of documents.

        """

        try:
            import html2text  # type: ignore
        except ImportError:
            raise ImportError("`html2text` package not found, please run `pip install html2text`")

        if not space_id and len(page_ids) == 0:
            raise ValueError("Must specify either `space_id` or `page_ids` or both.")

        docs = []

        text_maker = html2text.HTML2Text()
        text_maker.ignore_links = True
        text_maker.ignore_images = True

        if space_id:
            pages = self.confluence.get_all_pages_from_space(space=space_id, expand='body.storage.value')

            for page in pages:
                docs.append(Document(text=text_maker.handle(page['body']['storage']['value']), doc_id=page['id'],
                                     extra_info={"title": page['title']}))
        if len(page_ids) != 0:
            for page_id in page_ids:
                page = self.confluence.get_page_by_id(page_id=page_id, expand='body.storage.value')
                docs.append(Document(text=text_maker.handle(page['body']['storage']['value']), doc_id=page['id'],
                                     extra_info={"title": page['title']}))

        return docs


if __name__ == "__main__":
    reader = ConfluenceReader()
