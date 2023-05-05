"""Load Documents from a set of persistent Steamship Files."""
from typing import List, Optional

from llama_index.readers.base import BaseReader
from llama_index.readers.schema.base import Document


class SteamshipFileReader(BaseReader):
    """Reads persistent Steamship Files and converts them to Documents.

    Args:
        api_key: Steamship API key. Defaults to STEAMSHIP_API_KEY value if not provided.

    Note:
        Requires install of `steamship` package and an active Steamship API Key.
        To get a Steamship API Key, visit: https://steamship.com/account/api.
        Once you have an API Key, expose it via an environment variable named
        `STEAMSHIP_API_KEY` or pass it as an init argument (`api_key`).
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        """Initialize the Reader."""
        try:
            import steamship  # noqa: F401

            self.api_key = api_key
        except ImportError:
            raise ImportError(
                "`steamship` must be installed to use the SteamshipFileReader.\n"
                "Please run `pip install --upgrade steamship."
            )

    def load_data(
        self,
        workspace: str,
        query: Optional[str] = None,
        file_handles: Optional[List[str]] = None,
        collapse_blocks: bool = True,
        join_str: str = "\n\n",
    ) -> List[Document]:
        """Load data from persistent Steamship Files into Documents.

        Args:
            workspace: the handle for a Steamship workspace
                (see: https://docs.steamship.com/workspaces/index.html)
            query: a Steamship tag query for retrieving files
                (ex: 'filetag and value("import-id")="import-001"')
            file_handles: a list of Steamship File handles
                (ex: `smooth-valley-9kbdr`)
            collapse_blocks: whether to merge individual File Blocks into a
                single Document, or separate them.
            join_str: when collapse_blocks is True, this is how the block texts
                will be concatenated.

        Note:
            The collection of Files from both `query` and `file_handles` will be
            combined. There is no (current) support for deconflicting the collections
            (meaning that if a file appears both in the result set of the query and
            as a handle in file_handles, it will be loaded twice).
        """
        from steamship import File, Steamship

        client = Steamship(workspace=workspace, api_key=self.api_key)
        files = []
        if query:
            files_from_query = File.query(client=client, tag_filter_query=query).files
            files.extend(files_from_query)

        if file_handles:
            files.extend([File.get(client=client, handle=h) for h in file_handles])

        docs = []
        for file in files:
            extra_info = {"source": file.handle}

            for tag in file.tags:
                extra_info[tag.kind] = tag.value

            if collapse_blocks:
                text = join_str.join([b.text for b in file.blocks])
                docs.append(
                    Document(text=text, doc_id=file.handle, extra_info=extra_info)
                )
            else:
                docs.extend(
                    [
                        Document(text=b.text, doc_id=file.handle, extra_info=extra_info)
                        for b in file.blocks
                    ]
                )

        return docs
