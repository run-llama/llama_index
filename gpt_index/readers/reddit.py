"""Reader wrapping Reddit PMAW Pushshift API."""
from typing import Any, Callable, List, Optional

from gpt_index.readers.base import BaseReader
from gpt_index.readers.schema.base import Document


class RedditReader(BaseReader):
    """Search reddit comments or submissions using PMAW Pushshift API.

    See https://pypi.org/project/pmaw for details on how rate limiting and searching is handled

    Args:
        verbose (bool): Whether to print verbose output. Defaults to False.
    """

    def __init__(
        self,
        verbose: bool = False,
    ) -> None:
        """Initialize with parameters."""
        super().__init__(verbose=verbose)

    def load_data(
        self,
        query: str,
        mode: Optional[str] = "submissions",
        limit: Optional[int] = 10,
        subreddit: Optional[str] = "all",
        filter_fn: Optional[Callable] = None,
    ) -> List[Document]:
        """Perform query and return results.

        Args:
            query (str): Query to search for.
            mode (Optional[str]): Whether to search submissions or comments. Defaults to "submissions".
            limit (Optional[int]): Number of results to return. Defaults to 10.
            subreddit (Optional[str]): Subreddit to search. Defaults to 'all'.
            filter_fn (Optional[Callable]): Filter function to apply to results. Defaults to None.

        """
        try:
            from pmaw import PushshiftAPI
        except ImportError:
            raise ValueError("`pmaw` package not found, please run `pip install pmaw`")

        api = PushshiftAPI()
        if mode == "submissions":
            search_fn = api.search_submissions
            txtkey = "title"
        elif mode == "comments":
            search_fn = api.search_comments
            txtkey = "body"
        else:
            raise ValueError(
                f"Invalid mode: {mode}, valid options are 'submissions' or 'comments'"
            )

        if filter_fn is not None:
            api_results = search_fn(
                q=query, subreddit=subreddit, limit=limit, filter=filter_fn
            )
        else:
            api_results = search_fn(q=query, subreddit=subreddit, limit=limit)

        result = [
            Document(text=c[txtkey], doc_id=c["id"], extra_info=c) for c in api_results
        ]

        return result
