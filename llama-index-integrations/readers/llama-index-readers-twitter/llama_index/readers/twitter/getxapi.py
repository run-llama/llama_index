"""Reader that searches tweets via the GetXAPI Twitter / X data API."""

from typing import Any, List, Optional

from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document


DEFAULT_BASE_URL = "https://api.getxapi.com"
DEFAULT_SEARCH_PATH = "/twitter/tweet/advanced_search"


class GetXAPISearchReader(BasePydanticReader):
    """
    Twitter / X search reader backed by the GetXAPI HTTP API.

    Useful when access to the official Twitter developer API is not available.
    Authentication is a single Bearer token passed via the ``Authorization``
    header.

    Args:
        bearer_token (str): GetXAPI bearer token.
        base_url (Optional[str]): Override the API base URL.
            Defaults to ``https://api.getxapi.com``.
        timeout (Optional[float]): Per-request timeout in seconds. Defaults to ``30``.

    """

    is_remote: bool = True
    bearer_token: str
    base_url: str = DEFAULT_BASE_URL
    timeout: float = 30.0

    def __init__(
        self,
        bearer_token: str,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        super().__init__(
            bearer_token=bearer_token,
            base_url=base_url or DEFAULT_BASE_URL,
            timeout=timeout if timeout is not None else 30.0,
        )

    @classmethod
    def class_name(cls) -> str:
        return "GetXAPISearchReader"

    def _build_url(self) -> str:
        return f"{self.base_url.rstrip('/')}{DEFAULT_SEARCH_PATH}"

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.bearer_token}",
            "Accept": "application/json",
        }

    @staticmethod
    def _tweet_to_document(tweet: dict) -> Document:
        text = tweet.get("text") or tweet.get("full_text") or ""
        tweet_id = str(tweet.get("id") or tweet.get("id_str") or "")
        author = tweet.get("author") or {}
        username = author.get("username") or author.get("screen_name") or ""
        metadata = {
            "id": tweet_id,
            "url": (
                f"https://x.com/{username}/status/{tweet_id}"
                if tweet_id and username
                else None
            ),
            "author_username": username,
            "author_name": author.get("name"),
            "created_at": tweet.get("created_at"),
            "like_count": tweet.get("like_count") or tweet.get("favorite_count"),
            "retweet_count": tweet.get("retweet_count"),
            "reply_count": tweet.get("reply_count"),
        }
        metadata = {k: v for k, v in metadata.items() if v is not None}
        return Document(text=text, id_=tweet_id or None, metadata=metadata)

    def load_data(
        self,
        query: str,
        limit: Optional[int] = 50,
        **load_kwargs: Any,
    ) -> List[Document]:
        """
        Run an advanced search and return matching tweets as ``Document`` objects.

        Args:
            query (str): Twitter / X advanced search query.
            limit (Optional[int]): Maximum number of tweets to return. Defaults to ``50``.

        """
        try:
            import httpx
        except ImportError:
            raise ImportError(
                "`httpx` package not found, please run `pip install httpx`"
            )

        params: dict = {"q": query}
        if limit is not None:
            params["limit"] = int(limit)
        params.update(load_kwargs)

        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(
                self._build_url(),
                headers=self._headers(),
                params=params,
            )
            response.raise_for_status()
            payload = response.json()

        if isinstance(payload, list):
            tweets = payload
        elif isinstance(payload, dict):
            tweets = (
                payload.get("data")
                or payload.get("tweets")
                or payload.get("results")
                or []
            )
        else:
            tweets = []

        return [self._tweet_to_document(t) for t in tweets if isinstance(t, dict)]
