from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from llama_index.core.tools.tool_spec.base import BaseToolSpec
from desearch_py import Desearch


class TwitterScraperMedia(BaseModel):
    media_url: str = ""
    type: str = ""


class TwitterScraperUser(BaseModel):
    # Available in both, scraped and api based tweets.
    id: Optional[str] = Field(example="123456789")
    url: Optional[str] = Field(example="https://x.com/example_user")
    name: Optional[str] = Field(example="John Doe")
    username: Optional[str] = Field(example="johndoe")
    created_at: Optional[str] = Field(example="2023-01-01T00:00:00Z")

    # Only available in scraped tweets
    description: Optional[str] = Field(example="This is an example user description.")
    favourites_count: Optional[int] = Field(example=100)
    followers_count: Optional[int] = Field(example=1500)
    listed_count: Optional[int] = Field(example=10)
    media_count: Optional[int] = Field(example=50)
    profile_image_url: Optional[str] = Field(example="https://example.com/profile.jpg")
    statuses_count: Optional[int] = Field(example=500)
    verified: Optional[bool] = Field(example=True)


class BasicTwitterSearchResponse(BaseModel):
    # Available in both, scraped and api based tweets.
    user: Optional[TwitterScraperUser]
    id: Optional[str] = Field(example="987654321")
    text: Optional[str] = Field(example="This is an example tweet.")
    reply_count: Optional[int] = Field(example=10)
    retweet_count: Optional[int] = Field(example=5)
    like_count: Optional[int] = Field(example=100)
    view_count: Optional[int] = Field(example=1000)
    quote_count: Optional[int] = Field(example=2)
    impression_count: Optional[int] = Field(example=1500)
    bookmark_count: Optional[int] = Field(example=3)
    url: Optional[str] = Field(example="https://x.com/example_tweet")
    created_at: Optional[str] = Field(example="2023-01-01T00:00:00Z")
    media: Optional[List[TwitterScraperMedia]] = Field(default_factory=list, example=[])

    # Only available in scraped tweets
    is_quote_tweet: Optional[bool] = Field(example=False)
    is_retweet: Optional[bool] = Field(example=False)


class WebSearchResult(BaseModel):
    title: str = Field(
        ..., description="EXCLUSIVE Major coffee buyers face losses as Colombia ..."
    )
    snippet: str = Field(
        ...,
        description="Coffee farmers in Colombia, the world's No. 2 arabica producer, have failed to deliver up to 1 million bags of beans this year or nearly 10% ...",
    )
    link: str = Field(
        ...,
        description="https://www.reuters.com/world/americas/exclusive-major-coffee-buyers-face-losses-colombia-farmers-fail-deliver-2021-10-11/",
    )
    date: Optional[str] = Field(
        None, description="21 hours ago"
    )  # Optional, as it might not always be present
    source: str = Field(..., description="Reuters")

    author: Optional[str] = Field(None, description="Reuters")

    image: Optional[str] = Field(
        None,
        description="https://static.reuters.com/resources/2021/10/11/Reuters/Reuters_20211011_0000_01.jpg?w=800&h=533&q=80&crop=1",
    )
    favicon: Optional[str] = Field(
        None,
        description="https://static.reuters.com/resources/2021/10/11/Reuters/Reuters_20211011_0000_01.jpg?w=800&h=533&q=80&crop=1",
    )
    highlights: Optional[List[str]] = Field(
        None, description="List of highlights as strings."
    )


class DesearchToolSpec(BaseToolSpec):
    """Desearch tool spec."""

    spec_functions = [
        "ai_search_tool",
        "twitter_search_tool",
        "web_search_tool",
    ]

    def __init__(self, api_key: str) -> None:
        """Initialize with parameters."""
        self.client = Desearch(api_key=api_key)

    def ai_search_tool(
        self,
        prompt: str = Field(description="The search prompt or query."),
        tool: List[
            Literal[
                "web",
                "hackernews",
                "reddit",
                "wikipedia",
                "youtube",
                "twitter",
                "arxiv",
            ]
        ] = Field(description="List of tools to use. Must include at least one tool."),
        model: str = Field(
            default="NOVA",
            description="The model to use for the search. Value should 'NOVA', 'ORBIT' or 'HORIZON'",
        ),
        date_filter: Optional[str] = Field(
            default=None, description="Date filter for the search."
        ),
    ) -> str | dict:
        """
        Perform a search using Desearch.

        Args:
            prompt (str): The search prompt or query.
            tool (List[Literal["web", "hackernews", "reddit", "wikipedia", "youtube", "twitter", "arxiv"]]): List of tools to use. Must include at least one tool.
            model (str, optional): The model to use for the search. Defaults to "NOVA".
            date_filter (Optional[str], optional): Date filter for the search. Defaults to None.

        Returns:
            str | dict: The search result or an error string.

        """
        try:
            return self.client.search(
                prompt,
                tool,
                model,
                date_filter,
            )
        except Exception as e:
            return str(e)

    def twitter_search_tool(
        self,
        query: str = Field(description="The Twitter search query."),
        sort: str = Field(default="Top", description="Sort order for the results."),
        count: int = Field(default=10, description="Number of results to return."),
    ) -> BasicTwitterSearchResponse:
        """
        Perform a basic Twitter search using the Exa API.

        Args:
            query (str, optional): The Twitter search query. Defaults to None.
            sort (str, optional): Sort order for the results. Defaults to "Top".
            count (int, optional): Number of results to return. Defaults to 10.

        Returns:
            BasicTwitterSearchResponse: The search results.

        Raises:
            Exception: If an error occurs when calling the API.

        """
        try:
            return self.client.basic_twitter_search(query, sort, count)
        except Exception as e:
            return str(e)

    def web_search_tool(
        self,
        query: str = Field(description="The search query."),
        num: int = Field(default=10, description="Number of results to return."),
        start: int = Field(
            default=1, description="The starting index for the search results."
        ),
    ) -> List[WebSearchResult]:
        """
        Perform a basic web search using the Exa API.

        Args:
            query (str, optional): The search query. Defaults to None.
            num (int, optional): Number of results to return. Defaults to 10.
            start (int, optional): The starting index for the search results. Defaults to 1.

        Returns:
            List[WebSearchResult]: The search results.

        Raises:
            Exception: If an error occurs when calling the API.

        """
        try:
            return self.client.basic_web_search(query, num, start)
        except Exception as e:
            return str(e)
