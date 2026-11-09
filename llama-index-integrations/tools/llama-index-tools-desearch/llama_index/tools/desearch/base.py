"""Desearch tool spec."""

from typing import Annotated, Any, List, Literal, Optional, Union

from desearch_py import Desearch
from llama_index.core.async_utils import asyncio_run
from llama_index.core.tools.tool_spec.base import BaseToolSpec
from pydantic import BaseModel, Field

SearchSource = Literal[
    "web",
    "hackernews",
    "reddit",
    "wikipedia",
    "youtube",
    "twitter",
    "arxiv",
]

DateFilter = Literal[
    "PAST_24_HOURS",
    "PAST_2_DAYS",
    "PAST_WEEK",
    "PAST_2_WEEKS",
    "PAST_MONTH",
    "PAST_2_MONTHS",
    "PAST_YEAR",
    "PAST_2_YEARS",
]


def _to_jsonable(result: Any) -> Any:
    """Convert desearch-py response models into plain dicts and lists."""
    if isinstance(result, BaseModel):
        return result.model_dump(exclude_none=True)
    if isinstance(result, list):
        return [_to_jsonable(item) for item in result]
    if isinstance(result, dict):
        return {key: _to_jsonable(value) for key, value in result.items()}
    return result


class DesearchToolSpec(BaseToolSpec):
    """Desearch tool spec."""

    spec_functions = [
        ("ai_search_tool", "aai_search_tool"),
        ("twitter_search_tool", "atwitter_search_tool"),
        ("web_search_tool", "aweb_search_tool"),
    ]

    def __init__(self, api_key: str) -> None:
        """Initialize with a Desearch API key."""
        self.api_key = api_key

    async def aai_search_tool(
        self,
        prompt: Annotated[str, Field(description="The search prompt or query.")],
        tool: Annotated[
            List[SearchSource],
            Field(description="Sources to search. Must include at least one source."),
        ],
        date_filter: Annotated[
            Optional[DateFilter],
            Field(description="Only return results from this time window."),
        ] = "PAST_24_HOURS",
        count: Annotated[
            Optional[int],
            Field(description="Number of results per source (10 to 200)."),
        ] = None,
    ) -> Union[dict, str]:
        """
        Search several sources at once with Desearch AI search and get links plus a summary.

        Args:
            prompt (str): The search prompt or query.
            tool (List[str]): Sources to search: web, hackernews, reddit, wikipedia, youtube, twitter, arxiv.
            date_filter (Optional[str]): Time window, for example PAST_24_HOURS or PAST_WEEK. Defaults to PAST_24_HOURS.
            count (Optional[int]): Number of results per source (10 to 200).

        Returns:
            dict | str: The search result, or an error string.

        """
        try:
            async with Desearch(api_key=self.api_key) as client:
                result = await client.ai_search(
                    prompt=prompt,
                    tools=list(tool),
                    date_filter=date_filter,
                    count=count,
                )
            return _to_jsonable(result)
        except Exception as e:
            return str(e)

    def ai_search_tool(
        self,
        prompt: Annotated[str, Field(description="The search prompt or query.")],
        tool: Annotated[
            List[SearchSource],
            Field(description="Sources to search. Must include at least one source."),
        ],
        date_filter: Annotated[
            Optional[DateFilter],
            Field(description="Only return results from this time window."),
        ] = "PAST_24_HOURS",
        count: Annotated[
            Optional[int],
            Field(description="Number of results per source (10 to 200)."),
        ] = None,
    ) -> Union[dict, str]:
        """
        Search several sources at once with Desearch AI search and get links plus a summary.

        Args:
            prompt (str): The search prompt or query.
            tool (List[str]): Sources to search: web, hackernews, reddit, wikipedia, youtube, twitter, arxiv.
            date_filter (Optional[str]): Time window, for example PAST_24_HOURS or PAST_WEEK. Defaults to PAST_24_HOURS.
            count (Optional[int]): Number of results per source (10 to 200).

        Returns:
            dict | str: The search result, or an error string.

        """
        return asyncio_run(
            self.aai_search_tool(
                prompt=prompt, tool=tool, date_filter=date_filter, count=count
            )
        )

    async def atwitter_search_tool(
        self,
        query: Annotated[str, Field(description="The X (Twitter) search query.")],
        sort: Annotated[
            Literal["Top", "Latest"], Field(description="Sort order: Top or Latest.")
        ] = "Top",
        count: Annotated[
            int, Field(description="Number of posts to return (1 to 100).")
        ] = 10,
    ) -> Union[List[dict], str]:
        """
        Search posts on X (Twitter) using the Desearch API.

        Args:
            query (str): The X (Twitter) search query.
            sort (str, optional): Sort order, Top or Latest. Defaults to "Top".
            count (int, optional): Number of posts to return (1 to 100). Defaults to 10.

        Returns:
            List[dict] | str: The matching posts, or an error string.

        """
        try:
            async with Desearch(api_key=self.api_key) as client:
                result = await client.x_search(query=query, sort=sort, count=count)
            return _to_jsonable(result)
        except Exception as e:
            return str(e)

    def twitter_search_tool(
        self,
        query: Annotated[str, Field(description="The X (Twitter) search query.")],
        sort: Annotated[
            Literal["Top", "Latest"], Field(description="Sort order: Top or Latest.")
        ] = "Top",
        count: Annotated[
            int, Field(description="Number of posts to return (1 to 100).")
        ] = 10,
    ) -> Union[List[dict], str]:
        """
        Search posts on X (Twitter) using the Desearch API.

        Args:
            query (str): The X (Twitter) search query.
            sort (str, optional): Sort order, Top or Latest. Defaults to "Top".
            count (int, optional): Number of posts to return (1 to 100). Defaults to 10.

        Returns:
            List[dict] | str: The matching posts, or an error string.

        """
        return asyncio_run(
            self.atwitter_search_tool(query=query, sort=sort, count=count)
        )

    async def aweb_search_tool(
        self,
        query: Annotated[str, Field(description="The search query.")],
        num: Annotated[
            int, Field(description="Maximum number of results to return.")
        ] = 10,
        start: Annotated[
            int, Field(description="Number of results to skip, for pagination.")
        ] = 0,
    ) -> Union[List[dict], str]:
        """
        Run a web search using the Desearch API.

        Args:
            query (str): The search query.
            num (int, optional): Maximum number of results to return. Defaults to 10.
            start (int, optional): Number of results to skip, for pagination. Defaults to 0.

        Returns:
            List[dict] | str: The search results (title, link, snippet), or an error string.

        """
        try:
            async with Desearch(api_key=self.api_key) as client:
                result = await client.web_search(query=query, start=start)
            data = _to_jsonable(result)
            items = data.get("data", []) if isinstance(data, dict) else data
            return items[:num]
        except Exception as e:
            return str(e)

    def web_search_tool(
        self,
        query: Annotated[str, Field(description="The search query.")],
        num: Annotated[
            int, Field(description="Maximum number of results to return.")
        ] = 10,
        start: Annotated[
            int, Field(description="Number of results to skip, for pagination.")
        ] = 0,
    ) -> Union[List[dict], str]:
        """
        Run a web search using the Desearch API.

        Args:
            query (str): The search query.
            num (int, optional): Maximum number of results to return. Defaults to 10.
            start (int, optional): Number of results to skip, for pagination. Defaults to 0.

        Returns:
            List[dict] | str: The search results (title, link, snippet), or an error string.

        """
        return asyncio_run(self.aweb_search_tool(query=query, num=num, start=start))
