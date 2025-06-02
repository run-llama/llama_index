"""Autoretriever prompts."""

from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.prompts.prompt_type import PromptType
from llama_index.core.vector_stores.types import (
    FilterOperator,
    MetadataFilter,
    MetadataInfo,
    VectorStoreInfo,
    VectorStoreQuerySpec,
)

# NOTE: these prompts are inspired from langchain's self-query prompt,
# and adapted to our use case.
# https://github.com/hwchase17/langchain/tree/main/langchain/chains/query_constructor/prompt.py


PREFIX = """\
Your goal is to structure the user's query to match the request schema provided below.

<< Structured Request Schema >>
When responding use a markdown code snippet with a JSON object formatted in the \
following schema:

{schema_str}

The query string should contain only text that is expected to match the contents of \
documents. Any conditions in the filter should not be mentioned in the query as well.

Make sure that filters only refer to attributes that exist in the data source.
Make sure that filters take into account the descriptions of attributes.
Make sure that filters are only used as needed. If there are no filters that should be \
applied return [] for the filter value.\

If the user's query explicitly mentions number of documents to retrieve, set top_k to \
that number, otherwise do not set top_k.

"""

example_info_1 = VectorStoreInfo(
    content_info="Lyrics of a song",
    metadata_info=[
        MetadataInfo(name="artist", type="str", description="Name of the song artist"),
        MetadataInfo(
            name="genre",
            type="str",
            description='The song genre, one of "pop", "rock" or "rap"',
        ),
    ],
)

example_query_1 = "What are songs by Taylor Swift or Katy Perry about teenage romance in the dance pop genre"

example_output_1 = VectorStoreQuerySpec(
    query="what songs are about teenager love",
    filters=[
        MetadataFilter(key="artist", value="Taylor Swift"),
        MetadataFilter(key="artist", value="Katy Perry"),
        MetadataFilter(key="genre", value="pop"),
    ],
)

example_info_2 = VectorStoreInfo(
    content_info="Classic literature",
    metadata_info=[
        MetadataInfo(name="author", type="str", description="Author name"),
        MetadataInfo(
            name="book_title",
            type="str",
            description="Book title",
        ),
        MetadataInfo(
            name="year",
            type="int",
            description="Year Published",
        ),
        MetadataInfo(
            name="pages",
            type="int",
            description="Number of pages",
        ),
        MetadataInfo(
            name="summary",
            type="str",
            description="A short summary of the book",
        ),
    ],
)

example_query_2 = "What are some books by Jane Austen published after 1813 that explore the theme of marriage for social standing?"

example_output_2 = VectorStoreQuerySpec(
    query="What books related to theme of marriage for social standing?",
    filters=[
        MetadataFilter(key="year", value="1813", operator=FilterOperator.GT),
        MetadataFilter(key="author", value="Jane Austen"),
    ],
)

EXAMPLES = f"""\
<< Example 1. >>
Data Source:
```json
{example_info_1.model_dump_json(indent=4)}
```

User Query:
{example_query_1}

Structured Request:
```json
{example_output_1.model_dump_json()}


<< Example 2. >>
Data Source:
```json
{example_info_2.model_dump_json(indent=4)}
```

User Query:
{example_query_2}

Structured Request:
```json
{example_output_2.model_dump_json()}

```
""".replace("{", "{{").replace("}", "}}")


SUFFIX = """
<< Example 3. >>
Data Source:
```json
{info_str}
```

User Query:
{query_str}

Structured Request:
"""

DEFAULT_VECTARA_QUERY_PROMPT_TMPL = PREFIX + EXAMPLES + SUFFIX


# deprecated, kept for backwards compatibility
"""Vector store query prompt."""
VectorStoreQueryPrompt = PromptTemplate

DEFAULT_VECTARA_QUERY_PROMPT = PromptTemplate(
    template=DEFAULT_VECTARA_QUERY_PROMPT_TMPL,
    prompt_type=PromptType.VECTOR_STORE_QUERY,
)
