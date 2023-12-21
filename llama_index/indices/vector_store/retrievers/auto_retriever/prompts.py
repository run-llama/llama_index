"""Autoretriever prompts."""


from llama_index.prompts.base import PromptTemplate
from llama_index.prompts.prompt_type import PromptType
from llama_index.vector_stores.types import (
    ExactMatchFilter,
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

example_info = VectorStoreInfo(
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

example_query = "What are songs by Taylor Swift or Katy Perry in the dance pop genre"

example_output = VectorStoreQuerySpec(
    query="teenager love",
    filters=[
        ExactMatchFilter(key="artist", value="Taylor Swift"),
        ExactMatchFilter(key="artist", value="Katy Perry"),
        ExactMatchFilter(key="genre", value="pop"),
    ],
)

EXAMPLES = f"""\
<< Example 1. >>
Data Source:
```json
{example_info.json(indent=4)}
```

User Query:
{example_query}

Structured Request:
```json
{example_output.json()}
```
""".replace(
    "{", "{{"
).replace(
    "}", "}}"
)


SUFFIX = """
<< Example 2. >>
Data Source:
```json
{info_str}
```

User Query:
{query_str}

Structured Request:
"""

DEFAULT_VECTOR_STORE_QUERY_PROMPT_TMPL = PREFIX + EXAMPLES + SUFFIX


# deprecated, kept for backwards compatibility
"""Vector store query prompt."""
VectorStoreQueryPrompt = PromptTemplate

DEFAULT_VECTOR_STORE_QUERY_PROMPT = PromptTemplate(
    template=DEFAULT_VECTOR_STORE_QUERY_PROMPT_TMPL,
    prompt_type=PromptType.VECTOR_STORE_QUERY,
)
