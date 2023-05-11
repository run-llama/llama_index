

from typing import List

from llama_index.prompts.base import Prompt
from llama_index.prompts.prompt_type import PromptType

# single select
PREFIX = """\
Your goal is to structure the user's query to match the request schema provided below.

<< Structured Request Schema >>
When responding use a markdown code snippet with a JSON object formatted in the \
following schema:

{schema_str}

The query string should contain only text that is expected to match the contents of \
documents. Any conditions in the filter should not be mentioned in the query as well.

Make sure that filters only refer to attributes that exist in the data source.
Make sure that filters take into account the descriptions of attributes and only make \
comparisons that are feasible given the type of data being stored.
Make sure that filters are only used as needed. If there are no filters that should be \
applied return {{}} for the filter value.\

"""

EXAMPLES ="""\
<< Example 1. >>
Data Source:
```json
{
    "content_info": "Lyrics of a song",
    "metadata_info": [
        {
            "name": "artist",
            "type": "string",
            "description": "Name of the song artist"
        },
        {
            "name": "genre",
            "type": "string",
            "description": "The song genre, one of \"pop\", \"rock\" or \"rap\""
        }
    }
}
```

User Query:
What are songs by Taylor Swift or Katy Perry in the dance pop genre

Structured Request:
```json
{
    "query": "teenager love",
    "filters": [
        {
            "key": "artist",
            "value": "Taylor Swift"
        },
        {
            "key": "artist",
            "value": "Katy Perry"
        },
        {
            "key": "genre",
            "value": "pop"
        }
    ]
}
```
""".replace(
    "{", "{{"
).replace(
    "}", "}}"
)


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

DEFAULT_VECTOR_STORE_QUERY_PROMPT_TMPL = PREFIX + EXAMPLES + SUFFIX


class VectorStoreQueryPrompt(Prompt):
    """Vector store query prompt."""

    prompt_type: PromptType = PromptType.VECTOR_STORE_QUERY
    input_variables: List[str] = ['schema_str', "info_str", "query_str"]


DEFAULT_VECTOR_STORE_QUERY_PROMPT = VectorStoreQueryPrompt(
    template=DEFAULT_VECTOR_STORE_QUERY_PROMPT_TMPL,
)
