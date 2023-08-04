# Usage Pattern

## Get Started

Configuring the response synthesizer for a query engine using `response_mode`:

```python
from llama_index.schema import Node, NodeWithScore
from llama_index.response_synthesizers import get_response_synthesizer

response_synthesizer = get_response_synthesizer(response_mode='compact')

response = response_synthesizer.synthesize(
  "query text", 
  nodes=[NodeWithScore(node=Node(text="text"), score=1.0), ..]
)
```

Or, more commonly, in a query engine after you've created an index:

```python
query_engine = index.as_query_engine(response_synthesizer=response_synthesizer)
response = query_engine.query("query_text")
```

```{tip}
To learn how to build an index, see [Index](/core_modules/data_modules/index/root.md)
```

## Configuring the Response Mode
Response synthesizers are typically specified through a `response_mode` kwarg setting.

Several response synthesizers are implemented already in LlamaIndex:

- `refine`: ***create and refine*** an answer by sequentially going through each retrieved text chunk. 
    This makes a separate LLM call per Node/retrieved chunk. The first chunk is used in a query using the 
    `text_qa_template` prompt. Then the answer and the next chunk (and the original question) are used 
    in another query with the `refine_template` prompt. And so on until all chunks have been parsed.
    Good for more detailed answers.
- `compact` (default): ***compact*** the prompt during each LLM call by stuffing as 
    many text (concatenated from the retrieved chunks) that can fit within the maximum prompt size. 
    If the text is too long to fit in one prompt, it is splitted in as many parts as needed 
    (using a `TokenTextSplitter`). Each text part is considered a "chunk" and is sent to the 
    ***create and refine*** synthesizer. In short, it is like `refine`, but with less LLM calls.
- `tree_summarize`: Given a set of text chunks and the query, recursively construct a tree 
    and return the root node as the response. All retrieved chunks/nodes are concatenated and then
    splitted to fit the context window using the `text_qa_template` prompt, resulting in as many new "chunks".
    Each of these chunks are queried against the `text_qa_template` prompt, giving as many answers. If there is
    only one answer, then it's the final answer. If there are more than one answer, these themselves are 
    considered as chunks and sent recursively to the `tree_summarize` process (concatenated/splitted-to-fit/queried).
    Good for summarization purposes.
- `simple_summarize`: Truncates all text chunks to fit into a single LLM prompt. Good for quick
    summarization purposes, but may lose detail due to truncation.
- `no_text`: Only runs the retriever to fetch the nodes that would have been sent to the LLM, 
    without actually sending them. Then can be inspected by checking `response.source_nodes`.
- `accumulate`: Given a set of text chunks and the query, apply the query to each text
    chunk while accumulating the responses into an array. Returns a concatenated string of all
    responses. Good for when you need to run the same query separately against each text
    chunk.
- `compact_accumulate`: The same as accumulate, but will "compact" each LLM prompt similar to
    `compact`, and run the same query against each text chunk.

## Custom Response Synthesizers

Each response synthesizer inherits from `llama_index.response_synthesizers.base.BaseSynthesizer`. The base API is extremely simple, which makes it easy to create your own response synthesizer.

Maybe you want to customize which template is used at each step in `tree_summarize`, or maybe a new research paper came out detailing a new way to generate a response to a query, you can create your own response synthesizer and plug it into any query engine or use it on it's own.

Below we show the `__init__()` function, as well as the two abstract methods that every response synthesizer must implement. The basic requirements are to process a query and text chunks, and return a string (or string generator) response.

```python
class BaseSynthesizer(ABC):
    """Response builder class."""

    def __init__(
        self,
        service_context: Optional[ServiceContext] = None,
        streaming: bool = False,
    ) -> None:
        """Init params."""
        self._service_context = service_context or ServiceContext.from_defaults()
        self._callback_manager = self._service_context.callback_manager
        self._streaming = streaming

    @abstractmethod
    def get_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        """Get response."""
        ...

    @abstractmethod
    async def aget_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        """Get response."""
        ...
```
