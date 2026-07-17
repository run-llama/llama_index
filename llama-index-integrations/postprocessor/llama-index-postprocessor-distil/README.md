# LlamaIndex Postprocessor Integration: Distil

`DistilNodePostprocessor` reversibly compresses retrieved node text with
[distil](https://github.com/dshakes/distil) before it reaches the LLM synthesizer.

It runs each node's text through distil's line digest — keeping the head, the tail,
and every salient line, and replacing the dropped middle with a single
`<< +N lines, handle=XXXXXXXX >>` marker. The original text is written to distil's
local handle store, so the exact bytes are recoverable via distil's `distil_expand`
MCP tool or `distil.mcp_server.load_restore(handle)`. Short nodes pass through
unchanged.

## Installation

```bash
pip install llama-index-postprocessor-distil
```

## Usage

```python
from llama_index.postprocessor.distil import DistilNodePostprocessor

postprocessor = DistilNodePostprocessor()

# in a query engine
query_engine = index.as_query_engine(node_postprocessors=[postprocessor])
```

`query_aware` (default `True`) passes the query's terms to distil so lines naming what
the query asks for are pinned — this only ever widens the kept set, so it never drops
an answer.
