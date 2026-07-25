# LlamaIndex Postprocessor Integration: Distil

Reversibly compress retrieved node text with [distil](https://github.com/dshakes/distil)
before it reaches the LLM synthesizer.

`DistilNodePostprocessor` runs each node's text through distil's line digest: it keeps the
head, the tail, and every salient line, and replaces the dropped middle with a single
`<< +N lines, handle=XXXXXXXX >>` marker. The original text is written to distil's local
handle store, so the exact bytes stay recoverable via `distil.mcp_server.load_restore(handle)`
or distil's `distil_expand` MCP tool.

Short nodes — anything at or under head + tail + 1 lines — pass through untouched.

## Install

```bash
pip install llama-index-postprocessor-distil
```

## Usage

```python
from llama_index.core import VectorStoreIndex
from llama_index.postprocessor.distil import DistilNodePostprocessor

query_engine = index.as_query_engine(
    node_postprocessors=[DistilNodePostprocessor()],
)
```

`query_aware` (default `True`) passes the query's terms to distil as salience intent, so
lines naming what the query asks for are pinned. It only ever widens the set of lines kept.

## What it is good at

distil's digest is tuned for structured, line-oriented content — logs, code, command and
tool output. On those it drops a large fraction of the text while keeping every line that
carries a decision. On pure prose chunks it still keeps head, tail and salient lines, but
the reduction is smaller.

Compression only. Decision-equivalence between compressed and full context is certified
separately and offline by `distil bench` / `distil validate`.

## License

MIT
