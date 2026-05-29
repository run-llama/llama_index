# vecr-compress Node Postprocessor

[vecr-compress](https://github.com/h2cker/vecr) is an open-source LLM context compressor with a deterministic retention contract. In a RAG pipeline, it compresses retrieved nodes to fit a token budget while guaranteeing that structured tokens — order IDs, dates, URLs, emails, code references — are never dropped. Filler prose is hard-dropped; remaining content is scored by entropy and structural signals (digits, braces, capitalization) and packed greedily into the token budget.

## Overview

| Property | Value |
|---|---|
| Package | `vecr-compress` (install with `[llamaindex]` extra) |
| Import path | `vecr_compress.adapters.llamaindex` |
| License | Apache 2.0 |
| Python | 3.10+ |
| Retention contract | Deterministic (regex whitelist, 13 built-in rules) |
| Embedding required | No (lexical scoring) |

## Installation

```bash
pip install 'vecr-compress[llamaindex]'
```

> The older `llama-index-postprocessor-vecr` shim package (0.1.0, namespaced `llama_index.postprocessor.vecr`) re-exports this same postprocessor and is now deprecated in favour of the `[llamaindex]` extra on the core package. Existing installs keep working; new projects should use the extra.

## Usage

```python
from llama_index.core.schema import NodeWithScore, TextNode
from vecr_compress.adapters.llamaindex import VecrNodePostprocessor

processor = VecrNodePostprocessor(budget_tokens=1500)

kept = processor.postprocess_nodes(
    nodes,                          # list[NodeWithScore]
    query_str="the user's question",
)
```

Nodes containing retention-matching content (IDs, amounts, dates) are kept even at aggressive budgets. Filler-only nodes are dropped.

## Advanced usage

Access compression telemetry without mutating nodes:

```python
result = processor.compress_with_report(nodes, query_str="refund amount")
print(f"Ratio: {result.ratio:.1%}, pinned: {len(result.retained_matches)} facts")
for seg in result.dropped_segments:
    print("dropped:", seg["text"][:60])
```

### Opt-in question-aware scoring (v0.1.3+)

RAG over long prose benefits most from blending question relevance into the heuristic scorer. On the HotpotQA dev probe (N=100, distractor split, real multi-hop NL-QA) this lifts supporting-fact survival at ratio 0.5 by **+9.9 percentage points** (58.0% → 67.9%). Off by default so the deterministic retention contract stays the loud promise — see [BENCHMARK.md](https://github.com/h2cker/vecr/blob/main/docs/BENCHMARK.md#hotpotqa-spike--where-the-synthetic-bench-hits-its-ceiling) for methodology.

```python
processor = VecrNodePostprocessor(
    budget_tokens=1500,
    use_question_relevance=True,
)
kept = processor.postprocess_nodes(nodes, query_str="refund amount for order ORD-99172")
```

`query_str` (or `query_bundle.query_str`) is threaded through as the question when blending is on.

### Custom retention rules

```python
import re
from vecr_compress import RetentionRule, DEFAULT_RULES

rules = DEFAULT_RULES.with_extra([
    RetentionRule(name="sku", pattern=re.compile(r"\bSKU-[A-Z0-9]{6}\b")),
])
processor = VecrNodePostprocessor(budget_tokens=1500, retention_rules=rules)
```

## Chaining with a reranker

vecr-compress compresses nodes to fit a token budget; it is not a semantic reranker. The recommended pattern is: rerank first, then compress.

```python
from llama_index.postprocessor.cohere_rerank import CohereRerank
from vecr_compress.adapters.llamaindex import VecrNodePostprocessor

pipeline = [
    CohereRerank(top_n=10),
    VecrNodePostprocessor(budget_tokens=2000, use_question_relevance=True),
]
```

## API reference

See the [vecr-compress GitHub repo](https://github.com/h2cker/vecr) for full API docs, the retention contract specification ([RETENTION.md](https://github.com/h2cker/vecr/blob/main/RETENTION.md)), and changelog.
